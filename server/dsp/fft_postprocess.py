"""Per-bin port of features.AutoScaler for the FFT log bins.

Mirrors the L/M/H pipeline (smoother → asymmetric peak follower → soft gate
→ tanh → strength blend) so the FFT visualizer and any OSC consumer see the
same semantics on both pipelines: band-scoped (L/M/H) and per-bin (FFT).

Output is in [0, 1]; the WS broadcaster and OSC sender pick between this
"processed" stream and the raw dB stream based on `cfg.fft.send_raw_db`.

Threading: this object is created and configured from the asyncio loop
(via App / Dispatcher) but `process()` is called from the FFT worker
thread. Cross-thread mutations (band edges, taus, autoscale params) go
through `update_*` methods that take `_lock`; `process()` also takes
`_lock`. The lock is held for at most a few hundred microseconds (n_bins
~128, vector ops in numpy), and reconfigure events are rare.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from scipy.ndimage import gaussian_filter1d

EPS = 1e-12

# FFT log-bin dB values run materially higher than the equivalent
# time-domain band RMS dB for the same source: pure-tone energy concentrated
# in one rfft bin shows ~50–55 dB above its time-RMS, while broadband signals
# only differ by ~25 dB. We subtract this single calibration constant before
# the noise gate / peak follower so a given `noise_floor` value gates similarly
# on both pipelines. It can't be made exact (the relationship is signal-
# dependent) but a fixed offset lands the two pipelines in the same ballpark.
FFT_TO_RMS_CALIBRATION_DB = 40.0


class FFTPostProcessor:
    def __init__(self, n_bins: int, f_min: float, sr: float,
                 bands: dict, tau: dict,
                 tau_release_s: float, noise_floor: float, strength: float,
                 db_floor: float = -60.0, db_ceiling: float = 0.0,
                 hop_period_s: float = 512 / 48000.0,
                 tau_attack_s: float = 0.05,
                 peak_smear_oct: float = 0.3):
        self._lock = threading.Lock()
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self.sr = float(sr)
        self.bands = bands  # {"low": {"lo_hz","hi_hz"}, ...}
        self.tau = tau      # {"low","mid","high"}
        self.tau_release_s = float(tau_release_s)
        self.noise_floor = max(0.0, float(noise_floor))
        self.strength = max(0.0, min(1.0, float(strength)))
        self.db_floor = float(db_floor)
        self.db_ceiling = float(db_ceiling)
        self.hop_period_s = float(hop_period_s)
        self.tau_attack_s = float(tau_attack_s)
        self.peak_smear_oct = max(0.0, float(peak_smear_oct))
        self._allocate()

    # ----------------- allocation -----------------
    def _allocate(self) -> None:
        n = self.n_bins
        self.smooth_lin = np.zeros(n, dtype=np.float64)
        self.peak_lin = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.peak_smoothed = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.interp_db = np.zeros(n, dtype=np.float64)
        self.processed = np.zeros(n, dtype=np.float32)
        # scratch
        self._lin = np.zeros(n, dtype=np.float64)
        self._diff = np.zeros(n, dtype=np.float64)
        self._alpha = np.zeros(n, dtype=np.float64)
        self.tau_per_bin = self._build_per_bin_tau()
        self._recompute_alphas()
        self._recompute_smear()
        self._warmed = False

    def _recompute_smear(self) -> None:
        """Convert peak_smear_oct → Gaussian sigma in *bin index* units, and
        precompute the per-bin edge-normalization weights.

        The log-bin axis covers log2(fmax/fmin) octaves across n_bins, so
        bins-per-octave is constant — we just scale the requested octave
        sigma by it.

        `_smear_norm[i]` is the integral of the Gaussian kernel that lies
        inside [0, n_bins) when centered on bin `i`. We use it to renormalize
        a constant/zero-padded convolution so bins near the edges (especially
        the low-frequency end) are a proper weighted mean of just the
        in-range neighbors instead of being contaminated by reflected
        upper-bin values, which produced visible decay artefacts on the
        leftmost bins.
        """
        if self.peak_smear_oct <= 0.0:
            self._smear_sigma_bins = 0.0
            self._smear_norm = None
            return
        f_max = self.sr / 2.0
        if f_max <= self.f_min:
            self._smear_sigma_bins = 0.0
            self._smear_norm = None
            return
        bins_per_octave = self.n_bins / max(1e-6, math.log2(f_max / self.f_min))
        self._smear_sigma_bins = float(self.peak_smear_oct * bins_per_octave)
        ones = np.ones(self.n_bins, dtype=np.float64)
        self._smear_norm = gaussian_filter1d(
            ones, sigma=self._smear_sigma_bins, mode="constant", cval=0.0
        )

    def _build_per_bin_tau(self) -> np.ndarray:
        # Piecewise-linear interpolation of (log10(f_center), tau_band) anchored
        # at each L/M/H band's geometric-mean center. Outside the [low_center,
        # high_center] range, clamp to the nearest band's tau.
        anchors = []
        for name in ("low", "mid", "high"):
            b = self.bands[name]
            cen = math.sqrt(float(b["lo_hz"]) * float(b["hi_hz"]))
            anchors.append((math.log10(max(cen, 1e-3)), float(self.tau[name])))
        anchors.sort(key=lambda x: x[0])
        log_fmin = math.log10(max(self.f_min, 1e-3))
        f_max = self.sr / 2.0
        log_span = max(1e-6, math.log10(f_max) - log_fmin)
        out = np.zeros(self.n_bins, dtype=np.float64)
        for i in range(self.n_bins):
            log_f = log_fmin + ((i + 0.5) / self.n_bins) * log_span
            if log_f <= anchors[0][0]:
                out[i] = anchors[0][1]
            elif log_f >= anchors[-1][0]:
                out[i] = anchors[-1][1]
            else:
                a, b = (anchors[0], anchors[1]) if log_f <= anchors[1][0] else (anchors[1], anchors[2])
                t = (log_f - a[0]) / max(1e-9, b[0] - a[0])
                out[i] = a[1] + t * (b[1] - a[1])
        return out

    def _recompute_alphas(self) -> None:
        h = self.hop_period_s
        self.alpha_smooth = (1.0 - np.exp(-h / np.maximum(self.tau_per_bin, 1e-3))).astype(np.float64)
        self.alpha_attack = 1.0 - math.exp(-h / max(self.tau_attack_s, 1e-3))
        self.alpha_release = 1.0 - math.exp(-h / max(self.tau_release_s, 1e-3))

    # ----------------- live retune (asyncio thread) -----------------
    def reset(self) -> None:
        with self._lock:
            self.smooth_lin.fill(0.0)
            self.peak_lin.fill(max(self.noise_floor, EPS))
            self._warmed = False

    def reconfigure(self, *, n_bins=None, f_min=None, sr=None,
                    bands=None, tau=None, hop_period_s=None,
                    db_floor=None, db_ceiling=None) -> None:
        with self._lock:
            if n_bins is not None: self.n_bins = int(n_bins)
            if f_min is not None: self.f_min = float(f_min)
            if sr is not None: self.sr = float(sr)
            if bands is not None: self.bands = bands
            if tau is not None: self.tau = tau
            if hop_period_s is not None: self.hop_period_s = float(hop_period_s)
            if db_floor is not None: self.db_floor = float(db_floor)
            if db_ceiling is not None: self.db_ceiling = float(db_ceiling)
            self._allocate()

    def update_smoothing(self, tau: dict) -> None:
        with self._lock:
            self.tau = tau
            self.tau_per_bin = self._build_per_bin_tau()
            self._recompute_alphas()

    def update_bands(self, bands: dict) -> None:
        with self._lock:
            self.bands = bands
            self.tau_per_bin = self._build_per_bin_tau()
            self._recompute_alphas()

    def update_smear(self, peak_smear_oct: float) -> None:
        with self._lock:
            self.peak_smear_oct = max(0.0, float(peak_smear_oct))
            self._recompute_smear()

    def update_autoscale(self, *, tau_attack_s=None, tau_release_s=None,
                         noise_floor=None, strength=None) -> None:
        with self._lock:
            if tau_attack_s is not None:
                self.tau_attack_s = float(tau_attack_s)
            if tau_release_s is not None:
                self.tau_release_s = float(tau_release_s)
            if noise_floor is not None:
                self.noise_floor = max(0.0, float(noise_floor))
                np.maximum(self.peak_lin, max(self.noise_floor, EPS), out=self.peak_lin)
            if strength is not None:
                self.strength = max(0.0, min(1.0, float(strength)))
            self._recompute_alphas()

    # ----------------- hot path (FFT worker thread) -----------------
    def process(self, db_in: np.ndarray) -> np.ndarray:
        """Run the pipeline on `db_in` (float32, length n_bins, raw wire dB
        with -1000 sentinels for empty log bins). Returns the float32
        `processed` array (in [0, 1]). The returned array is owned by this
        instance — caller must copy before mutating.
        """
        with self._lock:
            n = self.n_bins
            # 1. Sentinel interpolation (vectorized).
            valid_mask = db_in >= -500.0
            valid_idx = np.where(valid_mask)[0]
            if valid_idx.size == 0:
                self.interp_db.fill(-120.0)
            else:
                self.interp_db[:] = np.interp(
                    np.arange(n),
                    valid_idx,
                    db_in[valid_idx].astype(np.float64),
                )

            # 2. dB → linear with calibration shift so the same noise_floor
            #    gates similarly to the L/M/H AutoScaler.
            np.subtract(self.interp_db, FFT_TO_RMS_CALIBRATION_DB, out=self._lin)
            np.divide(self._lin, 20.0, out=self._lin)
            np.power(10.0, self._lin, out=self._lin)

            # 3. Per-bin EMA smoothing.
            if not self._warmed:
                np.copyto(self.smooth_lin, self._lin)
                np.maximum(self._lin, max(self.noise_floor, EPS), out=self.peak_lin)
                self._warmed = True
            else:
                np.subtract(self._lin, self.smooth_lin, out=self._diff)
                np.multiply(self._diff, self.alpha_smooth, out=self._diff)
                np.add(self.smooth_lin, self._diff, out=self.smooth_lin)

            # 4. Asymmetric peak follower.
            np.subtract(self.smooth_lin, self.peak_lin, out=self._diff)
            rising = self._diff > 0
            self._alpha.fill(self.alpha_release)
            self._alpha[rising] = self.alpha_attack
            np.multiply(self._diff, self._alpha, out=self._diff)
            np.add(self.peak_lin, self._diff, out=self.peak_lin)

            # 4b. Spatial smear of the peak across bins. A single-frequency
            # tone otherwise drives ITS bin's peak high enough to fully self-
            # normalize, making the tone read smaller than its neighbors.
            # Smearing the peak across a Gaussian neighborhood (in log-bin
            # space, i.e. fixed octaves) keeps the local frequency contour
            # intact while still removing the long-term spectral envelope.
            # Edge handling: zero-pad the convolution and divide by the
            # precomputed per-bin kernel-mass `_smear_norm`. This makes the
            # output a proper weighted mean over only the in-range neighbors,
            # so the leftmost bins aren't pulled around by reflected
            # upper-bin values during decay.
            if self._smear_sigma_bins > 0.0:
                gaussian_filter1d(self.peak_lin, sigma=self._smear_sigma_bins,
                                  output=self.peak_smoothed, mode="constant",
                                  cval=0.0)
                np.divide(self.peak_smoothed, self._smear_norm,
                          out=self.peak_smoothed)
            else:
                np.copyto(self.peak_smoothed, self.peak_lin)

            # 5. AutoScaler core: subtract floor, divide by max(peak, floor), tanh.
            nf = max(self.noise_floor, EPS)
            denom = np.maximum(self.peak_smoothed, nf)
            gated = np.maximum(0.0, self.smooth_lin - self.noise_floor)
            scaled = np.tanh(gated / denom)

            # 6. Strength blend with raw-dB-mapped baseline.
            s = self.strength
            if s >= 1.0:
                np.copyto(self.processed, scaled.astype(np.float32))
            else:
                disp_floor = self.db_floor
                disp_span = max(1.0, self.db_ceiling - disp_floor)
                # noise floor expressed in WIRE dB units (post-calibration):
                noise_floor_wire_db = (
                    20.0 * math.log10(max(self.noise_floor, EPS)) + FFT_TO_RMS_CALIBRATION_DB
                )
                raw = np.clip((self.interp_db - disp_floor) / disp_span, 0.0, 1.0)
                raw[self.interp_db < noise_floor_wire_db] = 0.0
                blended = s * scaled + (1.0 - s) * raw
                self.processed[:] = blended.astype(np.float32)

            return self.processed
