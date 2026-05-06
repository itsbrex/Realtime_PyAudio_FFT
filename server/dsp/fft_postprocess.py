"""Per-bin port of features.AutoScaler for the FFT log bins.

Mirrors the L/M/H pipeline (pre-tilt → smoother → asymmetric peak follower
→ peak smear → soft gate → tanh → strength blend) so a single set of
controls tunes both pipelines in lockstep.

Output is in [0, 1]; the WS broadcaster and OSC sender pick between this
"processed" stream and the raw dB stream based on `cfg.fft.send_raw_db`.

Threading: configured from the asyncio loop, but `process()` runs on the
FFT worker thread. Cross-thread mutations go through `update_*` methods
that take `_lock`; `process()` also takes `_lock`. The lock is held for
hundreds of µs per hop (n_bins ~128, all-vector ops) and reconfigure
events are rare.
"""
from __future__ import annotations

import math
import threading

import numpy as np
from scipy.ndimage import gaussian_filter1d

EPS = 1e-12

# Reference frequency for the spectral tilt: per-bin offset is zero AT this
# frequency and grows ±tilt_db_per_oct dB per octave. 1 kHz puts the default
# tilt in the middle of the musical range — bins below are pulled down, bins
# above are pushed up — so a single global noise gate fires uniformly.
TILT_REF_HZ = 1000.0

# When `peak_smear_oct` is large relative to the spectrum width, the per-bin
# kernel mass `_smear_norm` near the edges can be tiny; dividing by it to
# normalise edge bins amplifies numerical noise into spurious peaks. Floor
# the divisor at this fraction of the kernel's centre mass.
SMEAR_NORM_MIN = 0.1


class FFTPostProcessor:
    def __init__(self, n_bins: int, f_min: float, sr: float,
                 bands: dict, tau: dict,
                 tau_release_s: float, noise_floor: float, strength: float,
                 db_floor: float = -60.0, db_ceiling: float = 0.0,
                 hop_period_s: float = 512 / 48000.0,
                 tau_attack_s: float = 0.05,
                 peak_smear_oct: float = 0.3,
                 tilt_db_per_oct: float = 3.0):
        self._lock = threading.Lock()
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self.sr = float(sr)
        self.bands = bands
        self.tau = tau
        self.tau_release_s = float(tau_release_s)
        self.noise_floor = max(0.0, float(noise_floor))
        self.strength = max(0.0, min(1.0, float(strength)))
        self.db_floor = float(db_floor)
        self.db_ceiling = float(db_ceiling)
        self.hop_period_s = float(hop_period_s)
        self.tau_attack_s = float(tau_attack_s)
        self.peak_smear_oct = max(0.0, float(peak_smear_oct))
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._allocate()

    # ----------------- allocation -----------------
    def _allocate(self) -> None:
        n = self.n_bins
        self.smooth_lin = np.zeros(n, dtype=np.float64)
        self.peak_lin = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.peak_smoothed = np.full(n, max(self.noise_floor, EPS), dtype=np.float64)
        self.interp_db = np.zeros(n, dtype=np.float64)
        self.processed = np.zeros(n, dtype=np.float32)
        # scratch (float64), repurposed across pipeline stages
        self._lin = np.zeros(n, dtype=np.float64)
        self._diff = np.zeros(n, dtype=np.float64)
        self._alpha = np.zeros(n, dtype=np.float64)
        self._scratch = np.zeros(n, dtype=np.float64)
        self._mask = np.zeros(n, dtype=bool)
        self._x = np.arange(n, dtype=np.float64)
        # cached on first hop (bin layout is stable until reconfigure)
        self._vi: np.ndarray | None = None         # int valid indices
        self._vi_x: np.ndarray | None = None       # float64 view of _vi for np.interp
        self._vi_db: np.ndarray | None = None      # float64 buffer for db_in[_vi]
        self.tau_per_bin = self._build_per_bin_tau()
        self.tilt_db = self._build_tilt_db()
        self._recompute_alphas()
        self._recompute_smear()
        self._warmed = False

    def _build_tilt_db(self) -> np.ndarray:
        n = self.n_bins
        log_fmin = math.log10(max(self.f_min, 1e-3))
        f_max = self.sr / 2.0
        log_span = max(1e-6, math.log10(f_max) - log_fmin)
        idx = np.arange(n, dtype=np.float64)
        log_f = log_fmin + ((idx + 0.5) / n) * log_span
        f_center = np.power(10.0, log_f)
        return (self.tilt_db_per_oct * np.log2(f_center / TILT_REF_HZ)).astype(np.float64)

    def _recompute_smear(self) -> None:
        """Convert peak_smear_oct → Gaussian sigma in *bin index* units, plus
        the per-bin edge-normalisation weights (clamped at SMEAR_NORM_MIN of
        the centre mass to avoid amplifying noise in low-mass edge bins).
        """
        if self.peak_smear_oct <= 0.0:
            self._smear_sigma_bins = 0.0
            self._smear_norm_clamped = None
            return
        f_max = self.sr / 2.0
        if f_max <= self.f_min:
            self._smear_sigma_bins = 0.0
            self._smear_norm_clamped = None
            return
        bins_per_octave = self.n_bins / max(1e-6, math.log2(f_max / self.f_min))
        self._smear_sigma_bins = float(self.peak_smear_oct * bins_per_octave)
        ones = np.ones(self.n_bins, dtype=np.float64)
        smear_norm = gaussian_filter1d(
            ones, sigma=self._smear_sigma_bins, mode="constant", cval=0.0
        )
        floor_val = SMEAR_NORM_MIN * float(np.max(smear_norm))
        self._smear_norm_clamped = np.maximum(smear_norm, floor_val)

    def _build_per_bin_tau(self) -> np.ndarray:
        anchors = []
        for name in ("low", "mid", "high"):
            b = self.bands[name]
            cen = math.sqrt(float(b["lo_hz"]) * float(b["hi_hz"]))
            anchors.append((math.log10(max(cen, 1e-3)), float(self.tau[name])))
        anchors.sort(key=lambda x: x[0])
        log_fmin = math.log10(max(self.f_min, 1e-3))
        f_max = self.sr / 2.0
        log_span = max(1e-6, math.log10(f_max) - log_fmin)
        log_f = log_fmin + ((np.arange(self.n_bins, dtype=np.float64) + 0.5) / self.n_bins) * log_span
        xp = np.array([a[0] for a in anchors], dtype=np.float64)
        fp = np.array([a[1] for a in anchors], dtype=np.float64)
        # np.interp clamps to fp[0]/fp[-1] outside [xp[0], xp[-1]] — exactly the
        # nearest-band behaviour we want.
        return np.interp(log_f, xp, fp)

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

    def update_tilt(self, tilt_db_per_oct: float) -> None:
        with self._lock:
            self.tilt_db_per_oct = float(tilt_db_per_oct)
            self.tilt_db = self._build_tilt_db()

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
        `processed` array (in [0, 1]). Caller must copy before mutating.
        """
        with self._lock:
            n = self.n_bins

            # 1. Sentinel interpolation. Bin layout is stable until reconfigure,
            #    so cache the valid-index array on first hop.
            if self._vi is None:
                vi = np.where(db_in >= -500.0)[0]
                self._vi = vi
                self._vi_x = vi.astype(np.float64)
                # Match db_in's dtype so np.take with out= passes the safe-cast rule.
                self._vi_db = np.empty(vi.shape[0], dtype=db_in.dtype)
            vi = self._vi
            if vi.size == 0:
                self.interp_db.fill(-120.0)
            elif vi.size == n:
                self.interp_db[:] = db_in   # fast f32→f64 copy, no gaps
            else:
                np.take(db_in, vi, out=self._vi_db)
                self.interp_db[:] = np.interp(self._x, self._vi_x, self._vi_db)

            # 2. Pre-tilt (in dB) + dB→linear into _lin.
            #    interp_db is the UNTILTED raw spectrum and is reused below
            #    for the strength<1 baseline; do not mutate it.
            np.add(self.interp_db, self.tilt_db, out=self._lin)
            np.multiply(self._lin, 1.0 / 20.0, out=self._lin)
            np.power(10.0, self._lin, out=self._lin)

            # 3. Per-bin EMA smoothing.
            if not self._warmed:
                np.copyto(self.smooth_lin, self._lin)
                # Don't saturate the peak follower to the first sample —
                # initialise to noise_floor and let attack chase up over
                # tau_attack_s. Avoids one transient locking in tau_release_s
                # of suppression on every other bin.
                self.peak_lin.fill(max(self.noise_floor, EPS))
                self._warmed = True
            else:
                np.subtract(self._lin, self.smooth_lin, out=self._diff)
                np.multiply(self._diff, self.alpha_smooth, out=self._diff)
                np.add(self.smooth_lin, self._diff, out=self.smooth_lin)

            # 4. Asymmetric peak follower (in-place, preallocated buffers).
            np.subtract(self.smooth_lin, self.peak_lin, out=self._diff)
            np.greater(self._diff, 0.0, out=self._mask)
            self._alpha.fill(self.alpha_release)
            self._alpha[self._mask] = self.alpha_attack
            np.multiply(self._diff, self._alpha, out=self._diff)
            np.add(self.peak_lin, self._diff, out=self.peak_lin)

            # 4b. Spatial Gaussian smear of the peak across log-frequency bins.
            if self._smear_sigma_bins > 0.0:
                gaussian_filter1d(self.peak_lin, sigma=self._smear_sigma_bins,
                                  output=self.peak_smoothed, mode="constant",
                                  cval=0.0)
                np.divide(self.peak_smoothed, self._smear_norm_clamped,
                          out=self.peak_smoothed)
            else:
                np.copyto(self.peak_smoothed, self.peak_lin)

            # 5. AutoScaler core: tanh( max(0, smooth - nf) / max(peak, nf) ).
            #    In-place via _diff (gated/scaled) and _alpha (denom).
            nf = max(self.noise_floor, EPS)
            np.subtract(self.smooth_lin, nf, out=self._diff)
            np.maximum(self._diff, 0.0, out=self._diff)
            np.maximum(self.peak_smoothed, nf, out=self._alpha)
            np.divide(self._diff, self._alpha, out=self._diff)
            np.tanh(self._diff, out=self._diff)

            # 6. Strength blend.
            s = self.strength
            if s >= 1.0:
                self.processed[:] = self._diff   # f64 → f32
                return self.processed

            # Raw dB-mapped baseline from UNTILTED interp_db, [0, 1]; gate-zero
            # bins whose untilted value is below the noise gate.
            disp_floor = self.db_floor
            disp_span = max(1.0, self.db_ceiling - disp_floor)
            np.subtract(self.interp_db, disp_floor, out=self._scratch)
            np.multiply(self._scratch, 1.0 / disp_span, out=self._scratch)
            np.clip(self._scratch, 0.0, 1.0, out=self._scratch)
            noise_floor_db = 20.0 * math.log10(nf)
            np.less(self.interp_db, noise_floor_db, out=self._mask)
            self._scratch[self._mask] = 0.0

            # blended = s·scaled + (1-s)·baseline
            np.multiply(self._diff, s, out=self._diff)
            np.multiply(self._scratch, 1.0 - s, out=self._scratch)
            np.add(self._diff, self._scratch, out=self._diff)
            self.processed[:] = self._diff
            return self.processed
