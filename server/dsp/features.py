"""Per-band exponential smoother + rolling auto-scaler.

Mirrors `dsp/fft_postprocess.py`: tilt is applied as a per-band LINEAR
pre-multiplier on the smoothed RMS, the noise floor is a single global
linear value, and the strength<1 baseline is a dB-mapped [0,1] curve
(with the same gate-zeroing). One `noise_floor` slider therefore gates
both pipelines in lockstep.
"""
from __future__ import annotations

import math

import numpy as np


def block_rms(x: np.ndarray) -> float:
    """RMS of a 1-D buffer; returns Python float.

    np.dot(x, x) is a single BLAS sdot/ddot call with no per-block temporary,
    vs np.square(...) which allocates a length-blocksize array each call.
    """
    return math.sqrt(float(np.dot(x, x)) / x.size)


class ExpSmoother:
    """Per-band one-pole exponential smoother with per-band tau.

    alpha_band = 1 - exp(-blocksize / (sr * tau_band)).
    """

    def __init__(self, sr: float, blocksize: int, tau: dict):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self._values = np.zeros(3, dtype=np.float64)
        self.set_tau(tau)

    def set_tau(self, tau: dict) -> None:
        dt = self.blocksize / self.sr
        self._alpha = np.array(
            [
                1.0 - math.exp(-dt / max(float(tau["low"]),  1e-3)),
                1.0 - math.exp(-dt / max(float(tau["mid"]),  1e-3)),
                1.0 - math.exp(-dt / max(float(tau["high"]), 1e-3)),
            ],
            dtype=np.float64,
        )

    def update(self, rms_lo: float, rms_md: float, rms_hi: float) -> tuple[float, float, float]:
        a = self._alpha
        v = self._values
        v[0] += a[0] * (rms_lo - v[0])
        v[1] += a[1] * (rms_md - v[1])
        v[2] += a[2] * (rms_hi - v[2])
        return float(v[0]), float(v[1]), float(v[2])

    @property
    def values(self) -> np.ndarray:
        return self._values

    def reset(self) -> None:
        self._values.fill(0.0)


TILT_REF_HZ = 1000.0
EPS = 1e-12


class AutoScaler:
    """Rolling per-band normalizer.

    Pipeline (pre-tilted smoothed RMS in `v`):
        peak  ← asymmetric one-pole follower of v (fast attack, slow release)
        gated ← max(0, v - noise_floor)
        sc    ← tanh( gated / max(peak, noise_floor) )          # in [0, 1]
        raw   ← clip( (20·log10(v) - db_floor) / (db_ceil-db_floor), 0, 1 )
                with raw=0 where v < noise_floor                # in [0, 1]
        out   ← strength·sc + (1-strength)·raw                  # in [0, 1]

    Tilt is applied as a per-band linear pre-multiplier on `values_in`
    (= 10^(tilt_db/20), where tilt_db = tilt_db_per_oct · log2(f_c / 1 kHz)).
    Same shape as the FFT post-processor, where tilt is added in dB before
    the dB→linear conversion — so a single global noise_floor gates both
    pipelines identically.
    """

    def __init__(self, sr: float, blocksize: int,
                 tau_attack_s: float = 0.05, tau_release_s: float = 60.0,
                 noise_floor: float = 1e-3, strength: float = 1.0,
                 tilt_db_per_oct: float = 0.0,
                 band_centers: tuple = (100.0, 1000.0, 8000.0),
                 db_floor: float = -60.0, db_ceiling: float = 0.0):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self.noise_floor = max(0.0, float(noise_floor))
        self.strength = max(0.0, min(1.0, float(strength)))
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._band_centers = np.asarray(band_centers, dtype=np.float64)
        self.db_floor = float(db_floor)
        self.db_ceiling = float(db_ceiling)
        self._tilt_lin = np.ones(3, dtype=np.float64)
        self._v = np.zeros(3, dtype=np.float64)         # tilted input
        self._diff = np.zeros(3, dtype=np.float64)      # peak follower delta
        self._alpha = np.zeros(3, dtype=np.float64)     # per-band attack/release
        self._mask = np.zeros(3, dtype=bool)            # rising-edge mask (preallocated)
        self._scaled = np.zeros(3, dtype=np.float64)    # tanh output
        self._raw = np.zeros(3, dtype=np.float64)       # dB-mapped baseline
        self._peak = np.full(3, max(self.noise_floor, EPS), dtype=np.float64)
        self._warmed = False
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)
        self._recompute_tilt_lin()
        self.set_taus(tau_attack_s=self.tau_attack_s, tau_release_s=self.tau_release_s)

    def _recompute_tilt_lin(self) -> None:
        if self.tilt_db_per_oct == 0.0:
            self._tilt_lin.fill(1.0)
            return
        tilt_db = self.tilt_db_per_oct * np.log2(np.maximum(self._band_centers, 1e-3) / TILT_REF_HZ)
        np.power(10.0, tilt_db / 20.0, out=self._tilt_lin)

    def set_taus(self, tau_attack_s: float, tau_release_s: float) -> None:
        dt = self.blocksize / self.sr
        self._a_atk = 1.0 - math.exp(-dt / max(float(tau_attack_s), 1e-3))
        self._a_rel = 1.0 - math.exp(-dt / max(float(tau_release_s), 1e-3))
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)

    def set_noise_floor(self, floor: float) -> None:
        self.noise_floor = max(float(floor), 0.0)
        np.maximum(self._peak, max(self.noise_floor, EPS), out=self._peak)

    def set_tilt(self, tilt_db_per_oct: float) -> None:
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._recompute_tilt_lin()

    def set_band_centers(self, centers) -> None:
        self._band_centers = np.asarray(centers, dtype=np.float64)
        self._recompute_tilt_lin()

    def set_strength(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, float(strength)))

    def reset(self) -> None:
        self._peak.fill(max(self.noise_floor, EPS))
        self._warmed = False

    def update(self, values_in: np.ndarray, out: np.ndarray) -> np.ndarray:
        nf = max(self.noise_floor, EPS)
        # Pre-tilt
        np.multiply(values_in, self._tilt_lin, out=self._v)
        v = self._v

        # Asymmetric peak follower (warm to noise_floor — don't saturate to first sample)
        if not self._warmed:
            self._peak.fill(nf)
            self._warmed = True
        # Asymmetric one-pole follower, alloc-free: fill alpha with release,
        # then masked-copy attack into rising bins. _mask, _alpha, _diff are
        # all preallocated so no per-block temporaries are produced.
        np.subtract(v, self._peak, out=self._diff)
        np.greater(self._diff, 0.0, out=self._mask)
        self._alpha.fill(self._a_rel)
        np.copyto(self._alpha, self._a_atk, where=self._mask)
        np.multiply(self._alpha, self._diff, out=self._alpha)
        np.add(self._peak, self._alpha, out=self._peak)

        # tanh compressor: scaled = tanh( max(0, v - nf) / max(peak, nf) )
        # Repurpose _alpha (no longer needed after the peak update) as the
        # denominator buffer to avoid the np.maximum(peak, nf) temp.
        np.subtract(v, self.noise_floor, out=self._scaled)
        np.maximum(self._scaled, 0.0, out=self._scaled)
        np.maximum(self._peak, nf, out=self._alpha)
        np.divide(self._scaled, self._alpha, out=self._scaled)
        np.tanh(self._scaled, out=self._scaled)

        s = self.strength
        if s >= 1.0:
            np.copyto(out, self._scaled)
            return out

        # dB-mapped baseline in [0, 1]; gate-zero below noise floor.
        # Reuse _mask (preallocated 3-bool) for the below-floor mask.
        span = max(1.0, self.db_ceiling - self.db_floor)
        np.less(v, self.noise_floor, out=self._mask)
        np.maximum(v, EPS, out=self._raw)
        np.log10(self._raw, out=self._raw)
        np.multiply(self._raw, 20.0, out=self._raw)
        np.subtract(self._raw, self.db_floor, out=self._raw)
        np.divide(self._raw, span, out=self._raw)
        np.clip(self._raw, 0.0, 1.0, out=self._raw)
        self._raw[self._mask] = 0.0

        if s <= 0.0:
            np.copyto(out, self._raw)
        else:
            # Repurpose _diff as scratch for (1-s)*_raw to keep the blend
            # alloc-free.
            np.multiply(self._scaled, s, out=out)
            np.multiply(self._raw, 1.0 - s, out=self._diff)
            np.add(out, self._diff, out=out)
        return out
