"""Per-band exponential smoother + rolling auto-scaler. Plan section 6.3."""
from __future__ import annotations

import math

import numpy as np


def block_rms(x: np.ndarray) -> float:
    """RMS of a 1-D buffer; returns Python float."""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


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


class AutoScaler:
    """Rolling per-band normalizer.

    Asymmetric peak follower (fast attack, slow release) + soft noise gate +
    tanh compressor. Output ~[0, 1]. `strength` blends scaled vs raw values.

    The noise gate is per-band: a single global `noise_floor` is multiplied by
    a frequency-dependent scale derived from `tilt_db_per_oct` evaluated at
    each band's geometric center (zero at 1 kHz). Positive tilt LIFTS high-band
    sensitivity (lower effective gate) and PUSHES DOWN low-band sensitivity
    (higher effective gate), compensating for the natural downward slope of
    music spectra so the same global gate fires uniformly across L/M/H. Mirrors
    the FFTPostProcessor tilt so both pipelines respond in lockstep.
    """

    def __init__(self, sr: float, blocksize: int,
                 tau_attack_s: float = 0.05, tau_release_s: float = 60.0,
                 noise_floor: float = 1e-3, strength: float = 1.0,
                 tilt_db_per_oct: float = 0.0,
                 band_centers: tuple = (100.0, 1000.0, 8000.0)):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self.noise_floor = float(noise_floor)
        self.strength = float(strength)
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._band_centers = np.asarray(band_centers, dtype=np.float64)
        self._noise_floor_per_band = np.full(3, self.noise_floor, dtype=np.float64)
        self._recompute_per_band_floor()
        self._peak = np.array(self._noise_floor_per_band, dtype=np.float64)
        self._scratch = np.zeros(3, dtype=np.float64)
        self._scaled = np.zeros(3, dtype=np.float64)
        self._warmed = False
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)
        self.set_taus(tau_attack_s=self.tau_attack_s, tau_release_s=self.tau_release_s)

    def _recompute_per_band_floor(self) -> None:
        # nf_band = noise_floor * 10^(-tilt_dB(f_c) / 20),
        # where tilt_dB(f_c) = tilt_db_per_oct * log2(f_c / TILT_REF_HZ).
        # Positive tilt → high band gets LOWER effective floor; low band gets HIGHER.
        if self.tilt_db_per_oct == 0.0 or self.noise_floor <= 0.0:
            self._noise_floor_per_band[:] = self.noise_floor
            return
        tilt_db = self.tilt_db_per_oct * np.log2(np.maximum(self._band_centers, 1e-3) / TILT_REF_HZ)
        scale = np.power(10.0, -tilt_db / 20.0)
        np.multiply(scale, self.noise_floor, out=self._noise_floor_per_band)

    def set_taus(self, tau_attack_s: float, tau_release_s: float) -> None:
        dt = self.blocksize / self.sr
        self._a_atk = 1.0 - math.exp(-dt / max(float(tau_attack_s), 1e-3))
        self._a_rel = 1.0 - math.exp(-dt / max(float(tau_release_s), 1e-3))
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)

    def set_noise_floor(self, floor: float) -> None:
        self.noise_floor = max(float(floor), 0.0)
        self._recompute_per_band_floor()
        np.maximum(self._peak, self._noise_floor_per_band, out=self._peak)

    def set_tilt(self, tilt_db_per_oct: float) -> None:
        self.tilt_db_per_oct = float(tilt_db_per_oct)
        self._recompute_per_band_floor()
        np.maximum(self._peak, self._noise_floor_per_band, out=self._peak)

    def set_band_centers(self, centers) -> None:
        self._band_centers = np.asarray(centers, dtype=np.float64)
        self._recompute_per_band_floor()
        np.maximum(self._peak, self._noise_floor_per_band, out=self._peak)

    def set_strength(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, float(strength)))

    def reset(self) -> None:
        np.copyto(self._peak, self._noise_floor_per_band)
        self._warmed = False

    def update(self, values_in: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Compute scaled output (length 3, float64) into `out`. Returns out."""
        nfpb = self._noise_floor_per_band
        if not self._warmed:
            np.maximum(self._peak, values_in, out=self._peak)
            self._warmed = True
        else:
            # asymmetric one-pole: alpha = a_atk where rising else a_rel
            self._scratch.fill(self._a_rel)
            rising = values_in > self._peak
            self._scratch[rising] = self._a_atk
            # peak += alpha * (values_in - peak)
            self._peak += self._scratch * (values_in - self._peak)
        denom = np.maximum(self._peak, nfpb)
        np.subtract(values_in, nfpb, out=self._scaled)
        np.maximum(self._scaled, 0.0, out=self._scaled)
        np.divide(self._scaled, denom, out=self._scaled)
        np.tanh(self._scaled, out=self._scaled)

        s = self.strength
        if s >= 1.0:
            np.copyto(out, self._scaled)
        elif s <= 0.0:
            np.copyto(out, values_in)
        else:
            # blend: out = s * scaled + (1-s) * values_in
            np.multiply(self._scaled, s, out=out)
            out += (1.0 - s) * values_in
        return out
