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


class AutoScaler:
    """Rolling per-band normalizer.

    Asymmetric peak follower (fast attack, slow release) + soft noise gate +
    tanh compressor. Output ~[0, 1]. `strength` blends scaled vs raw values.
    """

    def __init__(self, sr: float, blocksize: int,
                 tau_attack_s: float = 0.05, tau_release_s: float = 60.0,
                 noise_floor: float = 1e-3, strength: float = 1.0):
        self.sr = float(sr)
        self.blocksize = int(blocksize)
        self.noise_floor = float(noise_floor)
        self.strength = float(strength)
        self._peak = np.full(3, self.noise_floor, dtype=np.float64)
        self._scratch = np.zeros(3, dtype=np.float64)
        self._scaled = np.zeros(3, dtype=np.float64)
        self._warmed = False
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)
        self.set_taus(tau_attack_s=self.tau_attack_s, tau_release_s=self.tau_release_s)

    def set_taus(self, tau_attack_s: float, tau_release_s: float) -> None:
        dt = self.blocksize / self.sr
        self._a_atk = 1.0 - math.exp(-dt / max(float(tau_attack_s), 1e-3))
        self._a_rel = 1.0 - math.exp(-dt / max(float(tau_release_s), 1e-3))
        self.tau_attack_s = float(tau_attack_s)
        self.tau_release_s = float(tau_release_s)

    def set_noise_floor(self, floor: float) -> None:
        self.noise_floor = max(float(floor), 0.0)
        np.maximum(self._peak, self.noise_floor, out=self._peak)

    def set_strength(self, strength: float) -> None:
        self.strength = max(0.0, min(1.0, float(strength)))

    def reset(self) -> None:
        self._peak.fill(self.noise_floor)
        self._warmed = False

    def update(self, values_in: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Compute scaled output (length 3, float64) into `out`. Returns out."""
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
        denom = np.maximum(self._peak, self.noise_floor)
        np.subtract(values_in, self.noise_floor, out=self._scaled)
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
