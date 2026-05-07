"""Shared low-level math helpers for the DSP pipeline.

These are pure, allocation-aware primitives that were previously duplicated
across features.py and fft_postprocess.py. Single source of truth for the
constants (TILT_REF_HZ, LN10_OVER_20, the tau floor) and the canonical
formulas (tau→alpha, log-spaced bin centers, spectral tilt curve).
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-12

# 10^x ≡ exp(x · ln 10). Pre-multiplying by ln(10)/20 once and calling
# np.exp is several× faster than np.power(10.0, x/20) on ARM and x86, since
# np.power is internally log+mul+exp.
LN10_OVER_20 = math.log(10.0) / 20.0

# Reference frequency for spectral tilt: the per-bin offset is zero AT this
# frequency and grows ±db_per_oct dB per octave away from it.
TILT_REF_HZ = 1000.0

# Floor for tau values when computing alpha — guards div-by-zero and
# pathological "instant" alphas if a tau is mis-configured to ~0.
_TAU_FLOOR = 1e-3


def tau_to_alpha(dt: float, tau: float) -> float:
    """One-pole alpha for a given timestep / time constant.

    alpha = 1 - exp(-dt / max(tau, 1e-3)).
    Apply once per sample as `v += alpha * (target - v)`.
    """
    return 1.0 - math.exp(-dt / max(float(tau), _TAU_FLOOR))


def tau_to_alpha_into(dt: float, tau: np.ndarray, out: np.ndarray) -> None:
    """Vectorised tau→alpha, in-place into `out`. Same formula as tau_to_alpha.

    Allocation-free given a preallocated `out`. Safe to alias `tau` and `out`.
    """
    np.maximum(tau, _TAU_FLOOR, out=out)
    np.divide(-dt, out, out=out)
    np.exp(out, out=out)
    np.subtract(1.0, out, out=out)


def tilt_db_curve(freqs: np.ndarray, db_per_oct: float,
                  ref_hz: float = TILT_REF_HZ) -> np.ndarray:
    """Per-frequency tilt offset in dB: db_per_oct · log2(f / ref_hz).
    Allocates a new float64 array — used off the hot path."""
    return db_per_oct * np.log2(np.maximum(freqs, 1e-3) / ref_hz)


def log_bin_centers(n_bins: int, f_min: float, f_max: float) -> np.ndarray:
    """Geometric centers of n_bins log-spaced bins from f_min to f_max
    (matching the FFT viz layout). Allocates."""
    log_fmin = math.log10(max(f_min, 1e-3))
    log_span = max(1e-6, math.log10(max(f_max, 1e-3)) - log_fmin)
    log_f = log_fmin + ((np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins) * log_span
    return np.power(10.0, log_f)
