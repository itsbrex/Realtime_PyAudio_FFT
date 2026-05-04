"""Pure validators for control messages and persisted config values."""
from __future__ import annotations

import math
import re

MIN_HZ = 20.0
MIN_GAP_HZ = 50.0


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_band_cutoffs(low_hz, high_hz, sr):
    if not (_is_number(low_hz) and _is_number(high_hz) and _is_number(sr)):
        raise ValueError("cutoffs must be numeric")
    low_hz = float(low_hz)
    high_hz = float(high_hz)
    sr = float(sr)
    if not (math.isfinite(low_hz) and math.isfinite(high_hz)):
        raise ValueError("non-finite cutoff")
    if low_hz < MIN_HZ:
        raise ValueError(f"low_hz must be >= {MIN_HZ}")
    if high_hz <= low_hz + MIN_GAP_HZ:
        raise ValueError(f"high_hz must be > low_hz + {MIN_GAP_HZ}")
    if high_hz > 0.45 * sr:
        raise ValueError(f"high_hz must be <= 0.45*sr ({0.45 * sr:.0f} Hz)")
    return low_hz, high_hz


def validate_tau(tau_dict):
    if not isinstance(tau_dict, dict):
        raise ValueError("tau must be a dict")
    out = {}
    for k, v in tau_dict.items():
        if k not in ("low", "mid", "high"):
            raise ValueError(f"unknown band {k!r}")
        if not _is_number(v):
            raise ValueError(f"tau[{k}] must be numeric")
        v = float(v)
        if not (math.isfinite(v) and 0.005 <= v <= 2.0):
            raise ValueError(f"tau[{k}] must be in [5 ms, 2 s]")
        out[k] = v
    return out


def validate_n_fft_bins(n):
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n_fft_bins must be an int")
    if not (8 <= n <= 1024):
        raise ValueError("n_fft_bins must be in [8, 1024]")
    return n


def validate_autoscale(tau_release_s=None, noise_floor=None, strength=None):
    out = {}
    if tau_release_s is not None:
        if not _is_number(tau_release_s):
            raise ValueError("tau_release_s must be numeric")
        v = float(tau_release_s)
        if not (math.isfinite(v) and 5.0 <= v <= 300.0):
            raise ValueError("tau_release_s must be in [5 s, 300 s]")
        out["tau_release_s"] = v
    if noise_floor is not None:
        if not _is_number(noise_floor):
            raise ValueError("noise_floor must be numeric")
        v = float(noise_floor)
        if not (math.isfinite(v) and 0.0 <= v <= 0.1):
            raise ValueError("noise_floor must be in [0, 0.1] linear RMS")
        out["noise_floor"] = v
    if strength is not None:
        if not _is_number(strength):
            raise ValueError("strength must be numeric")
        v = float(strength)
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError("strength must be in [0, 1]")
        out["strength"] = v
    return out


def validate_ws_snapshot_hz(hz):
    if not _is_number(hz):
        raise ValueError("hz must be numeric")
    hz = float(hz)
    if not math.isfinite(hz):
        raise ValueError("hz must be finite")
    if not (15 <= hz <= 240):
        raise ValueError("hz must be in [15, 240]")
    return int(round(hz))


_PRESET_NAME_RE = re.compile(r"^[a-zA-Z0-9_\- ]+$")


def validate_preset_name(name):
    if not isinstance(name, str):
        raise ValueError("preset name must be a string")
    n = name.strip()
    if not (1 <= len(n) <= 64):
        raise ValueError("preset name must be 1-64 characters")
    if not _PRESET_NAME_RE.match(n):
        raise ValueError(
            "preset name may only contain letters, digits, spaces, hyphens, underscores"
        )
    return n


def validate_device_index(idx):
    if not isinstance(idx, int) or isinstance(idx, bool):
        raise ValueError("device index must be an int")
    if idx < 0:
        raise ValueError("device index must be >= 0")
    return idx
