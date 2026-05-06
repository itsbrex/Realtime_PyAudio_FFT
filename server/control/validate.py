"""Pure validators for control messages and persisted config values."""
from __future__ import annotations

import math
import re

MIN_HZ = 20.0
MIN_GAP_HZ = 50.0
BAND_NAMES = ("low", "mid", "high")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_band(name, lo_hz, hi_hz, sr):
    """Validate a single bandpass: MIN_HZ ≤ lo_hz, hi_hz > lo_hz + MIN_GAP_HZ, hi_hz ≤ 0.45·sr."""
    if not (_is_number(lo_hz) and _is_number(hi_hz) and _is_number(sr)):
        raise ValueError(f"{name}: cutoffs must be numeric")
    lo_hz = float(lo_hz)
    hi_hz = float(hi_hz)
    sr = float(sr)
    if not (math.isfinite(lo_hz) and math.isfinite(hi_hz)):
        raise ValueError(f"{name}: non-finite cutoff")
    if lo_hz < MIN_HZ:
        raise ValueError(f"{name}: lo_hz must be >= {MIN_HZ}")
    if hi_hz <= lo_hz + MIN_GAP_HZ:
        raise ValueError(f"{name}: hi_hz must be > lo_hz + {MIN_GAP_HZ}")
    if hi_hz > 0.45 * sr:
        raise ValueError(f"{name}: hi_hz must be <= 0.45*sr ({0.45 * sr:.0f} Hz)")
    return lo_hz, hi_hz


def validate_bands(bands_dict, sr):
    """Validate {low: {lo_hz, hi_hz}, mid: {...}, high: {...}}. Returns {name: (lo, hi)}.

    Bands may overlap or leave gaps — that's intentional. Each band is validated
    independently against MIN_HZ and 0.45·sr.
    """
    if not isinstance(bands_dict, dict):
        raise ValueError("bands must be a dict")
    out = {}
    for name in BAND_NAMES:
        b = bands_dict.get(name)
        if not isinstance(b, dict):
            raise ValueError(f"missing band {name!r}")
        out[name] = validate_band(name, b.get("lo_hz"), b.get("hi_hz"), sr)
    return out


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


def validate_autoscale(tau_attack_s=None, tau_release_s=None, noise_floor=None, strength=None):
    out = {}
    if tau_attack_s is not None:
        if not _is_number(tau_attack_s):
            raise ValueError("tau_attack_s must be numeric")
        v = float(tau_attack_s)
        if not (math.isfinite(v) and 0.001 <= v <= 1.0):
            raise ValueError("tau_attack_s must be in [1 ms, 1 s]")
        out["tau_attack_s"] = v
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


def validate_peak_smear_oct(v):
    if not _is_number(v):
        raise ValueError("peak_smear_oct must be numeric")
    v = float(v)
    if not (math.isfinite(v) and 0.0 <= v <= 3.0):
        raise ValueError("peak_smear_oct must be in [0, 3] octaves")
    return v


def validate_fft_tilt_db_per_oct(v):
    if not _is_number(v):
        raise ValueError("tilt_db_per_oct must be numeric")
    v = float(v)
    if not (math.isfinite(v) and -2.5 <= v <= 5.0):
        raise ValueError("tilt_db_per_oct must be in [-2.5, 5] dB/oct")
    return v


def validate_peak_decay_per_s(v):
    if not _is_number(v):
        raise ValueError("peak_decay_per_s must be numeric")
    v = float(v)
    if not (math.isfinite(v) and 0.0 <= v <= 10.0):
        raise ValueError("peak_decay_per_s must be in [0, 10] /s")
    return v


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
