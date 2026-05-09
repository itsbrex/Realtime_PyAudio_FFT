"""Three independent Butterworth bandpass filters: LOW / MID / HIGH, SOS form."""
from __future__ import annotations

import threading

import numpy as np
from scipy.signal import iirfilter, sosfilt

BAND_NAMES = ("low", "mid", "high")


class FilterBank:
    """Three independent bandpass filters, owned by the DSP worker.

    Each band has its own [lo_hz, hi_hz] edges. Bands may overlap or leave
    gaps — that's intentional; it lets you carve out specific energy regions
    (e.g. drop sub-rumble from "low" by setting low.lo_hz=30, or scoop a
    midrange hole). `process(x)` allocates per call (sosfilt has no out=).
    That's fine: it runs on a normal Python thread, not the audio callback.
    zi is float64 (IIR numerical stability); inputs are float32. Outputs
    may be float64; that's irrelevant downstream because RMS collapses each
    band to a scalar.
    """

    def __init__(self, sr: float, bands: dict, blocksize: int, order: int = 4):
        # bands: {"low": (lo_hz, hi_hz), "mid": (...), "high": (...)}
        self.sr = float(sr)
        self.order = int(order)
        self.blocksize = int(blocksize)
        self._lock = threading.Lock()
        self._design(bands)
        self._init_state()
        self.bands = {k: (float(v[0]), float(v[1])) for k, v in bands.items()}

    def _design(self, bands: dict):
        nyq = self.sr / 2.0
        self.sos_low = iirfilter(
            self.order, [bands["low"][0] / nyq, bands["low"][1] / nyq],
            btype="bandpass", ftype="butter", output="sos",
        )
        self.sos_mid = iirfilter(
            self.order, [bands["mid"][0] / nyq, bands["mid"][1] / nyq],
            btype="bandpass", ftype="butter", output="sos",
        )
        self.sos_high = iirfilter(
            self.order, [bands["high"][0] / nyq, bands["high"][1] / nyq],
            btype="bandpass", ftype="butter", output="sos",
        )

    def _init_state(self):
        self.zi_low = np.zeros((self.sos_low.shape[0], 2), dtype=np.float64)
        self.zi_mid = np.zeros((self.sos_mid.shape[0], 2), dtype=np.float64)
        self.zi_high = np.zeros((self.sos_high.shape[0], 2), dtype=np.float64)

    def retune(self, bands: dict) -> None:
        """Called from the asyncio loop. Brief click acceptable on retune."""
        with self._lock:
            self._design(bands)
            self._init_state()
            self.bands = {k: (float(v[0]), float(v[1])) for k, v in bands.items()}

    def set_order(self, order: int) -> None:
        """Change the Butterworth order for all three bands. Higher order →
        steeper skirts, more biquads, more CPU. Brief click on change."""
        with self._lock:
            self.order = int(order)
            self._design(self.bands)
            self._init_state()

    def reset_state(self) -> None:
        with self._lock:
            self._init_state()

    def process(self, x: np.ndarray):
        with self._lock:
            out_lo, self.zi_low = sosfilt(self.sos_low, x, zi=self.zi_low)
            out_md, self.zi_mid = sosfilt(self.sos_mid, x, zi=self.zi_mid)
            out_hi, self.zi_high = sosfilt(self.sos_high, x, zi=self.zi_high)
        return out_lo, out_md, out_hi
