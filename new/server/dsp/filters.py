"""Three-band Butterworth filter bank: LP / BP / HP, SOS form."""
from __future__ import annotations

import threading

import numpy as np
from scipy.signal import iirfilter, sosfilt


class FilterBank:
    """Stateful LP/BP/HP filter bank, owned by the DSP worker.

    `process(x)` allocates per call (sosfilt has no out=). That's fine: it runs
    on a normal Python thread, not the audio callback. zi is float64 (IIR
    numerical stability); inputs are float32. Outputs may be float64; that's
    irrelevant downstream because RMS collapses each band to a scalar.
    """

    def __init__(self, sr: float, low_hz: float, high_hz: float, blocksize: int, order: int = 4):
        self.sr = float(sr)
        self.order = int(order)
        self.blocksize = int(blocksize)
        self._lock = threading.Lock()
        self._design(low_hz, high_hz)
        self._init_state()
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)

    def _design(self, low_hz: float, high_hz: float):
        nyq = self.sr / 2.0
        wn_lo = low_hz / nyq
        wn_hi = high_hz / nyq
        self.sos_lp = iirfilter(self.order, wn_lo, btype="lowpass",  ftype="butter", output="sos")
        self.sos_bp = iirfilter(self.order, [wn_lo, wn_hi], btype="bandpass", ftype="butter", output="sos")
        self.sos_hp = iirfilter(self.order, wn_hi, btype="highpass", ftype="butter", output="sos")

    def _init_state(self):
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2), dtype=np.float64)
        self.zi_bp = np.zeros((self.sos_bp.shape[0], 2), dtype=np.float64)
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2), dtype=np.float64)

    def retune(self, low_hz: float, high_hz: float) -> None:
        """Called from the asyncio loop. Brief click acceptable on retune."""
        with self._lock:
            self._design(low_hz, high_hz)
            self._init_state()
            self.low_hz = float(low_hz)
            self.high_hz = float(high_hz)

    def reset_state(self) -> None:
        with self._lock:
            self._init_state()

    def process(self, x: np.ndarray):
        with self._lock:
            out_lo, self.zi_lp = sosfilt(self.sos_lp, x, zi=self.zi_lp)
            out_md, self.zi_bp = sosfilt(self.sos_bp, x, zi=self.zi_bp)
            out_hi, self.zi_hp = sosfilt(self.sos_hp, x, zi=self.zi_hp)
        return out_lo, out_md, out_hi
