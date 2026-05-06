"""Cross-thread feature/FFT result stores.

FeatureStore: DSP worker -> sender. Mutex-guarded; ~1us hold.
FFTStore: FFT worker -> sender. Single-slot drop-old hand-off.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class FeatureStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._raw = (0.0, 0.0, 0.0)
        self._scaled = (0.0, 0.0, 0.0)
        self._seq = 0

    def publish(self, raw, scaled) -> None:
        with self._lock:
            self._raw = (float(raw[0]), float(raw[1]), float(raw[2]))
            self._scaled = (float(scaled[0]), float(scaled[1]), float(scaled[2]))
            self._seq += 1

    def read(self):
        with self._lock:
            return self._seq, self._raw, self._scaled


class FFTStore:
    """Holds the latest FFT frame as a (raw_db, processed) pair.

    Both arrays are updated atomically by `publish`; `read` returns whichever
    the caller asks for via `kind`. Consumers (OSC, WS) decide which to read
    based on `cfg.fft.send_raw_db`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._raw_db: Optional[np.ndarray] = None
        self._processed: Optional[np.ndarray] = None
        self._seq = 0

    def publish(self, raw_db: np.ndarray, processed: Optional[np.ndarray]) -> None:
        with self._lock:
            self._raw_db = raw_db
            self._processed = processed
            self._seq += 1

    def read(self, kind: str = "raw_db"):
        """Return (seq, frame). `kind` ∈ {'raw_db', 'processed'}. If the
        requested stream isn't available (post-processor not running), falls
        back to raw_db so consumers still get something sensible.
        """
        with self._lock:
            if kind == "processed" and self._processed is not None:
                return self._seq, self._processed
            return self._seq, self._raw_db
