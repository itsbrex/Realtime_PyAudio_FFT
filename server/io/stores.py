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
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0

    def publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self._seq += 1

    def read(self):
        with self._lock:
            return self._seq, self._latest
