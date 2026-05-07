"""Cross-thread feature/FFT result stores.

FeatureStore: DSP worker -> sender. Mutex-guarded; ~1us hold.
FFTStore: FFT worker -> sender. Single-slot drop-old hand-off.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class FeatureStore:
    """Latest L/M/H raw + scaled snapshot.

    Backing storage is a pair of preallocated float64 arrays (length 3). The
    DSP worker calls `publish(raw_arr, scaled_arr, ...)` with its own numpy
    buffers; we copy in place under the lock. Hot consumers (the per-block
    OSC sender) use `read_scaled_into(out)` to copy values into a caller-owned
    scratch — zero allocation on the hot path. The lower-rate WS broadcast
    loop uses `read()`, which still allocates a tuple (it's about to JSON-
    encode anyway, so the saving would be invisible).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._raw = np.zeros(3, dtype=np.float64)
        self._scaled = np.zeros(3, dtype=np.float64)
        self._seq = 0
        self._t_recv_ns = 0  # perf_counter_ns of the audio block this snapshot was derived from

    def publish(self, raw, scaled, t_recv_ns: int = 0) -> None:
        with self._lock:
            np.copyto(self._raw, raw, casting="unsafe")
            np.copyto(self._scaled, scaled, casting="unsafe")
            self._t_recv_ns = int(t_recv_ns)
            self._seq += 1

    def read(self):
        with self._lock:
            return (self._seq,
                    (self._raw[0], self._raw[1], self._raw[2]),
                    (self._scaled[0], self._scaled[1], self._scaled[2]),
                    self._t_recv_ns)

    def read_scaled_into(self, out: np.ndarray):
        """Copy the scaled triple into `out` (length-3 float64). Returns
        (seq, t_recv_ns). No allocation."""
        with self._lock:
            np.copyto(out, self._scaled)
            return self._seq, self._t_recv_ns


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
        self._t_recv_ns = 0  # perf_counter_ns of the audio block that completed this hop

    def publish(self, raw_db: np.ndarray, processed: Optional[np.ndarray], t_recv_ns: int = 0) -> None:
        with self._lock:
            self._raw_db = raw_db
            self._processed = processed
            self._t_recv_ns = int(t_recv_ns)
            self._seq += 1

    def read(self, kind: str = "raw_db"):
        """Return (seq, frame, t_recv_ns). `kind` ∈ {'raw_db', 'processed'}.
        Falls back to raw_db if the requested stream isn't available."""
        with self._lock:
            if kind == "processed" and self._processed is not None:
                return self._seq, self._processed, self._t_recv_ns
            return self._seq, self._raw_db, self._t_recv_ns
