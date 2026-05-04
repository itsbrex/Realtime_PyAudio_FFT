"""DSP worker thread: filter -> RMS -> smooth -> autoscale -> publish."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .features import ExpSmoother, AutoScaler, block_rms
from .filters import FilterBank

log = logging.getLogger(__name__)

MAX_DSP_BACKLOG_BLOCKS = 4  # ~21 ms at 48k/256


class DSPWorker(threading.Thread):
    def __init__(self, ring, dsp_event: threading.Event, stop_flag: threading.Event,
                 filter_bank: FilterBank, smoother: ExpSmoother, auto_scaler: AutoScaler,
                 features_store, on_publish, blocksize: int, perf_ring: np.ndarray):
        super().__init__(name="dsp-worker", daemon=True)
        self.ring = ring
        self.dsp_event = dsp_event
        self.stop_flag = stop_flag
        self.filter_bank = filter_bank
        self.smoother = smoother
        self.auto_scaler = auto_scaler
        self.features_store = features_store
        self.on_publish = on_publish  # threadsafe callback (e.g. set sender_event)
        self.blocksize = blocksize
        self.dsp_in = np.zeros(blocksize, dtype=np.float32)
        self.scaled_buf = np.zeros(3, dtype=np.float64)
        self.read_block_idx = 0
        self.dsp_drops = 0
        self.perf_ring = perf_ring
        self.perf_len = perf_ring.shape[0]
        self.perf_idx = 0

    def reset(self) -> None:
        """Called from the asyncio loop on device hot-switch. Worker thread
        is paused via stop_flag in the orchestrator before this runs."""
        self.read_block_idx = 0

    def run(self) -> None:
        while True:
            if not self.dsp_event.wait(timeout=0.1):
                if self.stop_flag.is_set():
                    return
                continue
            self.dsp_event.clear()
            if self.stop_flag.is_set():
                return

            wi = self.ring.write_idx
            if wi - self.read_block_idx > MAX_DSP_BACKLOG_BLOCKS:
                skipped = (wi - 1) - self.read_block_idx
                if skipped > 0:
                    self.dsp_drops += skipped
                self.read_block_idx = wi - 1

            if self.read_block_idx < wi:
                if self.ring.try_read_block(self.read_block_idx, self.dsp_in):
                    t0 = time.perf_counter_ns()
                    lo, md, hi = self.filter_bank.process(self.dsp_in)
                    rms_lo = block_rms(lo)
                    rms_md = block_rms(md)
                    rms_hi = block_rms(hi)
                    self.smoother.update(rms_lo, rms_md, rms_hi)
                    self.auto_scaler.update(self.smoother.values, self.scaled_buf)
                    raw = (
                        float(self.smoother.values[0]),
                        float(self.smoother.values[1]),
                        float(self.smoother.values[2]),
                    )
                    scaled = (float(self.scaled_buf[0]), float(self.scaled_buf[1]), float(self.scaled_buf[2]))
                    self.features_store.publish(raw, scaled)
                    try:
                        self.on_publish()
                    except Exception as e:
                        log.debug("on_publish raised: %s", e)
                    t1 = time.perf_counter_ns()
                    i = self.perf_idx
                    self.perf_ring[i % self.perf_len] = t1 - t0
                    self.perf_idx = i + 1
                    self.read_block_idx += 1
                else:
                    self.dsp_drops += 1
                    self.read_block_idx = max(self.read_block_idx + 1, wi - 1)
