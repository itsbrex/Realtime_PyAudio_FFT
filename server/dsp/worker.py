"""DSP worker thread: filter -> RMS -> smooth -> autoscale -> publish."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .features import ExpSmoother, AutoScaler, block_rms
from .filters import FilterBank
from .onset import OnsetTracker

log = logging.getLogger(__name__)

MAX_DSP_BACKLOG_BLOCKS = 4  # ~21 ms at 48k/256


class DSPWorker(threading.Thread):
    def __init__(self, ring, dsp_event: threading.Event, stop_flag: threading.Event,
                 filter_bank: FilterBank, smoother: ExpSmoother, auto_scaler: AutoScaler,
                 onset_tracker: OnsetTracker,
                 features_store, on_publish, blocksize: int, perf_ring: np.ndarray):
        super().__init__(name="dsp-worker", daemon=True)
        self.ring = ring
        self.dsp_event = dsp_event
        self.stop_flag = stop_flag
        self.filter_bank = filter_bank
        self.smoother = smoother
        self.auto_scaler = auto_scaler
        self.onset_tracker = onset_tracker
        self.features_store = features_store
        self.on_publish = on_publish  # threadsafe callback (e.g. set sender_event)
        self.blocksize = blocksize
        self.dsp_in = np.zeros(blocksize, dtype=np.float32)
        self.scaled_buf = np.zeros(3, dtype=np.float64)
        self.onsets_buf = np.zeros(3, dtype=np.int8)
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
            # Self-heal after a producer reset (ring.reset() on device hot-switch
            # rewinds write_idx to 0). Without this, read_block_idx stays in the
            # old monotonic range forever and the worker never advances again.
            if self.read_block_idx > wi:
                self.read_block_idx = wi
            if wi - self.read_block_idx > MAX_DSP_BACKLOG_BLOCKS:
                skipped = (wi - 1) - self.read_block_idx
                if skipped > 0:
                    self.dsp_drops += skipped
                self.read_block_idx = wi - 1

            # Drain every block currently in the ring on each wake. threading.Event.set()
            # is idempotent — N callbacks between two wakes coalesce into one — so a one-
            # block-per-wake loop can never catch up after falling behind, and ends up
            # publishing stale block timestamps (lmh_e2e ≈ N · block_period). Draining
            # keeps FeatureStore on the freshest block and the IIR filter state correct.
            while self.read_block_idx < wi:
                if self.ring.try_read_block(self.read_block_idx, self.dsp_in):
                    t_recv_ns = int(self.ring.block_t_ns[self.read_block_idx & self.ring.mask])
                    t0 = time.perf_counter_ns()
                    lo, md, hi = self.filter_bank.process(self.dsp_in)
                    rms_lo = block_rms(lo)
                    rms_md = block_rms(md)
                    rms_hi = block_rms(hi)
                    self.smoother.update(rms_lo, rms_md, rms_hi)
                    self.auto_scaler.update(self.smoother.values, self.scaled_buf)
                    # Onset detection runs on the FULLY post-processed L/M/H
                    # signals (after smoother + autoscaler + strength blend)
                    # so that the same UI knobs that shape the bars/lines —
                    # bandpass edges, smoothing, autoscale strength, noise
                    # gate — also tune the trackers. The user dials each
                    # band's signal in until it 'pulses' cleanly on the
                    # transients of interest (kicks / snares / hats); the
                    # detectors ride those exact signals.
                    bpm = self.onset_tracker.update(self.scaled_buf, self.onsets_buf)
                    # Pass numpy refs straight through; FeatureStore copies
                    # into its own preallocated buffers under the lock so we
                    # don't allocate per-block tuples here.
                    self.features_store.publish(
                        self.smoother.values, self.scaled_buf,
                        self.onsets_buf, self.onset_tracker.onset_count, bpm,
                        t_recv_ns,
                    )
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
