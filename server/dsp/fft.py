"""FFT worker: hop-driven, log-spaced bins, dB output."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

MAX_BACKLOG_HOPS = 2
EPS = 1e-12
# Sentinel for log bins with no rfft bin mapped (happens at the low end where
# rfft frequency resolution is coarser than the log spacing). Chosen far below
# any physically achievable dB value (rfft of EPS clamps at -240 dB) so the UI
# can distinguish "empty bin" from "very quiet real measurement". OSC sender
# replaces this with db_floor before transmitting to keep the wire contract.
EMPTY_BIN_SENTINEL = -1000.0


def build_log_bin_map(window_size: int, sr: float, n_bins: int, f_min: float):
    """Compute log-spaced bin assignment for an rfft of length window_size.

    Returns (bin_assign, bin_valid_mask, bin_idx_valid, bin_counts).
    bin_assign[k] is the log-bin index for rfft bin k, or -1 if outside [f_min, sr/2].
    bin_counts[j] is the number of rfft bins mapped to log bin j (0 for empty bins).
    """
    n_rfft = window_size // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_rfft, dtype=np.float64)
    f_max = sr / 2.0
    if f_max <= f_min:
        f_max = f_min * 2.0
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    bin_assign = np.full(n_rfft, -1, dtype=np.int64)
    for k, f in enumerate(freqs):
        if f < f_min or f > f_max:
            continue
        idx = int(np.searchsorted(edges, f, side="right") - 1)
        if idx < 0:
            continue
        if idx >= n_bins:
            idx = n_bins - 1
        bin_assign[k] = idx
    bin_valid_mask = bin_assign >= 0
    bin_idx_valid = bin_assign[bin_valid_mask].astype(np.int64)
    bin_counts = np.bincount(bin_idx_valid, minlength=n_bins).astype(np.float64)
    if bin_counts.shape[0] > n_bins:
        bin_counts = bin_counts[:n_bins]
    return bin_assign, bin_valid_mask, bin_idx_valid, bin_counts


class FFTWorker(threading.Thread):
    def __init__(self, ring, fft_event: threading.Event, fft_enabled: threading.Event,
                 stop_flag: threading.Event, fft_store, on_publish,
                 blocksize: int, sr: float, window_size: int, hop: int,
                 n_bins: int, f_min: float, perf_ring: np.ndarray,
                 db_floor: float = -60.0):
        super().__init__(name="fft-worker", daemon=True)
        self.ring = ring
        self.fft_event = fft_event
        self.fft_enabled = fft_enabled
        self.stop_flag = stop_flag
        self.fft_store = fft_store
        self.on_publish = on_publish
        self.blocksize = blocksize
        self.sr = float(sr)
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.n_bins = int(n_bins)
        self.f_min = float(f_min)
        self.db_floor = float(db_floor)
        self.read_block_idx = 0
        self.fft_drops = 0
        self.perf_ring = perf_ring
        self.perf_len = perf_ring.shape[0]
        self.perf_idx = 0
        self._lock = threading.Lock()
        self._allocate()

    def _allocate(self) -> None:
        ws = self.window_size
        self.n_blocks_per_window = ws // self.blocksize
        self.hop_blocks = self.hop // self.blocksize
        if self.n_blocks_per_window < 1 or self.hop_blocks < 1:
            raise ValueError("window_size/hop must be multiples of blocksize")
        self.window_buf = np.zeros(ws, dtype=np.float32)
        self.hann = np.hanning(ws).astype(np.float32)
        self.spectrum = np.zeros(ws // 2 + 1, dtype=np.complex64)
        self.mag_buf = np.zeros(ws // 2 + 1, dtype=np.float32)
        self.db_buf = np.zeros(ws // 2 + 1, dtype=np.float32)
        self.bin_assign, self.bin_valid_mask, self.bin_idx_valid, self.bin_counts = build_log_bin_map(
            ws, self.sr, self.n_bins, self.f_min
        )
        # Precompute the divisor (1/count where count>0; 1 elsewhere — empty bins
        # are overwritten with floor below) and the empty-bin mask.
        self._bin_count_inv = np.where(self.bin_counts > 0, 1.0 / np.maximum(self.bin_counts, 1.0), 0.0)
        self._bin_empty_mask = self.bin_counts == 0

    # ------------- live retune from asyncio thread -------------
    def reconfigure(self, n_bins: int | None = None, sr: float | None = None,
                    window_size: int | None = None, hop: int | None = None,
                    f_min: float | None = None) -> None:
        with self._lock:
            if n_bins is not None: self.n_bins = int(n_bins)
            if sr is not None: self.sr = float(sr)
            if window_size is not None: self.window_size = int(window_size)
            if hop is not None: self.hop = int(hop)
            if f_min is not None: self.f_min = float(f_min)
            self._allocate()
            self.read_block_idx = 0  # alignment changed

    def reset(self) -> None:
        with self._lock:
            self.read_block_idx = 0

    def run(self) -> None:
        while True:
            self.fft_event.wait(timeout=0.1)
            if self.stop_flag.is_set():
                return
            self.fft_event.clear()
            if not self.fft_enabled.is_set():
                continue

            with self._lock:
                wi = self.ring.write_idx

                # Self-heal after a producer reset (ring.reset() on device
                # hot-switch). See dsp/worker.py for the same pattern.
                if self.read_block_idx > wi:
                    self.read_block_idx = 0

                if wi - self.read_block_idx > self.n_blocks_per_window + MAX_BACKLOG_HOPS * self.hop_blocks:
                    target_block = wi - self.n_blocks_per_window
                    skipped_hops = (target_block - self.read_block_idx) // self.hop_blocks
                    if skipped_hops > 0:
                        self.fft_drops += int(skipped_hops)
                        self.read_block_idx += int(skipped_hops) * self.hop_blocks

                if wi - self.read_block_idx < self.n_blocks_per_window:
                    continue

                t0 = time.perf_counter_ns()
                if not self.ring.try_read_window(self.read_block_idx, self.n_blocks_per_window, self.window_buf):
                    self.fft_drops += 1
                    self.read_block_idx = max(
                        self.read_block_idx + self.hop_blocks,
                        wi - self.n_blocks_per_window,
                    )
                    continue

                np.multiply(self.window_buf, self.hann, out=self.window_buf)
                np.fft.rfft(self.window_buf, out=self.spectrum)
                np.abs(self.spectrum, out=self.mag_buf)
                np.add(self.mag_buf, EPS, out=self.mag_buf)
                np.log10(self.mag_buf, out=self.db_buf)
                np.multiply(self.db_buf, 20.0, out=self.db_buf)
                bins = np.bincount(
                    self.bin_idx_valid,
                    weights=self.db_buf[self.bin_valid_mask],
                    minlength=self.n_bins,
                )
                if bins.shape[0] > self.n_bins:
                    bins = bins[: self.n_bins]
                # Mean dB per log bin (where it has any rfft bins mapped to it);
                # empty log bins get the floor so they render as "no signal" instead
                # of "0 dB", which the UI would otherwise paint as max.
                np.multiply(bins, self._bin_count_inv, out=bins)
                if self._bin_empty_mask.any():
                    bins[self._bin_empty_mask] = EMPTY_BIN_SENTINEL
                self.fft_store.publish(bins)
                try:
                    self.on_publish()
                except Exception as e:
                    log.debug("fft on_publish raised: %s", e)
                t1 = time.perf_counter_ns()
                i = self.perf_idx
                self.perf_ring[i % self.perf_len] = t1 - t0
                self.perf_idx = i + 1
                self.read_block_idx += self.hop_blocks
