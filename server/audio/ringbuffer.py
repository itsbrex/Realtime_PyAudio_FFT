"""SPSC block-aligned slot ring with seqlock-style publish.

Plan section 4.2. Single producer (audio callback), two consumers
(DSP worker, FFT worker). Each slot holds exactly one block of mono
float32 samples. The producer publishes by writing the per-slot
block_seq AFTER the data copy, then the global write_idx. Consumers
verify each read against block_seq before AND after copying out, which
catches a producer lapping mid-read regardless of memory model.
"""
from __future__ import annotations

import numpy as np


class SlotRing:
    def __init__(self, n_slots_pow2: int, blocksize: int):
        if n_slots_pow2 & (n_slots_pow2 - 1):
            raise ValueError("n_slots must be a power of two")
        if n_slots_pow2 < 8:
            raise ValueError("n_slots must be >= 8")
        self.n = n_slots_pow2
        self.mask = n_slots_pow2 - 1
        self.blocksize = blocksize
        self.slots = np.zeros((n_slots_pow2, blocksize), dtype=np.float32)
        self.block_seq = np.zeros(n_slots_pow2, dtype=np.int64)
        self.write_idx = 0  # monotonic block count; published last

    # ------------- producer (audio callback) -------------
    def write_block(self, src: np.ndarray) -> None:
        wi = self.write_idx
        slot = wi & self.mask
        np.copyto(self.slots[slot], src)        # 1. data
        self.block_seq[slot] = wi + 1           # 2. per-slot publish
        self.write_idx = wi + 1                 # 3. global publish

    # ------------- DSP consumer (one block) -------------
    def try_read_block(self, read_idx: int, out: np.ndarray) -> bool:
        slot = read_idx & self.mask
        s1 = int(self.block_seq[slot])
        if s1 != read_idx + 1:
            return False
        np.copyto(out, self.slots[slot])
        return int(self.block_seq[slot]) == s1

    # ------------- FFT consumer (n consecutive blocks into out[]) -------------
    def try_read_window(self, start_block_idx: int, n_blocks: int, out: np.ndarray) -> bool:
        bs = self.blocksize
        for k in range(n_blocks):
            slot = (start_block_idx + k) & self.mask
            expected = start_block_idx + k + 1
            s1 = int(self.block_seq[slot])
            if s1 != expected:
                return False
            np.copyto(out[k * bs:(k + 1) * bs], self.slots[slot])
            if int(self.block_seq[slot]) != s1:
                return False
        return True

    def reset(self) -> None:
        """Wipe state; called on device hot-switch."""
        self.write_idx = 0
        self.block_seq.fill(0)
        self.slots.fill(0.0)
