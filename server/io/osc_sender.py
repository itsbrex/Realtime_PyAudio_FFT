"""OSC UDP sender. One SimpleUDPClient per destination."""
from __future__ import annotations

import asyncio
import logging
import socket
import struct
import time

import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from ..config import OscDest

log = logging.getLogger(__name__)


# Pre-encoded OSC packet for "/audio/lmh ,fff <f><f><f>".
# Address "/audio/lmh" padded to 12 bytes, type tag ",fff" padded to 8 bytes,
# then 3 big-endian float32 args = 32 bytes total. Floats start at offset 20.
_LMH_TEMPLATE = b"/audio/lmh\x00\x00,fff\x00\x00\x00\x00" + b"\x00" * 12
_LMH_FLOAT_OFFSET = 20
_LMH_PACK = struct.Struct("!fff").pack_into

# Pre-encoded "/audio/beat ,i <int=1>". Beat is sent ONLY on onset blocks
# (no zero pulses), so the value is always 1 and the whole packet is a
# constant — no per-send packing.
# "/audio/beat" (11 bytes) + NUL padding to 12 (1 NUL) = 12.
# ",i" (2 bytes) + NUL padding to 4 (2 NULs) = 4.
# int32 BE = 4 bytes (value 1 = "\x00\x00\x00\x01").
# Total: 20 bytes.
_BEAT_PACKET = b"/audio/beat\x00,i\x00\x00\x00\x00\x00\x01"

# Pre-encoded "/audio/bpm ,f <f>". Float arg is mutated in place.
# "/audio/bpm" (10 bytes) + NUL padding to 12 (2 NULs) = 12.
# ",f" (2 bytes) + NUL padding to 4 (2 NULs) = 4.
# float32 BE = 4 bytes. Float starts at offset 16. Total: 20.
_BPM_TEMPLATE = b"/audio/bpm\x00\x00,f\x00\x00\x00\x00\x00\x00"
_BPM_FLOAT_OFFSET = 16
_BPM_PACK = struct.Struct("!f").pack_into


class OscSender:
    def __init__(self, destinations: list[OscDest]):
        self._dests = list(destinations)
        self._clients = [SimpleUDPClient(d.host, d.port) for d in self._dests]
        # Per-block hot path: bypass pythonosc (which builds a fresh
        # OscMessage + bytes object per call) by mutating a single
        # preallocated packet buffer and sending it via raw UDP. Zero
        # allocation per send.
        self._lmh_buf = bytearray(_LMH_TEMPLATE)
        self._bpm_buf = bytearray(_BPM_TEMPLATE)
        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_sock.setblocking(False)
        self._addrs = [(d.host, int(d.port)) for d in self._dests]
        # Preallocated f32 scratch for FFT sends; resized lazily if n_bins
        # changes (rare). Avoids a per-hop n_bins-sized alloc from
        # `np.maximum(bins, db_floor)` and `arr * gain`.
        self._fft_scratch: np.ndarray | None = None

    def send_meta(self, sr: int, blocksize: int, n_fft_bins: int, bands: dict) -> None:
        # Payload: [sr, blocksize, n_fft_bins, low_lo, low_hi, mid_lo, mid_hi, high_lo, high_hi]
        msg = [
            int(sr), int(blocksize), int(n_fft_bins),
            float(bands["low"][0]),  float(bands["low"][1]),
            float(bands["mid"][0]),  float(bands["mid"][1]),
            float(bands["high"][0]), float(bands["high"][1]),
        ]
        for c in self._clients:
            try:
                c.send_message("/audio/meta", msg)
            except Exception as e:
                log.debug("osc meta send failed: %s", e)

    def send_lmh(self, low: float, mid: float, high: float) -> None:
        # Zero-alloc hot path: pack into the preallocated packet buffer
        # and sendto each destination. struct.pack_into and socket.sendto
        # do not allocate Python objects.
        _LMH_PACK(self._lmh_buf, _LMH_FLOAT_OFFSET, low, mid, high)
        buf = self._lmh_buf
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc lmh send failed: %s", e)

    def send_beat(self) -> None:
        """Sends `/audio/beat 1`. Constant packet — no packing needed.
        Called only on onset blocks; absence of a beat is silence on this
        address (no `0`s sent), giving downstream consumers a clean
        rising-edge trigger semantic."""
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(_BEAT_PACKET, addr)
            except Exception as e:
                log.debug("osc beat send failed: %s", e)

    def send_bpm(self, bpm: float) -> None:
        """Sends `/audio/bpm <f>`. Per-block; cheap (4-byte float pack +
        20-byte UDP packet). Downstream caches whatever was last received."""
        _BPM_PACK(self._bpm_buf, _BPM_FLOAT_OFFSET, float(bpm))
        buf = self._bpm_buf
        sock = self._udp_sock
        for addr in self._addrs:
            try:
                sock.sendto(buf, addr)
            except Exception as e:
                log.debug("osc bpm send failed: %s", e)

    def _ensure_fft_scratch(self, n: int) -> np.ndarray:
        s = self._fft_scratch
        if s is None or s.size != n:
            s = np.empty(n, dtype=np.float32)
            self._fft_scratch = s
        return s

    def send_fft(self, bins: np.ndarray, db_floor: float) -> None:
        # OSC 'f' tag is 32-bit float. The wire array contains -1000 sentinels
        # for empty log bins; np.maximum clamps both those (and any sub-floor
        # real values, harmless) to db_floor in a single fused C pass. We
        # write into a preallocated scratch so the publisher's buffer stays
        # untouched (still being read by the WS encoder concurrently) without
        # a per-hop allocation.
        scratch = self._ensure_fft_scratch(bins.shape[0])
        np.maximum(bins, np.float32(db_floor), out=scratch)
        payload = scratch.tolist()
        for c in self._clients:
            try:
                c.send_message("/audio/fft", payload)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)

    def send_fft_processed(self, bins: np.ndarray, gain: float = 1.0) -> None:
        # Post-processed values are already in [0, 1] (no sentinels). master
        # gain >1 may push them above 1.0 — that's by design and downstream
        # consumers must accept it.
        if gain != 1.0:
            scratch = self._ensure_fft_scratch(bins.shape[0])
            np.multiply(bins, np.float32(gain), out=scratch)
            payload = scratch.tolist()
        else:
            # tolist() itself allocates a Python list (unavoidable for
            # pythonosc), but skip the array multiply.
            payload = bins.tolist()
        for c in self._clients:
            try:
                c.send_message("/audio/fft", payload)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)


async def osc_sender_task(stop, sender_event: asyncio.Event, sender: OscSender,
                          features_store, fft_store, get_send_fft, get_fft_enabled,
                          get_db_floor, get_send_raw_db, get_master_gain,
                          perf_lmh_e2e: np.ndarray | None = None,
                          perf_fft_e2e: np.ndarray | None = None,
                          perf_idx_state: dict | None = None):
    """Wakes on sender_event; sends one /audio/lmh per audio block.

    `perf_lmh_e2e` / `perf_fft_e2e` are optional int64 ring buffers (ns); when
    provided, end-to-end latency from audio-block receive to OSC dispatch is
    written into them. `perf_idx_state` is a shared dict so the App can read
    the write index for ring statistics.
    """
    last_seq = 0
    last_fft_seq = 0
    lmh_len = perf_lmh_e2e.shape[0] if perf_lmh_e2e is not None else 0
    fft_len = perf_fft_e2e.shape[0] if perf_fft_e2e is not None else 0
    scaled_scratch = np.zeros(3, dtype=np.float64)
    while not stop.is_set():
        try:
            await asyncio.wait_for(sender_event.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        sender_event.clear()
        if stop.is_set():
            return
        seq, t_recv_ns, beat, bpm = features_store.read_scaled_into(scaled_scratch)
        if seq != last_seq:
            last_seq = seq
            g = get_master_gain()
            sender.send_lmh(scaled_scratch[0] * g, scaled_scratch[1] * g, scaled_scratch[2] * g)
            sender.send_bpm(bpm)
            if beat:
                sender.send_beat()
            if perf_lmh_e2e is not None and t_recv_ns:
                latency = time.perf_counter_ns() - t_recv_ns
                if latency > 0:
                    i = perf_idx_state["lmh"]
                    perf_lmh_e2e[i % lmh_len] = latency
                    perf_idx_state["lmh"] = i + 1
        if get_send_fft() and get_fft_enabled():
            kind = "raw_db" if get_send_raw_db() else "processed"
            fseq, frame, ft_recv_ns = fft_store.read(kind)
            if fseq != last_fft_seq and frame is not None:
                last_fft_seq = fseq
                if kind == "raw_db":
                    # Master gain is a feature-output multiplier; it does not
                    # apply to the raw dB monitor stream.
                    sender.send_fft(frame, get_db_floor())
                else:
                    sender.send_fft_processed(frame, get_master_gain())
                if perf_fft_e2e is not None and ft_recv_ns:
                    latency = time.perf_counter_ns() - ft_recv_ns
                    if latency > 0:
                        i = perf_idx_state["fft"]
                        perf_fft_e2e[i % fft_len] = latency
                        perf_idx_state["fft"] = i + 1
