"""OSC UDP sender. One SimpleUDPClient per destination."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from ..config import OscDest

log = logging.getLogger(__name__)


class OscSender:
    def __init__(self, destinations: list[OscDest]):
        self._dests = list(destinations)
        self._clients = [SimpleUDPClient(d.host, d.port) for d in self._dests]

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
        msg = [float(low), float(mid), float(high)]
        for c in self._clients:
            try:
                c.send_message("/audio/lmh", msg)
            except Exception as e:
                log.debug("osc lmh send failed: %s", e)

    def send_fft(self, bins: np.ndarray, db_floor: float) -> None:
        # OSC 'f' tag is 32-bit float. The wire array contains -1000 sentinels
        # for empty log bins; np.maximum clamps both those (and any sub-floor
        # real values, harmless) to db_floor in a single fused C pass. The
        # output is a fresh array so we don't mutate the publisher's buffer
        # (it's still being read by the WS encoder concurrently).
        arr = np.maximum(bins, np.float32(db_floor))
        payload = arr.tolist()
        for c in self._clients:
            try:
                c.send_message("/audio/fft", payload)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)

    def send_fft_processed(self, bins: np.ndarray, gain: float = 1.0) -> None:
        # Post-processed values are already in [0, 1] (no sentinels). master
        # gain >1 may push them above 1.0 — that's by design and downstream
        # consumers must accept it.
        arr = np.asarray(bins, dtype=np.float32)
        if gain != 1.0:
            arr = arr * np.float32(gain)
        payload = arr.tolist()
        for c in self._clients:
            try:
                c.send_message("/audio/fft", payload)
            except Exception as e:
                log.debug("osc fft send failed: %s", e)


async def osc_sender_task(stop, sender_event: asyncio.Event, sender: OscSender,
                          features_store, fft_store, get_send_fft, get_fft_enabled,
                          get_db_floor, get_send_raw_db, get_master_gain):
    """Wakes on sender_event; sends one /audio/lmh per audio block."""
    last_seq = 0
    last_fft_seq = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(sender_event.wait(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        sender_event.clear()
        if stop.is_set():
            return
        seq, _raw, scaled = features_store.read()
        if seq != last_seq:
            last_seq = seq
            g = get_master_gain()
            sender.send_lmh(scaled[0] * g, scaled[1] * g, scaled[2] * g)
        if get_send_fft() and get_fft_enabled():
            kind = "raw_db" if get_send_raw_db() else "processed"
            fseq, frame = fft_store.read(kind)
            if fseq != last_fft_seq and frame is not None:
                last_fft_seq = fseq
                if kind == "raw_db":
                    # Master gain is a feature-output multiplier; it does not
                    # apply to the raw dB monitor stream.
                    sender.send_fft(frame, get_db_floor())
                else:
                    sender.send_fft_processed(frame, get_master_gain())
