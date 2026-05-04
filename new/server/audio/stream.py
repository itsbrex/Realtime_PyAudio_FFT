"""sounddevice.InputStream lifecycle wrapper."""
from __future__ import annotations

import logging

import sounddevice as sd

log = logging.getLogger(__name__)


class StreamHandle:
    def __init__(self, stream: sd.InputStream):
        self.stream = stream
        self.samplerate = float(stream.samplerate)
        self.latency = float(getattr(stream, "latency", 0.0) or 0.0)
        self.blocksize = int(stream.blocksize) if stream.blocksize else 0
        self.channels = int(stream.channels)
        self.device = stream.device

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop(ignore_errors=True)
        except Exception:
            pass

    def close(self):
        try:
            self.stream.close(ignore_errors=True)
        except Exception:
            pass


def open_input_stream(device: int | None, blocksize: int, channels: int, callback) -> StreamHandle:
    """Open the input stream and return a handle. Sample rate is device-driven."""
    stream = sd.InputStream(
        device=device,
        samplerate=None,
        blocksize=blocksize,
        channels=channels,
        dtype="float32",
        callback=callback,
        latency="low",
    )
    h = StreamHandle(stream)
    log.info(
        "stream open: device=%s sr=%.0f blocksize=%d channels=%d latency=%.1fms",
        device, h.samplerate, blocksize, channels, h.latency * 1000.0
    )
    if h.latency > 2.0 * (blocksize / max(h.samplerate, 1.0)):
        log.warning(
            "device latency %.1fms exceeds 2x blocksize/sr; check Audio MIDI Setup buffer",
            h.latency * 1000.0,
        )
    return h
