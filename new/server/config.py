"""Config dataclass, YAML load/save, debounced atomic Persister."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .control import validate as V

log = logging.getLogger(__name__)


@dataclass
class DeviceCfg:
    name: str | None = None
    index: int | None = None


@dataclass
class AudioCfg:
    device: DeviceCfg = field(default_factory=DeviceCfg)
    blocksize: int = 256
    channels: int = 1


@dataclass
class DspCfg:
    low_hz: float = 250.0
    high_hz: float = 4000.0
    tau: dict = field(default_factory=lambda: {"low": 0.15, "mid": 0.06, "high": 0.02})


@dataclass
class AutoscaleCfg:
    tau_release_s: float = 60.0
    noise_floor: float = 0.001
    strength: float = 1.0


@dataclass
class FftCfg:
    enabled: bool = False
    n_bins: int = 128
    window_size: int = 1024
    hop: int = 512
    f_min: float = 30.0
    db_floor: float = -80.0
    db_ceiling: float = 0.0


@dataclass
class OscDest:
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass
class OscCfg:
    destinations: list = field(default_factory=lambda: [OscDest()])
    send_fft: bool = False


@dataclass
class WsCfg:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    snapshot_hz: int = 60
    http_port: int = 8766  # static-file server for the UI (ES modules need http://)


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    dsp: DspCfg = field(default_factory=DspCfg)
    autoscale: AutoscaleCfg = field(default_factory=AutoscaleCfg)
    fft: FftCfg = field(default_factory=FftCfg)
    osc: OscCfg = field(default_factory=OscCfg)
    ws: WsCfg = field(default_factory=WsCfg)


def load_config(path: Path | str) -> Config:
    path = Path(path)
    cfg = Config()
    if not path.exists():
        return cfg
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except Exception as e:
        log.warning("config load failed (%s); using defaults", e)
        return cfg
    if not isinstance(raw, dict):
        log.warning("config root is not a mapping; using defaults")
        return cfg

    known = {"audio", "dsp", "autoscale", "fft", "osc", "ws"}
    for k in raw:
        if k not in known:
            log.warning("config: unknown top-level key %r (ignored)", k)

    # audio
    a_raw = raw.get("audio", {}) or {}
    dev_raw = a_raw.get("device", {}) or {}
    cfg.audio.device = DeviceCfg(
        name=dev_raw.get("name") if isinstance(dev_raw.get("name"), str) else None,
        index=dev_raw.get("index") if isinstance(dev_raw.get("index"), int) else None,
    )
    if isinstance(a_raw.get("blocksize"), int) and 32 <= a_raw["blocksize"] <= 4096:
        cfg.audio.blocksize = a_raw["blocksize"]
    if isinstance(a_raw.get("channels"), int) and a_raw["channels"] in (1, 2):
        cfg.audio.channels = a_raw["channels"]

    # dsp - validate as if it were a control message; bad values fall back
    d_raw = raw.get("dsp", {}) or {}
    sr_guess = 48000.0  # validation here uses a guess; real sr applies on retune
    try:
        if "low_hz" in d_raw and "high_hz" in d_raw:
            lo, hi = V.validate_band_cutoffs(d_raw["low_hz"], d_raw["high_hz"], sr_guess)
            cfg.dsp.low_hz, cfg.dsp.high_hz = lo, hi
    except Exception as e:
        log.warning("config dsp.low_hz/high_hz invalid (%s); using defaults", e)
    try:
        if "tau" in d_raw:
            cfg.dsp.tau = {**cfg.dsp.tau, **V.validate_tau(d_raw["tau"])}
    except Exception as e:
        log.warning("config dsp.tau invalid (%s); using defaults", e)

    # autoscale
    as_raw = raw.get("autoscale", {}) or {}
    try:
        ok = V.validate_autoscale(
            tau_release_s=as_raw.get("tau_release_s"),
            noise_floor=as_raw.get("noise_floor"),
            strength=as_raw.get("strength"),
        )
        for k, v in ok.items():
            setattr(cfg.autoscale, k, v)
    except Exception as e:
        log.warning("config autoscale invalid (%s); using defaults", e)

    # fft
    f_raw = raw.get("fft", {}) or {}
    if isinstance(f_raw.get("enabled"), bool):
        cfg.fft.enabled = f_raw["enabled"]
    try:
        if "n_bins" in f_raw:
            cfg.fft.n_bins = V.validate_n_fft_bins(f_raw["n_bins"])
    except Exception as e:
        log.warning("config fft.n_bins invalid (%s); using default", e)
    if isinstance(f_raw.get("window_size"), int) and f_raw["window_size"] in (256, 512, 1024, 2048, 4096):
        cfg.fft.window_size = f_raw["window_size"]
    if isinstance(f_raw.get("hop"), int) and 32 <= f_raw["hop"] <= cfg.fft.window_size:
        cfg.fft.hop = f_raw["hop"]
    if isinstance(f_raw.get("f_min"), (int, float)) and 1.0 <= f_raw["f_min"] <= 1000.0:
        cfg.fft.f_min = float(f_raw["f_min"])
    if isinstance(f_raw.get("db_floor"), (int, float)):
        cfg.fft.db_floor = float(f_raw["db_floor"])
    if isinstance(f_raw.get("db_ceiling"), (int, float)):
        cfg.fft.db_ceiling = float(f_raw["db_ceiling"])

    # osc
    o_raw = raw.get("osc", {}) or {}
    dests = o_raw.get("destinations")
    if isinstance(dests, list) and dests:
        good = []
        for d in dests:
            if isinstance(d, dict) and isinstance(d.get("host"), str) and isinstance(d.get("port"), int):
                good.append(OscDest(host=d["host"], port=d["port"]))
        if good:
            cfg.osc.destinations = good
    if isinstance(o_raw.get("send_fft"), bool):
        cfg.osc.send_fft = o_raw["send_fft"]

    # ws
    w_raw = raw.get("ws", {}) or {}
    if isinstance(w_raw.get("enabled"), bool):
        cfg.ws.enabled = w_raw["enabled"]
    if isinstance(w_raw.get("host"), str):
        cfg.ws.host = w_raw["host"]
    if isinstance(w_raw.get("port"), int):
        cfg.ws.port = w_raw["port"]
    if isinstance(w_raw.get("http_port"), int):
        cfg.ws.http_port = w_raw["http_port"]
    try:
        if "snapshot_hz" in w_raw:
            cfg.ws.snapshot_hz = V.validate_ws_snapshot_hz(w_raw["snapshot_hz"])
    except Exception as e:
        log.warning("config ws.snapshot_hz invalid (%s); using default", e)

    return cfg


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def write_yaml_atomic(path: Path | str, data: dict) -> None:
    """Atomic write: tmp file then os.replace. Never partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class Persister:
    """Drag-aware debounced persister for config.yaml.

    Schedules atomic writes on the asyncio loop. Two debounce levels:
      - commit=False (drag): 1s delay, but a 250ms cap from the *first*
        unsaved change ensures we never lag too far behind.
      - commit=True / discrete events: 50ms delay, also capped at 250ms.
    """

    DRAG_DELAY = 1.0
    COMMIT_DELAY = 0.05
    MAX_DELAY = 0.25

    def __init__(self, path: Path | str, get_state):
        self.path = Path(path)
        self._get_state = get_state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._earliest_deadline: float | None = None
        self._dirty = False

    def attach(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def request(self, commit: bool):
        if self._loop is None:
            return
        self._dirty = True
        now = self._loop.time()
        delay = self.COMMIT_DELAY if commit else self.DRAG_DELAY
        target = now + delay
        if self._earliest_deadline is None:
            self._earliest_deadline = now + self.MAX_DELAY
        target = min(target, self._earliest_deadline)
        if self._handle is not None:
            self._handle.cancel()
        self._handle = self._loop.call_at(target, self._flush)

    def _flush(self):
        self._handle = None
        self._earliest_deadline = None
        if not self._dirty:
            return
        self._dirty = False
        try:
            data = self._get_state()
            write_yaml_atomic(self.path, data)
        except Exception as e:
            log.warning("persist failed: %s", e)

    def flush_now_sync(self):
        """Synchronous flush for shutdown. Cancels pending timer."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if not self._dirty:
            return
        self._dirty = False
        try:
            data = self._get_state()
            write_yaml_atomic(self.path, data)
        except Exception as e:
            log.warning("final persist failed: %s", e)
