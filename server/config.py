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
class BandCfg:
    lo_hz: float
    hi_hz: float


@dataclass
class DspCfg:
    # Three independent bandpass bands. Defaults: low cuts sub-rumble below 30 Hz;
    # high caps at 16 kHz to avoid empty top-octave on most consumer mics.
    low: BandCfg = field(default_factory=lambda: BandCfg(30.0, 250.0))
    mid: BandCfg = field(default_factory=lambda: BandCfg(250.0, 4000.0))
    high: BandCfg = field(default_factory=lambda: BandCfg(4000.0, 16000.0))
    tau: dict = field(default_factory=lambda: {"low": 0.15, "mid": 0.06, "high": 0.02})


@dataclass
class AutoscaleCfg:
    tau_attack_s: float = 0.05
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
    db_floor: float = -90.0
    db_ceiling: float = -10.0
    # Spatial-smear width (in octaves) applied to the per-bin peak follower
    # before the auto-scaler divides by it. 0 = each bin self-normalizes
    # independently (single-tone bins can over-attenuate themselves). Larger
    # values share normalization across neighboring bins so a single-frequency
    # spike still reads as taller-than-average against its spectral context.
    peak_smear_oct: float = 0.3
    # Spectral tilt added to the wire dB before the noise gate / peak follower:
    # +tilt_db_per_oct dB per octave above 1 kHz, -tilt_db_per_oct dB per octave
    # below. Compensates for the natural downward slope of music/voice spectra
    # so a single global noise_floor gates uniformly across frequency. 0 disables.
    tilt_db_per_oct: float = 3.0
    # When True, OSC + WS send the raw wire dB stream. When False (default),
    # they send the post-processed stream (smoothed, peak-normalized, gated,
    # tanh-compressed, strength-blended) — same semantics as L/M/H over OSC.
    send_raw_db: bool = False


@dataclass
class OscDest:
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass
class OscCfg:
    destinations: list = field(default_factory=lambda: [OscDest()])
    send_fft: bool = False


@dataclass
class UiCfg:
    # Visual peak-hold decay rate (per second) for the L/M/H bars and FFT
    # visualizers. Pure UI-side concern — does not affect DSP or OSC payload.
    peak_decay_per_s: float = 0.6


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
    ui: UiCfg = field(default_factory=UiCfg)


def _bands_dict(cfg: DspCfg) -> dict:
    return {
        "low":  {"lo_hz": cfg.low.lo_hz,  "hi_hz": cfg.low.hi_hz},
        "mid":  {"lo_hz": cfg.mid.lo_hz,  "hi_hz": cfg.mid.hi_hz},
        "high": {"lo_hz": cfg.high.lo_hz, "hi_hz": cfg.high.hi_hz},
    }


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

    known = {"audio", "dsp", "autoscale", "fft", "osc", "ws", "ui"}
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

    bands_raw = {k: d_raw.get(k) for k in ("low", "mid", "high") if isinstance(d_raw.get(k), dict)}
    if len(bands_raw) == 3:
        try:
            ok = V.validate_bands(bands_raw, sr_guess)
            cfg.dsp.low  = BandCfg(*ok["low"])
            cfg.dsp.mid  = BandCfg(*ok["mid"])
            cfg.dsp.high = BandCfg(*ok["high"])
        except Exception as e:
            log.warning("config dsp bands invalid (%s); using defaults", e)
    elif "low_hz" in d_raw and "high_hz" in d_raw:
        # Migrate legacy LP/BP/HP cutoffs into three bandpasses.
        try:
            lo = float(d_raw["low_hz"])
            hi = float(d_raw["high_hz"])
            low_floor = 30.0
            high_ceiling = min(16000.0, 0.45 * sr_guess - 1.0)
            migrated = {
                "low":  {"lo_hz": low_floor, "hi_hz": lo},
                "mid":  {"lo_hz": lo,        "hi_hz": hi},
                "high": {"lo_hz": hi,        "hi_hz": high_ceiling},
            }
            ok = V.validate_bands(migrated, sr_guess)
            cfg.dsp.low  = BandCfg(*ok["low"])
            cfg.dsp.mid  = BandCfg(*ok["mid"])
            cfg.dsp.high = BandCfg(*ok["high"])
            log.info("migrated legacy dsp.low_hz/high_hz to 3-bandpass schema")
        except Exception as e:
            log.warning("config dsp legacy migration failed (%s); using defaults", e)

    try:
        if "tau" in d_raw:
            cfg.dsp.tau = {**cfg.dsp.tau, **V.validate_tau(d_raw["tau"])}
    except Exception as e:
        log.warning("config dsp.tau invalid (%s); using defaults", e)

    # autoscale
    as_raw = raw.get("autoscale", {}) or {}
    try:
        ok = V.validate_autoscale(
            tau_attack_s=as_raw.get("tau_attack_s"),
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
    if isinstance(f_raw.get("peak_smear_oct"), (int, float)):
        v = float(f_raw["peak_smear_oct"])
        if 0.0 <= v <= 3.0:
            cfg.fft.peak_smear_oct = v
    if "tilt_db_per_oct" in f_raw:
        try:
            cfg.fft.tilt_db_per_oct = V.validate_fft_tilt_db_per_oct(f_raw["tilt_db_per_oct"])
        except Exception as e:
            log.warning("config fft.tilt_db_per_oct invalid (%s); using default", e)
    if isinstance(f_raw.get("send_raw_db"), bool):
        cfg.fft.send_raw_db = f_raw["send_raw_db"]

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

    # ui
    u_raw = raw.get("ui", {}) or {}
    try:
        if "peak_decay_per_s" in u_raw:
            cfg.ui.peak_decay_per_s = V.validate_peak_decay_per_s(u_raw["peak_decay_per_s"])
    except Exception as e:
        log.warning("config ui.peak_decay_per_s invalid (%s); using default", e)

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
