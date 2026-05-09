"""Config dataclass, YAML load/save, debounced atomic Persister.

There are NO hardcoded defaults in this file. Every value is loaded from
the YAML config file (default: ``<repo>/configs/main.yaml``). The dataclasses
below are bare typed schemas; ``load_config`` constructs them from a
fully-populated dict read off disk. The committed ``configs/main.yaml`` is
the canonical default state — the runtime persister keeps it complete.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .control import validate as V

log = logging.getLogger(__name__)

CANONICAL_CONFIG_PATH = (Path(__file__).resolve().parent.parent / "configs" / "main.yaml")


@dataclass
class DeviceCfg:
    name: str | None
    index: int | None


@dataclass
class AudioCfg:
    device: DeviceCfg
    blocksize: int
    channels: int


@dataclass
class BandCfg:
    lo_hz: float
    hi_hz: float


@dataclass
class DspCfg:
    low: BandCfg
    mid: BandCfg
    high: BandCfg
    tau: dict
    tau_attack: dict


@dataclass
class AutoscaleCfg:
    tau_attack_s: float
    tau_release_s: float
    noise_floor: float
    strength: float
    master_gain: float


@dataclass
class FftCfg:
    enabled: bool
    n_bins: int
    window_size: int
    hop: int
    f_min: float
    db_floor: float
    db_ceiling: float
    peak_smear_oct: float
    tilt_db_per_oct: float
    send_raw_db: bool


@dataclass
class BeatCfg:
    sensitivity: float    # K_HIGH multiplier on adaptive threshold
    refractory_s: float   # min inter-onset interval (seconds)
    slow_tau_s: float     # slow envelope tau (seconds)


@dataclass
class OscDest:
    host: str
    port: int


@dataclass
class OscCfg:
    destinations: list
    send_fft: bool


@dataclass
class UiCfg:
    peak_decay_per_s: float
    layout: dict


@dataclass
class WsCfg:
    enabled: bool
    host: str
    port: int
    snapshot_hz: int
    http_port: int


@dataclass
class Config:
    audio: AudioCfg
    dsp: DspCfg
    autoscale: AutoscaleCfg
    fft: FftCfg
    beat: BeatCfg
    osc: OscCfg
    ws: WsCfg
    ui: UiCfg


def _build_config(d: dict) -> Config:
    """Build a Config from a fully-populated dict.

    Missing keys raise KeyError — config files are expected to be
    complete (the runtime persister always writes the full state).
    Values are routed through validators where the live control path has
    them, so a corrupt-but-parseable YAML still surfaces a clear error.
    """
    a = d["audio"]
    dev = a.get("device") or {}
    audio = AudioCfg(
        device=DeviceCfg(name=dev.get("name"), index=dev.get("index")),
        blocksize=int(a["blocksize"]),
        channels=int(a["channels"]),
    )

    ds = d["dsp"]
    bands_raw = {k: ds[k] for k in ("low", "mid", "high")}
    # validate_bands needs a sample rate; 48k is the standard guess used
    # at config-load time (real sr applies on retune in main.App).
    ok = V.validate_bands(bands_raw, 48000.0)
    dsp = DspCfg(
        low=BandCfg(*ok["low"]),
        mid=BandCfg(*ok["mid"]),
        high=BandCfg(*ok["high"]),
        tau=V.validate_tau(ds["tau"]),
        tau_attack=V.validate_tau(ds["tau_attack"]),
    )

    asd = d["autoscale"]
    ok_as = V.validate_autoscale(
        tau_attack_s=asd["tau_attack_s"],
        tau_release_s=asd["tau_release_s"],
        noise_floor=asd["noise_floor"],
        strength=asd["strength"],
        master_gain=asd["master_gain"],
    )
    autoscale = AutoscaleCfg(**ok_as)

    fd = d["fft"]
    fft = FftCfg(
        enabled=bool(fd["enabled"]),
        n_bins=V.validate_n_fft_bins(fd["n_bins"]),
        window_size=int(fd["window_size"]),
        hop=int(fd["hop"]),
        f_min=float(fd["f_min"]),
        db_floor=float(fd["db_floor"]),
        db_ceiling=float(fd["db_ceiling"]),
        peak_smear_oct=float(fd["peak_smear_oct"]),
        tilt_db_per_oct=V.validate_fft_tilt_db_per_oct(fd["tilt_db_per_oct"]),
        send_raw_db=bool(fd["send_raw_db"]),
    )

    # Beat detector — section is optional (added after the initial schema)
    # so legacy main.yaml files without a `beat:` block keep working with
    # the BeatTracker's compiled-in defaults.
    bd = d.get("beat") or {}
    beat_defaults = {
        "sensitivity": 1.8,
        "refractory_s": 0.25,
        "slow_tau_s": 0.30,
    }
    ok_beat = V.validate_beat(
        sensitivity=bd.get("sensitivity", beat_defaults["sensitivity"]),
        refractory_s=bd.get("refractory_s", beat_defaults["refractory_s"]),
        slow_tau_s=bd.get("slow_tau_s", beat_defaults["slow_tau_s"]),
    )
    beat = BeatCfg(**ok_beat)

    od = d["osc"]
    dests = [OscDest(host=str(x["host"]), port=int(x["port"])) for x in od["destinations"]]
    osc = OscCfg(destinations=dests, send_fft=bool(od["send_fft"]))

    wd = d["ws"]
    ws = WsCfg(
        enabled=bool(wd["enabled"]),
        host=str(wd["host"]),
        port=int(wd["port"]),
        snapshot_hz=V.validate_ws_snapshot_hz(wd["snapshot_hz"]),
        http_port=int(wd["http_port"]),
    )

    ud = d["ui"]
    ui = UiCfg(
        peak_decay_per_s=V.validate_peak_decay_per_s(ud["peak_decay_per_s"]),
        layout=V.validate_ui_layout(ud["layout"]),
    )

    return Config(audio=audio, dsp=dsp, autoscale=autoscale, fft=fft, beat=beat, osc=osc, ws=ws, ui=ui)


def load_config(path: Path | str) -> Config:
    """Load and validate a config file. No fallbacks, no merging.

    Raises ``FileNotFoundError`` if missing. Raises ``ValueError`` if the
    file parses but a section is missing or invalid (missing keys surface
    as KeyError from ``_build_config``, re-wrapped here for clarity).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a YAML mapping: {path}")
    try:
        return _build_config(raw)
    except KeyError as e:
        raise ValueError(f"config {path} missing required key: {e}") from e


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
