"""Server entry point: orchestrates startup/shutdown, owns all state.

Threads:
  - PortAudio C thread runs AudioCallback (memcpy + signal).
  - DSP worker thread (filter + RMS + smoother + autoscaler -> FeatureStore).
  - FFT worker thread (optional; window + rfft + log-bin -> FFTStore).
  - Main asyncio loop: OSC sender, WS server, broadcaster, status.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np

from .audio import devices as devmod
from .audio.callback import AudioCallback
from .audio.ringbuffer import SlotRing
from .audio.stream import StreamHandle, open_input_stream
from .config import Config, OscDest, Persister, config_to_dict, load_config
from .control.dispatcher import Dispatcher
from .dsp.features import AutoScaler, ExpSmoother
from .dsp.fft import FFTWorker
from .dsp.filters import FilterBank
from .dsp.worker import DSPWorker
from .io.http_server import StaticHTTPServer
from .io.osc_sender import OscDest as OscDestSender, OscSender, osc_sender_task
from .io.stores import FFTStore, FeatureStore
from .io.ws_server import WSServer

log = logging.getLogger(__name__)


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="audio-server", description="Realtime audio feature server")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--no-ws", action="store_true", help="disable the WebSocket server (headless OSC)")
    p.add_argument("--open", action="store_true", help="open the bundled UI in the default browser")
    p.add_argument("--device", type=int, default=None, help="override input device index")
    p.add_argument("--log-level", default="INFO", help="logging level")
    return p.parse_args(argv)


class App:
    """Long-lived state shared across threads. Mutations happen on the asyncio loop."""

    def __init__(self, args, cfg: Config, config_path: Path):
        self.args = args
        self.cfg = cfg
        self.config_path = config_path
        self.config_dir = config_path.parent.resolve() if config_path.parent != Path("") else Path.cwd()

        # ----- Threading primitives -----
        self.dsp_event = threading.Event()
        self.fft_event = threading.Event()
        self.fft_enabled = threading.Event()
        if cfg.fft.enabled:
            self.fft_enabled.set()
        self.stop_flag = threading.Event()

        # ----- Stores -----
        self.features_store = FeatureStore()
        self.fft_store = FFTStore()

        # ----- Ring buffer -----
        self.ring = SlotRing(n_slots_pow2=32, blocksize=cfg.audio.blocksize)

        # ----- Perf rings (int64 ns) -----
        self.perf_cb  = np.zeros(128, dtype=np.int64)
        self.perf_dsp = np.zeros(128, dtype=np.int64)
        self.perf_fft = np.zeros(64,  dtype=np.int64)
        self.perf_ws  = np.zeros(32,  dtype=np.int64)

        # ----- Async / loop bits -----
        self.loop: asyncio.AbstractEventLoop | None = None
        self.sender_event: asyncio.Event | None = None  # set by DSP via call_soon_threadsafe

        # ----- Initialised by start() -----
        self.callback: AudioCallback | None = None
        self.stream: StreamHandle | None = None
        self.filter_bank: FilterBank | None = None
        self.smoother: ExpSmoother | None = None
        self.auto_scaler: AutoScaler | None = None
        self.dsp_worker: DSPWorker | None = None
        self.fft_worker: FFTWorker | None = None
        self.osc_sender: OscSender | None = None
        self.osc_task: asyncio.Task | None = None
        self.ws: WSServer | None = None
        self.http: StaticHTTPServer | None = None
        self.persister: Persister | None = None

        self._filter_retune_handle: asyncio.TimerHandle | None = None
        self._device_switch_lock: asyncio.Lock | None = None

    # ----------------- starting up -----------------
    def _build_pipeline_for_sr(self, sr: float) -> None:
        cfg = self.cfg
        self.filter_bank = FilterBank(
            sr=sr,
            low_hz=cfg.dsp.low_hz,
            high_hz=cfg.dsp.high_hz,
            blocksize=cfg.audio.blocksize,
        )
        self.smoother = ExpSmoother(sr=sr, blocksize=cfg.audio.blocksize, tau=cfg.dsp.tau)
        self.auto_scaler = AutoScaler(
            sr=sr,
            blocksize=cfg.audio.blocksize,
            tau_release_s=cfg.autoscale.tau_release_s,
            noise_floor=cfg.autoscale.noise_floor,
            strength=cfg.autoscale.strength,
        )

    def _signal_dsp_published(self) -> None:
        """Called from the DSP worker thread on each publish.

        Posts sender_event.set() onto the asyncio loop so the OSC sender
        wakes promptly. Schedules from a worker thread, so we use
        call_soon_threadsafe.
        """
        loop = self.loop
        ev = self.sender_event
        if loop is None or ev is None:
            return
        try:
            loop.call_soon_threadsafe(ev.set)
        except RuntimeError:
            pass  # loop closed during shutdown

    def _signal_fft_published(self) -> None:
        # FFT path has its own pacing on the WS broadcaster; OSC FFT sends
        # are handled by the same sender_event the DSP path already pings.
        pass

    def start(self) -> None:
        cfg = self.cfg
        self.loop = asyncio.get_event_loop()
        self.sender_event = asyncio.Event()
        self._device_switch_lock = asyncio.Lock()

        # -------- Resolve & open audio device --------
        device_idx = devmod.resolve_initial_device(cfg.audio.device, self.args.device)
        log.info("initial device index: %s", device_idx)

        # Build callback (perf ring is shared)
        # We need the stream's sample rate before building DSP.
        # Open a placeholder callback first, then swap. Cleanest: build callback,
        # open stream (which calls back at start()), build DSP from sr, START stream.
        self.callback = AudioCallback(
            ring=self.ring,
            dsp_event=self.dsp_event,
            fft_event=self.fft_event,
            channels=cfg.audio.channels,
            blocksize=cfg.audio.blocksize,
            perf_ring=self.perf_cb,
        )
        self.stream = open_input_stream(
            device=device_idx,
            blocksize=cfg.audio.blocksize,
            channels=cfg.audio.channels,
            callback=self.callback,
        )

        self._build_pipeline_for_sr(self.stream.samplerate)

        # -------- Workers --------
        self.dsp_worker = DSPWorker(
            ring=self.ring,
            dsp_event=self.dsp_event,
            stop_flag=self.stop_flag,
            filter_bank=self.filter_bank,
            smoother=self.smoother,
            auto_scaler=self.auto_scaler,
            features_store=self.features_store,
            on_publish=self._signal_dsp_published,
            blocksize=cfg.audio.blocksize,
            perf_ring=self.perf_dsp,
        )
        self.fft_worker = FFTWorker(
            ring=self.ring,
            fft_event=self.fft_event,
            fft_enabled=self.fft_enabled,
            stop_flag=self.stop_flag,
            fft_store=self.fft_store,
            on_publish=self._signal_dsp_published,  # share — OSC FFT uses same wakeup
            blocksize=cfg.audio.blocksize,
            sr=self.stream.samplerate,
            window_size=cfg.fft.window_size,
            hop=cfg.fft.hop,
            n_bins=cfg.fft.n_bins,
            f_min=cfg.fft.f_min,
            perf_ring=self.perf_fft,
            db_floor=cfg.fft.db_floor,
        )
        self.dsp_worker.start()
        self.fft_worker.start()

        # -------- OSC --------
        dests = [OscDestSender(host=d.host, port=d.port) for d in cfg.osc.destinations]
        self.osc_sender = OscSender(dests)
        self.osc_sender.send_meta(
            sr=int(self.stream.samplerate),
            blocksize=cfg.audio.blocksize,
            n_fft_bins=cfg.fft.n_bins,
            low_hz=cfg.dsp.low_hz,
            high_hz=cfg.dsp.high_hz,
        )
        self.osc_task = self.loop.create_task(
            osc_sender_task(
                stop=self.stop_flag,
                sender_event=self.sender_event,
                sender=self.osc_sender,
                features_store=self.features_store,
                fft_store=self.fft_store,
                get_send_fft=lambda: self.cfg.osc.send_fft,
                get_fft_enabled=lambda: self.fft_enabled.is_set(),
            ),
            name="osc-sender",
        )

        # -------- Persister --------
        self.persister = Persister(self.config_path, get_state=lambda: config_to_dict(self.cfg))
        self.persister.attach(self.loop)

        # -------- WebSocket server (optional) --------
        if cfg.ws.enabled and not self.args.no_ws:
            dispatcher = Dispatcher(self)
            self.ws = WSServer(
                host=cfg.ws.host,
                port=cfg.ws.port,
                snapshot_hz=cfg.ws.snapshot_hz,
                features_store=self.features_store,
                fft_store=self.fft_store,
                get_meta=self.snapshot_meta,
                get_devices=lambda: self.list_devices_with_probe(False),
                get_presets=self.list_presets,
                get_server_status=self.snapshot_server_status,
                get_fft_enabled=lambda: self.fft_enabled.is_set(),
                dispatcher_handle=dispatcher,
            )
            self.ws.perf_ring = self.perf_ws

            # Static HTTP server for the UI (ES modules need http://, not file://).
            ui_root = (Path(__file__).resolve().parent.parent / "ui")
            if ui_root.exists():
                self.http = StaticHTTPServer(host=cfg.ws.host, port=cfg.ws.http_port, root=ui_root)
                try:
                    self.http.start()
                except OSError as e:
                    log.warning("static http server failed to bind on port %d: %s", cfg.ws.http_port, e)
                    self.http = None
            else:
                log.warning("ui directory not found at %s; static http server skipped", ui_root)
        else:
            self.ws = None

        # -------- Start stream LAST, after workers are up --------
        self.stream.start()

        # -------- Optional UI launch --------
        if self.args.open:
            if self.ws is not None and self.http is not None:
                url = f"http://{cfg.ws.host}:{cfg.ws.http_port}/index.html"
                try:
                    webbrowser.open(url)
                except Exception as e:
                    log.warning("failed to open browser: %s", e)
            else:
                log.info("--open ignored: WS or HTTP server disabled")

    # ----------------- snapshots / readers exposed to dispatcher + ws -----------------
    def current_sr(self) -> float:
        return float(self.stream.samplerate) if self.stream else 48000.0

    def snapshot_meta(self) -> dict:
        cfg = self.cfg
        device = {"index": None, "name": None}
        if self.stream is not None:
            try:
                info = devmod.device_info(int(self.stream.device)) if isinstance(self.stream.device, int) else None
                if info:
                    device = {"index": int(self.stream.device), "name": info.get("name")}
            except Exception:
                pass
        return {
            "sr": int(self.current_sr()),
            "blocksize": cfg.audio.blocksize,
            "n_fft_bins": cfg.fft.n_bins,
            "low_hz": cfg.dsp.low_hz,
            "high_hz": cfg.dsp.high_hz,
            "fft_enabled": self.fft_enabled.is_set(),
            "fft_db_floor": cfg.fft.db_floor,
            "fft_db_ceiling": cfg.fft.db_ceiling,
            "tau": dict(cfg.dsp.tau),
            "autoscale": {
                "tau_release_s": cfg.autoscale.tau_release_s,
                "noise_floor": cfg.autoscale.noise_floor,
                "strength": cfg.autoscale.strength,
            },
            "ws_snapshot_hz": self.ws.snapshot_hz if self.ws else cfg.ws.snapshot_hz,
            "device": device,
        }

    def list_devices_with_probe(self, probe: bool) -> list:
        items = devmod.list_input_devices()
        if probe:
            results = devmod.signal_active_probe(items, duration=0.2)
            for it in items:
                it["probed_signal"] = bool(results.get(it["index"], False))
                it["probed_at"] = time.strftime("%H:%M:%S")
        return items

    def list_presets(self) -> list[dict]:
        items = []
        try:
            for p in sorted(self.config_dir.glob("preset-*.yaml")):
                name = p.stem[len("preset-"):]
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                saved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime))
                items.append({"name": name, "saved_at": saved_at})
        except Exception:
            pass
        items.sort(key=lambda x: x["saved_at"], reverse=True)
        return items

    def preset_path(self, name: str) -> Path:
        # Caller has already passed `name` through validate_preset_name, so it's
        # a stripped, anchored, character-class-restricted string.
        return self.config_dir / f"preset-{name}.yaml"

    def preset_body(self, name: str) -> dict:
        cfg = self.cfg
        return {
            "name": name,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dsp": {
                "low_hz": cfg.dsp.low_hz,
                "high_hz": cfg.dsp.high_hz,
                "tau": dict(cfg.dsp.tau),
            },
            "autoscale": {
                "tau_release_s": cfg.autoscale.tau_release_s,
                "noise_floor": cfg.autoscale.noise_floor,
                "strength": cfg.autoscale.strength,
            },
            "fft": {
                "n_bins": cfg.fft.n_bins,
                "window_size": cfg.fft.window_size,
                "hop": cfg.fft.hop,
                "f_min": cfg.fft.f_min,
            },
        }

    # ----------------- perf summary -----------------
    @staticmethod
    def _ring_stats(ring: np.ndarray, count: int) -> tuple[float, float]:
        if count == 0:
            return 0.0, 0.0
        n = min(count, ring.shape[0])
        valid = ring[:n] if count <= ring.shape[0] else ring
        # filter zeros — slot may be unwritten
        nz = valid[valid > 0]
        if nz.size == 0:
            return 0.0, 0.0
        avg_ns = float(np.mean(nz))
        p95_ns = float(np.percentile(nz, 95))
        return avg_ns / 1e6, p95_ns / 1e6  # ms

    def snapshot_server_status(self) -> dict:
        cfg = self.cfg
        sr = self.current_sr()
        block_period_ms = cfg.audio.blocksize / sr * 1000.0
        hop_period_ms = (cfg.fft.hop / sr) * 1000.0
        ws_hz = self.ws.snapshot_hz if self.ws else cfg.ws.snapshot_hz
        ws_period_ms = 1000.0 / max(ws_hz, 1)

        cb_avg, cb_p95 = self._ring_stats(self.perf_cb,  self.callback.perf_idx if self.callback else 0)
        ds_avg, ds_p95 = self._ring_stats(self.perf_dsp, self.dsp_worker.perf_idx if self.dsp_worker else 0)
        ff_avg, ff_p95 = self._ring_stats(self.perf_fft, self.fft_worker.perf_idx if self.fft_worker else 0)
        ws_avg, ws_p95 = self._ring_stats(self.perf_ws,  self.ws._perf_idx if self.ws else 0)

        def load(avg_ms, period_ms):
            if period_ms <= 0:
                return 0.0
            return float(avg_ms / period_ms * 100.0)

        return {
            "type": "server_status",
            "cb_overruns": int(self.callback.cb_overruns) if self.callback else 0,
            "dsp_drops": int(self.dsp_worker.dsp_drops) if self.dsp_worker else 0,
            "fft_drops": int(self.fft_worker.fft_drops) if self.fft_worker else 0,
            "perf": {
                "block_period_ms": block_period_ms,
                "hop_period_ms": hop_period_ms,
                "ws_period_ms": ws_period_ms,
                "cb":  {"avg_ms": cb_avg, "p95_ms": cb_p95, "load_pct": load(cb_avg, block_period_ms)},
                "dsp": {"avg_ms": ds_avg, "p95_ms": ds_p95, "load_pct": load(ds_avg, block_period_ms)},
                "fft": {"avg_ms": ff_avg, "p95_ms": ff_p95, "load_pct": load(ff_avg, hop_period_ms),
                        "enabled": self.fft_enabled.is_set()},
                "ws":  {"avg_ms": ws_avg, "p95_ms": ws_p95, "load_pct": load(ws_avg, ws_period_ms)},
            },
        }

    # ----------------- mutators called from dispatcher -----------------
    def schedule_filter_retune(self, low_hz: float, high_hz: float) -> None:
        """Server-side 50ms debounce of cutoff retunes."""
        loop = self.loop
        if loop is None:
            return
        if self._filter_retune_handle is not None:
            self._filter_retune_handle.cancel()
        self._filter_retune_handle = loop.call_later(
            0.05, lambda: self.filter_bank.retune(low_hz, high_hz)
        )

    async def hot_switch_device(self, new_idx: int) -> None:
        """Tear down stream, rebuild for new device, start fresh.

        The DSP/FFT workers stay alive — they idle on dsp_event/fft_event
        timeouts while the stream is down (callback fires nothing).
        """
        async with self._device_switch_lock:
            old_stream = self.stream
            old_stream.stop()
            old_stream.close()

            # Reset state
            self.ring.reset()
            self.filter_bank.reset_state()
            self.auto_scaler.reset()
            self.smoother.reset()
            self.dsp_worker.read_block_idx = 0
            self.fft_worker.reset()
            # zero perf rings so new sr's load is visible cleanly
            self.perf_cb.fill(0); self.perf_dsp.fill(0); self.perf_fft.fill(0); self.perf_ws.fill(0)
            self.callback.perf_idx = 0
            self.dsp_worker.perf_idx = 0
            self.fft_worker.perf_idx = 0
            if self.ws is not None:
                self.ws._perf_idx = 0

            # Open new stream
            self.stream = open_input_stream(
                device=new_idx,
                blocksize=self.cfg.audio.blocksize,
                channels=self.cfg.audio.channels,
                callback=self.callback,
            )
            new_sr = self.stream.samplerate

            # Rebuild filter + autoscaler + smoother for new sr (alphas depend on sr)
            self.filter_bank = FilterBank(
                sr=new_sr,
                low_hz=self.cfg.dsp.low_hz,
                high_hz=self.cfg.dsp.high_hz,
                blocksize=self.cfg.audio.blocksize,
            )
            self.smoother = ExpSmoother(
                sr=new_sr,
                blocksize=self.cfg.audio.blocksize,
                tau=self.cfg.dsp.tau,
            )
            self.auto_scaler = AutoScaler(
                sr=new_sr,
                blocksize=self.cfg.audio.blocksize,
                tau_release_s=self.cfg.autoscale.tau_release_s,
                noise_floor=self.cfg.autoscale.noise_floor,
                strength=self.cfg.autoscale.strength,
            )
            # Swap into worker
            self.dsp_worker.filter_bank = self.filter_bank
            self.dsp_worker.smoother = self.smoother
            self.dsp_worker.auto_scaler = self.auto_scaler
            # FFT worker rebuild for sr
            self.fft_worker.reconfigure(sr=new_sr)

            # Update cfg with the device choice
            info = devmod.device_info(new_idx) or {}
            self.cfg.audio.device.index = new_idx
            self.cfg.audio.device.name = info.get("name")

            # Start the new stream
            self.stream.start()

            # Push fresh meta over OSC
            self.osc_sender.send_meta(
                sr=int(new_sr),
                blocksize=self.cfg.audio.blocksize,
                n_fft_bins=self.cfg.fft.n_bins,
                low_hz=self.cfg.dsp.low_hz,
                high_hz=self.cfg.dsp.high_hz,
            )

    # ----------------- shutdown -----------------
    async def shutdown(self) -> None:
        log.info("shutdown: stopping audio stream")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        log.info("shutdown: stopping workers")
        self.stop_flag.set()
        self.dsp_event.set()
        self.fft_event.set()
        for w in (self.dsp_worker, self.fft_worker):
            if w is not None:
                w.join(timeout=1.0)
        if self.osc_task is not None:
            self.osc_task.cancel()
            try:
                await self.osc_task
            except (asyncio.CancelledError, Exception):
                pass
        if self.ws is not None:
            await self.ws.stop()
        if self.http is not None:
            self.http.stop()
        if self.persister is not None:
            self.persister.flush_now_sync()


async def _run(args, cfg, config_path):
    app = App(args, cfg, config_path)
    app.start()
    if app.ws is not None:
        await app.ws.start()

    stop_event = asyncio.Event()

    def _signal_handler():
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows fallback

    try:
        await stop_event.wait()
    finally:
        await app.shutdown()


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if args.no_ws:
        cfg.ws.enabled = False
    try:
        asyncio.run(_run(args, cfg, config_path))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
