"""Inbound WS message dispatcher: validate + apply + persist + reply.

Returns (targeted_replies, broadcasts) — never raises. The WS layer turns
exceptions into {"type":"error",...}; the dispatcher prefers to do that
itself with a precise reason.
"""
from __future__ import annotations

import asyncio
import logging

import yaml

from . import validate as V
from ..config import write_yaml_atomic

log = logging.getLogger(__name__)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge: overlay wins; nested dicts merge key-by-key."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Dispatcher:
    def __init__(self, app):
        # `app` is the orchestrator (server.main.App); we touch its attributes directly.
        self.app = app

    async def __call__(self, msg: dict):
        t = msg.get("type")
        h = self._handlers.get(t)
        if h is None:
            return [{"type": "error", "reason": f"unknown type {t!r}"}], []
        try:
            return await h(self, msg)
        except ValueError as e:
            return [{"type": "error", "reason": str(e)}], []
        except Exception as e:
            log.exception("dispatcher handler %s failed", t)
            return [{"type": "error", "reason": f"internal: {e}"}], []

    # ---------- handlers ----------

    async def _set_fft(self, msg):
        enabled = bool(msg.get("enabled", False))
        if enabled:
            # Reset post-processor state so we don't reuse stale smoother / peak
            # values from before the FFT was disabled.
            if self.app.fft_postprocessor is not None:
                self.app.fft_postprocessor.reset()
            self.app.fft_enabled.set()
        else:
            self.app.fft_enabled.clear()
        self.app.cfg.fft.enabled = enabled
        # The ENABLE toggle is the single user-facing FFT switch — keep OSC
        # transmission in lockstep so consumers downstream get the FFT stream
        # whenever the UI says FFT is on.
        self.app.cfg.osc.send_fft = enabled
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_band(self, msg):
        band = msg.get("band")
        if band not in ("low", "mid", "high"):
            raise ValueError("band must be 'low', 'mid', or 'high'")
        commit = bool(msg.get("commit", True))
        sr = self.app.current_sr()
        lo, hi = V.validate_band(band, msg.get("lo_hz"), msg.get("hi_hz"), sr)
        # Mutate cfg, then schedule a debounced retune that reads cfg at fire time.
        getattr(self.app.cfg.dsp, band).lo_hz = lo
        getattr(self.app.cfg.dsp, band).hi_hz = hi
        self.app.schedule_filter_retune()
        # The FFT post-processor's per-bin smoothing tau is anchored on band
        # geometric centers — keep it in sync.
        if self.app.fft_postprocessor is not None:
            self.app.fft_postprocessor.update_bands(self.app.snapshot_meta()["bands"])
        # AutoScaler's per-band linear pre-tilt + bandwidth-aware noise
        # subtraction both depend on the band edges — keep in sync.
        self.app.auto_scaler.set_bands(self.app._bands_tuple_dict())
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_smoothing(self, msg):
        commit = bool(msg.get("commit", True))
        tau = V.validate_tau(msg.get("tau") or {})
        tau_atk = V.validate_tau(msg.get("tau_attack") or {})
        if tau:
            self.app.cfg.dsp.tau = {**self.app.cfg.dsp.tau, **tau}
        if tau_atk:
            self.app.cfg.dsp.tau_attack = {**self.app.cfg.dsp.tau_attack, **tau_atk}
        self.app.smoother.set_tau(self.app.cfg.dsp.tau, self.app.cfg.dsp.tau_attack)
        if self.app.fft_postprocessor is not None:
            self.app.fft_postprocessor.update_smoothing(
                self.app.cfg.dsp.tau, self.app.cfg.dsp.tau_attack,
            )
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_autoscale(self, msg):
        commit = bool(msg.get("commit", True))
        ok = V.validate_autoscale(
            tau_attack_s=msg.get("tau_attack_s"),
            tau_release_s=msg.get("tau_release_s"),
            noise_floor=msg.get("noise_floor"),
            strength=msg.get("strength"),
            master_gain=msg.get("master_gain"),
        )
        if "tau_attack_s" in ok or "tau_release_s" in ok:
            atk = ok.get("tau_attack_s", self.app.cfg.autoscale.tau_attack_s)
            rel = ok.get("tau_release_s", self.app.cfg.autoscale.tau_release_s)
            self.app.auto_scaler.set_taus(atk, rel)
            if "tau_attack_s" in ok:
                self.app.cfg.autoscale.tau_attack_s = ok["tau_attack_s"]
            if "tau_release_s" in ok:
                self.app.cfg.autoscale.tau_release_s = ok["tau_release_s"]
        if "noise_floor" in ok:
            self.app.auto_scaler.set_noise_floor(ok["noise_floor"])
            self.app.cfg.autoscale.noise_floor = ok["noise_floor"]
        if "strength" in ok:
            self.app.auto_scaler.set_strength(ok["strength"])
            self.app.cfg.autoscale.strength = ok["strength"]
        if "master_gain" in ok:
            self.app.cfg.autoscale.master_gain = ok["master_gain"]
        if self.app.fft_postprocessor is not None:
            self.app.fft_postprocessor.update_autoscale(
                tau_attack_s=ok.get("tau_attack_s"),
                tau_release_s=ok.get("tau_release_s"),
                noise_floor=ok.get("noise_floor"),
                strength=ok.get("strength"),
            )
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _list_devices(self, msg):
        probe = bool(msg.get("probe", False))
        items = await asyncio.to_thread(self.app.list_devices_with_probe, probe)
        return [{"type": "devices", "items": items}], []

    async def _set_device(self, msg):
        idx = V.validate_device_index(msg.get("index"))
        await self.app.hot_switch_device(idx)
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()},
                    {"type": "devices", "items": self.app.list_devices_with_probe(False)}]

    async def _set_n_fft_bins(self, msg):
        n = V.validate_n_fft_bins(msg.get("n"))
        self.app.fft_worker.reconfigure(n_bins=n)
        self.app.cfg.fft.n_bins = n
        # n_bins controls the FFT-viz log-bin density; the L/M/H AutoScaler
        # caps its noise budget at the per-band log-bin count, so keep it in
        # sync.
        self.app.auto_scaler.set_fft_geometry(n_bins=n)
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_ws_snapshot_hz(self, msg):
        commit = bool(msg.get("commit", True))
        hz = V.validate_ws_snapshot_hz(msg.get("hz"))
        self.app.ws.set_snapshot_hz(hz)
        self.app.cfg.ws.snapshot_hz = hz
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    # ---------- presets ----------

    async def _list_presets(self, msg):
        items = self.app.list_presets()
        return [{"type": "presets", "items": items}], []

    async def _save_preset(self, msg):
        name = V.validate_preset_name(msg.get("name", ""))
        body = self.app.preset_body(name)
        path = self.app.preset_path(name)
        await asyncio.to_thread(write_yaml_atomic, path, body)
        items = self.app.list_presets()
        return [{"type": "presets", "items": items}], []

    async def _load_preset(self, msg):
        name = V.validate_preset_name(msg.get("name", ""))
        path = self.app.preset_path(name)
        if not path.exists():
            raise ValueError(f"preset {name!r} not found")
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            raise ValueError(f"preset {name!r} parse failed: {e}")
        # Deep-merge over main.yaml so any missing keys fall back to the active config.
        try:
            main_data = yaml.safe_load(self.app.config_path.read_text()) or {}
            if isinstance(main_data, dict) and isinstance(data, dict):
                data = _deep_merge(main_data, data)
        except Exception as e:
            log.warning("preset %r: main.yaml fallback unavailable (%s)", name, e)

        sr = self.app.current_sr()
        applied: list[str] = []
        # dsp
        d = data.get("dsp", {}) or {}
        bands_raw = {k: d.get(k) for k in ("low", "mid", "high") if isinstance(d.get(k), dict)}
        if len(bands_raw) == 3:
            try:
                ok = V.validate_bands(bands_raw, sr)
                for name, (lo, hi) in ok.items():
                    cfg_band = getattr(self.app.cfg.dsp, name)
                    cfg_band.lo_hz, cfg_band.hi_hz = lo, hi
                self.app.schedule_filter_retune()
                if self.app.fft_postprocessor is not None:
                    self.app.fft_postprocessor.update_bands(self.app.snapshot_meta()["bands"])
                self.app.auto_scaler.set_bands(self.app._bands_tuple_dict())
                applied.append("dsp.bands")
            except ValueError as e:
                log.warning("preset dsp bands invalid: %s", e)
        if "tau" in d or "tau_attack" in d:
            try:
                if "tau" in d:
                    tau = V.validate_tau(d["tau"])
                    self.app.cfg.dsp.tau = {**self.app.cfg.dsp.tau, **tau}
                if "tau_attack" in d:
                    tau_atk = V.validate_tau(d["tau_attack"])
                    self.app.cfg.dsp.tau_attack = {**self.app.cfg.dsp.tau_attack, **tau_atk}
                self.app.smoother.set_tau(self.app.cfg.dsp.tau, self.app.cfg.dsp.tau_attack)
                if self.app.fft_postprocessor is not None:
                    self.app.fft_postprocessor.update_smoothing(
                        self.app.cfg.dsp.tau, self.app.cfg.dsp.tau_attack,
                    )
                applied.append("dsp.tau")
            except ValueError as e:
                log.warning("preset dsp.tau / dsp.tau_attack invalid: %s", e)
        # autoscale
        a = data.get("autoscale", {}) or {}
        try:
            ok = V.validate_autoscale(
                tau_attack_s=a.get("tau_attack_s"),
                tau_release_s=a.get("tau_release_s"),
                noise_floor=a.get("noise_floor"),
                strength=a.get("strength"),
                master_gain=a.get("master_gain"),
            )
            if "tau_attack_s" in ok or "tau_release_s" in ok:
                atk = ok.get("tau_attack_s", self.app.cfg.autoscale.tau_attack_s)
                rel = ok.get("tau_release_s", self.app.cfg.autoscale.tau_release_s)
                self.app.auto_scaler.set_taus(atk, rel)
                if "tau_attack_s" in ok:
                    self.app.cfg.autoscale.tau_attack_s = ok["tau_attack_s"]
                if "tau_release_s" in ok:
                    self.app.cfg.autoscale.tau_release_s = ok["tau_release_s"]
            if "noise_floor" in ok:
                self.app.auto_scaler.set_noise_floor(ok["noise_floor"])
                self.app.cfg.autoscale.noise_floor = ok["noise_floor"]
            if "strength" in ok:
                self.app.auto_scaler.set_strength(ok["strength"])
                self.app.cfg.autoscale.strength = ok["strength"]
            if "master_gain" in ok:
                self.app.cfg.autoscale.master_gain = ok["master_gain"]
            if ok and self.app.fft_postprocessor is not None:
                self.app.fft_postprocessor.update_autoscale(
                    tau_attack_s=ok.get("tau_attack_s"),
                    tau_release_s=ok.get("tau_release_s"),
                    noise_floor=ok.get("noise_floor"),
                    strength=ok.get("strength"),
                )
            if ok:
                applied.append("autoscale")
        except ValueError as e:
            log.warning("preset autoscale invalid: %s", e)
        # fft view
        f = data.get("fft", {}) or {}
        if "n_bins" in f:
            try:
                n = V.validate_n_fft_bins(f["n_bins"])
                self.app.fft_worker.reconfigure(n_bins=n)
                self.app.cfg.fft.n_bins = n
                self.app.auto_scaler.set_fft_geometry(n_bins=n)
                applied.append("fft.n_bins")
            except ValueError as e:
                log.warning("preset fft.n_bins invalid: %s", e)
        # window/hop/f_min if specified and reasonable
        ws = f.get("window_size") if isinstance(f.get("window_size"), int) else None
        hop = f.get("hop") if isinstance(f.get("hop"), int) else None
        fmin = f.get("f_min") if isinstance(f.get("f_min"), (int, float)) else None
        if ws or hop or fmin:
            try:
                self.app.fft_worker.reconfigure(window_size=ws, hop=hop, f_min=fmin)
                if ws: self.app.cfg.fft.window_size = ws
                if hop: self.app.cfg.fft.hop = hop
                if fmin is not None: self.app.cfg.fft.f_min = float(fmin)
                # Window resize changes Δf_lin = sr/n_fft, which scales the
                # K_lin term in the L/M/H AutoScaler's noise budget. f_min
                # changes log_bins_per_octave, which scales the N_log cap.
                if ws:
                    self.app.auto_scaler.set_n_fft_window(ws)
                if fmin is not None:
                    self.app.auto_scaler.set_fft_geometry(f_min=float(fmin))
                applied.append("fft.window")
            except Exception as e:
                log.warning("preset fft window/hop invalid: %s", e)
        if "peak_smear_oct" in f:
            try:
                v = V.validate_peak_smear_oct(f["peak_smear_oct"])
                self.app.cfg.fft.peak_smear_oct = v
                if self.app.fft_postprocessor is not None:
                    self.app.fft_postprocessor.update_smear(v)
                applied.append("fft.peak_smear_oct")
            except ValueError as e:
                log.warning("preset fft.peak_smear_oct invalid: %s", e)
        if "tilt_db_per_oct" in f:
            try:
                v = V.validate_fft_tilt_db_per_oct(f["tilt_db_per_oct"])
                self.app.cfg.fft.tilt_db_per_oct = v
                if self.app.fft_postprocessor is not None:
                    self.app.fft_postprocessor.update_tilt(v)
                self.app.auto_scaler.set_tilt(v)
                applied.append("fft.tilt_db_per_oct")
            except ValueError as e:
                log.warning("preset fft.tilt_db_per_oct invalid: %s", e)

        if not applied:
            raise ValueError(f"preset {name!r} produced no valid fields")
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_send_raw_db(self, msg):
        send_raw = bool(msg.get("send_raw_db", False))
        self.app.cfg.fft.send_raw_db = send_raw
        self.app.persister.request(commit=True)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_peak_smear(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_peak_smear_oct(msg.get("peak_smear_oct"))
        self.app.cfg.fft.peak_smear_oct = v
        if self.app.fft_postprocessor is not None:
            self.app.fft_postprocessor.update_smear(v)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_ui_layout(self, msg):
        commit = bool(msg.get("commit", True))
        layout = V.validate_ui_layout(msg.get("layout") or {})
        self.app.cfg.ui.layout = layout
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_peak_decay(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_peak_decay_per_s(msg.get("peak_decay_per_s"))
        self.app.cfg.ui.peak_decay_per_s = v
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    async def _set_fft_tilt(self, msg):
        commit = bool(msg.get("commit", True))
        v = V.validate_fft_tilt_db_per_oct(msg.get("tilt_db_per_oct"))
        self.app.cfg.fft.tilt_db_per_oct = v
        if self.app.fft_postprocessor is not None:
            self.app.fft_postprocessor.update_tilt(v)
        # Mirror to L/M/H AutoScaler so its per-band linear pre-tilt updates too.
        self.app.auto_scaler.set_tilt(v)
        self.app.persister.request(commit=commit)
        return [], [{"type": "meta", **self.app.snapshot_meta()}]

    _handlers = {
        "set_fft": _set_fft,
        "set_band": _set_band,
        "set_smoothing": _set_smoothing,
        "set_autoscale": _set_autoscale,
        "list_devices": _list_devices,
        "set_device": _set_device,
        "set_n_fft_bins": _set_n_fft_bins,
        "set_ws_snapshot_hz": _set_ws_snapshot_hz,
        "set_fft_send_raw_db": _set_fft_send_raw_db,
        "set_fft_peak_smear": _set_fft_peak_smear,
        "set_fft_tilt": _set_fft_tilt,
        "set_peak_decay": _set_peak_decay,
        "set_ui_layout": _set_ui_layout,
        "list_presets": _list_presets,
        "save_preset": _save_preset,
        "load_preset": _load_preset,
    }
