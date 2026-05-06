// Sliders, dropdowns, presets. All numeric controls are drag-aware
// (commit:false during drag, commit:true on release).

import { store } from "./store.js";
import { send } from "./ws.js";
import { makeFreqAxis } from "./freq_axis.js";

const PRESET_NAME_RE = /^[a-zA-Z0-9_\- ]+$/;

// log-mapped slider helpers
function logSliderToValue(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const t = (parseFloat(el.value) - min) / (max - min);
  return Math.exp(Math.log(min) + t * (Math.log(max) - Math.log(min)));
}
function valueToLogSlider(el, v) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const t = (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min));
  el.value = String(min + t * (max - min));
}
function readSlider(el) {
  return el.dataset.log ? logSliderToValue(el) : parseFloat(el.value);
}
function writeSlider(el, v) {
  if (el.dataset.log) valueToLogSlider(el, v);
  else el.value = String(v);
}

// Snap a frequency to a "nice" multiple that scales with f. Step doubles each
// octave: ~8 below 320 Hz, 16 by 640 Hz, 32 by 1.3k, 64 by 2.5k, 128 above 5k.
export function snapHz(f) {
  const step = Math.min(128, Math.max(8, Math.pow(2, Math.round(Math.log2(f / 40)))));
  return Math.max(step, Math.round(f / step) * step);
}
// Snap seconds to nearest multiple of 10 (min 10s).
export function snapSec(s) {
  return Math.max(10, Math.round(s / 10) * 10);
}

function bindDragAware(el, label, fmt, getMsg, snapFn) {
  let dragging = false;
  const read = () => {
    const v = readSlider(el);
    return snapFn ? snapFn(v) : v;
  };
  const update = (commit) => {
    const v = read();
    label.textContent = fmt(v);
    const msg = getMsg(v);
    if (msg) send({ ...msg, commit });
  };
  el.addEventListener("input", () => { dragging = true; update(false); });
  const commit = () => {
    if (dragging) { update(true); dragging = false; }
    else update(true); // keyboard step
  };
  el.addEventListener("change", commit);
  el.addEventListener("pointerup", commit);
  return {
    read,
    setValue: (v) => { writeSlider(el, v); label.textContent = fmt(v); },
  };
}

export function setupControls() {
  // Visual frequency-axis picker (drag colored regions / their edges).
  const freqAxis = makeFreqAxis(
    document.getElementById("freq-axis"),
    () => store.meta.sr || 48000,
  );

  // Bandpass edges are now exclusively controlled via the freq_axis widget.

  // Tau (UI in ms; protocol in seconds)
  const tauEls = {
    low:  document.getElementById("tau-low"),
    mid:  document.getElementById("tau-mid"),
    high: document.getElementById("tau-high"),
  };
  const tauLabs = {
    low:  document.getElementById("tau-low-val"),
    mid:  document.getElementById("tau-mid-val"),
    high: document.getElementById("tau-high-val"),
  };
  const fmtMs = (v) => `${Math.round(v)} ms`;
  const tauCtls = {};
  for (const k of ["low", "mid", "high"]) {
    tauCtls[k] = bindDragAware(tauEls[k], tauLabs[k], fmtMs, () => {
      const tau = {
        low:  readSlider(tauEls.low)  / 1000,
        mid:  readSlider(tauEls.mid)  / 1000,
        high: readSlider(tauEls.high) / 1000,
      };
      return { type: "set_smoothing", tau };
    });
  }

  // Peak follower (asymmetric attack/release)
  const tauAtkEl  = document.getElementById("autoscale-attack");
  const tauAtkLab = document.getElementById("autoscale-attack-val");
  const tauAtkCtl = bindDragAware(tauAtkEl, tauAtkLab, (v) => `${Math.round(v)} ms`,
    (v) => ({ type: "set_autoscale", tau_attack_s: v / 1000 }));

  const tauRelEl  = document.getElementById("autoscale-tau");
  const tauRelLab = document.getElementById("autoscale-tau-val");
  const tauRelCtl = bindDragAware(tauRelEl, tauRelLab, (v) => `${Math.round(v)} s`,
    (v) => ({ type: "set_autoscale", tau_release_s: v }), snapSec);

  // FFT peak-smear slider — slider value is hundredths of an octave (0..200 → 0..2.0 oct).
  const smearEl  = document.getElementById("fft-peak-smear");
  const smearLab = document.getElementById("fft-peak-smear-val");
  let smearDragging = false;
  const updateSmear = (commit) => {
    const v = parseFloat(smearEl.value) / 100;
    smearLab.textContent = v <= 0 ? "off" : `${v.toFixed(2)} oct`;
    send({ type: "set_fft_peak_smear", peak_smear_oct: v, commit });
  };
  smearEl.addEventListener("input",   () => { smearDragging = true; updateSmear(false); });
  smearEl.addEventListener("change",  () => { updateSmear(true); smearDragging = false; });
  smearEl.addEventListener("pointerup", () => { if (smearDragging) { updateSmear(true); smearDragging = false; } });
  const smearCtl = { setValue: (v) => { smearEl.value = String(Math.round(v * 100)); smearLab.textContent = v <= 0 ? "off" : `${v.toFixed(2)} oct`; } };

  // FFT spectral-tilt slider — slider value is tenths of a dB/oct (-60..120 → -6..12 dB/oct).
  const tiltEl  = document.getElementById("fft-tilt");
  const tiltLab = document.getElementById("fft-tilt-val");
  let tiltDragging = false;
  const updateTilt = (commit) => {
    const v = parseFloat(tiltEl.value) / 10;
    tiltLab.textContent = v === 0 ? "flat" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB/oct`;
    send({ type: "set_fft_tilt", tilt_db_per_oct: v, commit });
  };
  tiltEl.addEventListener("input",   () => { tiltDragging = true; updateTilt(false); });
  tiltEl.addEventListener("change",  () => { updateTilt(true); tiltDragging = false; });
  tiltEl.addEventListener("pointerup", () => { if (tiltDragging) { updateTilt(true); tiltDragging = false; } });
  const tiltCtl = { setValue: (v) => {
    tiltEl.value = String(Math.round(v * 10));
    tiltLab.textContent = v === 0 ? "flat" : `${v >= 0 ? "+" : ""}${v.toFixed(1)} dB/oct`;
  } };

  const floorEl  = document.getElementById("autoscale-floor");
  const floorLab = document.getElementById("autoscale-floor-val");
  // Linear slider 0..1000; map to 0..0.1 linear floor with curve. Use x^3 for fine low end.
  const sliderToFloor = (s) => Math.pow(s / 1000, 3) * 0.1;
  const floorToSlider = (f) => 1000 * Math.pow(Math.max(0, f) / 0.1, 1 / 3);
  let floorDragging = false;
  const updateFloor = (commit) => {
    const v = sliderToFloor(parseFloat(floorEl.value));
    floorLab.textContent = v <= 0 ? "off" : `${(20 * Math.log10(v)).toFixed(0)} dBFS`;
    send({ type: "set_autoscale", noise_floor: v, commit });
  };
  floorEl.addEventListener("input",   () => { floorDragging = true; updateFloor(false); });
  floorEl.addEventListener("change",  () => { updateFloor(true); floorDragging = false; });
  floorEl.addEventListener("pointerup", () => { if (floorDragging) { updateFloor(true); floorDragging = false; } });
  const floorCtl = { setValue: (f) => { floorEl.value = String(floorToSlider(f)); floorLab.textContent = f <= 0 ? "off" : `${(20 * Math.log10(f)).toFixed(0)} dBFS`; } };

  const strengthEl  = document.getElementById("autoscale-strength");
  const strengthLab = document.getElementById("autoscale-strength-val");
  let strDragging = false;
  const updateStrength = (commit) => {
    const v = parseFloat(strengthEl.value) / 100;
    strengthLab.textContent = `${Math.round(v * 100)}%`;
    send({ type: "set_autoscale", strength: v, commit });
  };
  strengthEl.addEventListener("input", () => { strDragging = true; updateStrength(false); });
  strengthEl.addEventListener("change", () => { updateStrength(true); strDragging = false; });
  strengthEl.addEventListener("pointerup", () => { if (strDragging) { updateStrength(true); strDragging = false; } });
  const strengthCtl = { setValue: (v) => { strengthEl.value = String(Math.round(v * 100)); strengthLab.textContent = `${Math.round(v * 100)}%`; } };

  // UI refresh rate — discrete slider snapped to industry-standard frame rates.
  // The same value drives the server-side WS snapshot rate AND the browser's
  // RAF render throttle (set in store.target_ui_fps; main.js reads it).
  const UI_FPS_STEPS = [15, 24, 30, 60, 90, 120];
  const wsEl  = document.getElementById("ws-hz");
  const wsLab = document.getElementById("ws-hz-val");
  const nearestFpsIdx = (hz) => {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < UI_FPS_STEPS.length; i++) {
      const d = Math.abs(UI_FPS_STEPS[i] - hz);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  };
  let wsDragging = false;
  const updateWs = (commit) => {
    const idx = parseInt(wsEl.value, 10);
    const hz = UI_FPS_STEPS[Math.max(0, Math.min(UI_FPS_STEPS.length - 1, idx))];
    wsLab.textContent = `${hz} fps`;
    store.target_ui_fps = hz;
    send({ type: "set_ws_snapshot_hz", hz, commit });
  };
  wsEl.addEventListener("input",   () => { wsDragging = true; updateWs(false); });
  wsEl.addEventListener("change",  () => { updateWs(true); wsDragging = false; });
  wsEl.addEventListener("pointerup", () => { if (wsDragging) { updateWs(true); wsDragging = false; } });
  const wsCtl = {
    setValue: (hz) => {
      const idx = nearestFpsIdx(hz);
      wsEl.value = String(idx);
      const snapped = UI_FPS_STEPS[idx];
      wsLab.textContent = `${snapped} fps`;
      store.target_ui_fps = snapped;
    },
  };

  // Visual peak-hold decay rate (UI-only; affects bars + FFT viz). Slider value
  // is hundredths of a unit/sec (1..500 → 0.01..5.0 /s).
  const peakDecayEl  = document.getElementById("peak-decay");
  const peakDecayLab = document.getElementById("peak-decay-val");
  let peakDecayDragging = false;
  const updatePeakDecay = (commit) => {
    const v = parseFloat(peakDecayEl.value) / 100;
    peakDecayLab.textContent = `${v.toFixed(2)}/s`;
    store.peak_decay_per_s = v;
    send({ type: "set_peak_decay", peak_decay_per_s: v, commit });
  };
  peakDecayEl.addEventListener("input",   () => { peakDecayDragging = true; updatePeakDecay(false); });
  peakDecayEl.addEventListener("change",  () => { updatePeakDecay(true); peakDecayDragging = false; });
  peakDecayEl.addEventListener("pointerup", () => { if (peakDecayDragging) { updatePeakDecay(true); peakDecayDragging = false; } });
  const peakDecayCtl = { setValue: (v) => {
    peakDecayEl.value = String(Math.round(v * 100));
    peakDecayLab.textContent = `${v.toFixed(2)}/s`;
    store.peak_decay_per_s = v;
  } };

  // FFT toggle
  const fftToggle = document.getElementById("fft-toggle");
  fftToggle.addEventListener("change", () => {
    send({ type: "set_fft", enabled: fftToggle.checked });
  });

  // FFT raw-dB toggle. The server is authoritative — toggling sends a
  // set_fft_send_raw_db message; the server reflects the new value back in
  // meta. Both the viz AND the OSC payload switch in lockstep so what's
  // shown is byte-identical to what's sent.
  const fftRawDb = document.getElementById("fft-raw-db");
  fftRawDb.addEventListener("change", () => {
    send({ type: "set_fft_send_raw_db", send_raw_db: fftRawDb.checked });
  });
  const fftRawDbCtl = { setValue: (v) => { fftRawDb.checked = !!v; } };

  // Devices
  const deviceSelect = document.getElementById("device-select");
  document.getElementById("device-refresh").addEventListener("click", () => send({ type: "list_devices", probe: false }));
  document.getElementById("device-probe").addEventListener("click", () => send({ type: "list_devices", probe: true }));
  deviceSelect.addEventListener("change", () => {
    const idx = parseInt(deviceSelect.value, 10);
    if (Number.isFinite(idx)) send({ type: "set_device", index: idx });
  });

  // Presets
  const presetName  = document.getElementById("preset-name");
  const presetSave  = document.getElementById("preset-save");
  const presetList  = document.getElementById("preset-list");
  const presetLoad  = document.getElementById("preset-load");
  const validatePresetName = () => {
    const v = presetName.value.trim();
    const ok = v.length >= 1 && v.length <= 64 && PRESET_NAME_RE.test(v);
    presetSave.disabled = !ok;
    presetName.style.borderColor = (v.length > 0 && !ok) ? "#e64a4a" : "";
    return ok ? v : null;
  };
  presetName.addEventListener("input", validatePresetName);
  validatePresetName();
  presetSave.addEventListener("click", () => {
    const v = validatePresetName();
    if (!v) return;
    send({ type: "save_preset", name: v });
    presetName.value = "";
    validatePresetName();
  });
  presetLoad.addEventListener("click", () => {
    const name = presetList.value;
    if (!name) return;
    send({ type: "load_preset", name });
  });

  return {
    syncMeta() {
      const m = store.meta;
      const bands = m.bands || {};
      freqAxis.syncBands(bands);
      tauCtls.low.setValue((m.tau.low || 0) * 1000);
      tauCtls.mid.setValue((m.tau.mid || 0) * 1000);
      tauCtls.high.setValue((m.tau.high || 0) * 1000);
      tauAtkCtl.setValue((m.autoscale.tau_attack_s ?? 0.05) * 1000);
      tauRelCtl.setValue(m.autoscale.tau_release_s || 60);
      smearCtl.setValue(m.fft_peak_smear_oct ?? 0.3);
      tiltCtl.setValue(m.fft_tilt_db_per_oct ?? 3.0);
      floorCtl.setValue(m.autoscale.noise_floor || 0);
      strengthCtl.setValue(m.autoscale.strength ?? 1.0);
      wsCtl.setValue(m.ws_snapshot_hz || 60);
      peakDecayCtl.setValue(m.ui_peak_decay_per_s ?? 0.6);
      fftToggle.checked = !!m.fft_enabled;
      if (m.fft_send_raw_db !== undefined) fftRawDbCtl.setValue(!!m.fft_send_raw_db);
    },
    syncDevices() {
      const cur = store.meta?.device?.index;
      deviceSelect.innerHTML = "";
      const sorted = [...store.devices].sort((a, b) => a.index - b.index);
      for (const d of sorted) {
        const opt = document.createElement("option");
        opt.value = String(d.index);
        const probed = d.probed_signal ? " ★" : "";
        opt.textContent = `[${d.index}] ${d.name} (${d.hostapi})${probed}`;
        if (d.index === cur) opt.selected = true;
        deviceSelect.appendChild(opt);
      }
    },
    syncPresets() {
      const cur = presetList.value;
      presetList.innerHTML = "";
      // Already sorted by saved_at desc on the server.
      for (const p of store.presets) {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = `${p.name}  (${p.saved_at.replace("T", " ").replace("Z", "")})`;
        presetList.appendChild(opt);
      }
      if (cur) presetList.value = cur;
    },
    snapBackOnError(_reason) {
      // Re-sync sliders from the last known meta — simple & always correct.
      this.syncMeta();
    }
  };
}
