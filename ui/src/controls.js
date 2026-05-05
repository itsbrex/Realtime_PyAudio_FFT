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

  // Band edges — three independent bandpasses, each with lo/hi sliders.
  const fmtHz = (v) => `${Math.round(v)} Hz`;
  const bandCtls = {};
  for (const band of ["low", "mid", "high"]) {
    const loEl  = document.getElementById(`${band}-lo`);
    const hiEl  = document.getElementById(`${band}-hi`);
    const loLab = document.getElementById(`${band}-lo-val`);
    const hiLab = document.getElementById(`${band}-hi-val`);
    const sendBand = () => ({
      type: "set_band",
      band,
      lo_hz: loCtl.read(),
      hi_hz: hiCtl.read(),
    });
    const loCtl = bindDragAware(loEl, loLab, fmtHz, sendBand, snapHz);
    const hiCtl = bindDragAware(hiEl, hiLab, fmtHz, sendBand, snapHz);
    bandCtls[band] = { lo: loCtl, hi: hiCtl };
  }

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

  // Auto-scaler
  const tauRelEl  = document.getElementById("autoscale-tau");
  const tauRelLab = document.getElementById("autoscale-tau-val");
  const tauRelCtl = bindDragAware(tauRelEl, tauRelLab, (v) => `${Math.round(v)} s`,
    (v) => ({ type: "set_autoscale", tau_release_s: v }), snapSec);

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

  // WS hz
  const wsEl  = document.getElementById("ws-hz");
  const wsLab = document.getElementById("ws-hz-val");
  const wsCtl = bindDragAware(wsEl, wsLab, (v) => `${Math.round(v)} Hz`,
    (v) => ({ type: "set_ws_snapshot_hz", hz: Math.round(v) }));

  // FFT toggle
  const fftToggle = document.getElementById("fft-toggle");
  fftToggle.addEventListener("change", () => {
    send({ type: "set_fft", enabled: fftToggle.checked });
  });

  // FFT display tilt (UI-only; OSC/WS payload stays honest dB)
  const fftTilt = document.getElementById("fft-tilt");
  store.fft_tilt_enabled = fftTilt.checked;
  fftTilt.addEventListener("change", () => {
    store.fft_tilt_enabled = fftTilt.checked;
  });

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
      for (const band of ["low", "mid", "high"]) {
        const b = bands[band];
        if (!b) continue;
        bandCtls[band].lo.setValue(b.lo_hz);
        bandCtls[band].hi.setValue(b.hi_hz);
      }
      freqAxis.syncBands(bands);
      tauCtls.low.setValue((m.tau.low || 0) * 1000);
      tauCtls.mid.setValue((m.tau.mid || 0) * 1000);
      tauCtls.high.setValue((m.tau.high || 0) * 1000);
      tauRelCtl.setValue(m.autoscale.tau_release_s || 60);
      floorCtl.setValue(m.autoscale.noise_floor || 0);
      strengthCtl.setValue(m.autoscale.strength ?? 1.0);
      wsCtl.setValue(m.ws_snapshot_hz || 60);
      fftToggle.checked = !!m.fft_enabled;
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
