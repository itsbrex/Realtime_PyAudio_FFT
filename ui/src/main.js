// Entry point. Wires WS handlers -> store, sets up controls, runs RAF loop.

import { connect, onMessage, onError } from "./ws.js";
import { store, avgRing, p95Ring } from "./store.js";
import { setupControls } from "./controls.js";
import { makeLines } from "./viz/lmh_lines.js";
import { makeBars }  from "./viz/lmh_bars.js";
import { makeScene } from "./viz/lmh_scene.js";
import { makeFft }   from "./viz/fft_2d.js";
import { setupTooltips } from "./tooltips.js";

const controls = setupControls();

const lines = makeLines(document.getElementById("viz-lines"));
const bars  = makeBars(document.getElementById("viz-bars"));
const scene = makeScene(document.getElementById("viz-scene"));
const fft   = makeFft(document.getElementById("viz-fft"));

// ----- WS handlers -----
onMessage("snapshot", (m) => {
  store.low = m.low; store.mid = m.mid; store.high = m.high;
  store.low_raw = m.low_raw; store.mid_raw = m.mid_raw; store.high_raw = m.high_raw;
});

onMessage("meta", (m) => {
  store.meta = { ...store.meta, ...m };
  if (m.fft_db_floor !== undefined) store.fft_db_floor = m.fft_db_floor;
  if (m.fft_db_ceiling !== undefined) store.fft_db_ceiling = m.fft_db_ceiling;
  if (m.fft_send_raw_db !== undefined) store.fft_send_raw_db = !!m.fft_send_raw_db;
  // When FFT is disabled, drop the last frame so the viz shows "FFT disabled"
  // instead of a frozen spectrum from the moment of toggle-off.
  if (m.fft_enabled === false) store.fft_bins = null;
  controls.syncMeta();
});

onMessage("devices", (m) => {
  store.devices = m.items || [];
  controls.syncDevices();
});

onMessage("presets", (m) => {
  store.presets = m.items || [];
  controls.syncPresets();
});

onMessage("server_status", (m) => {
  store.status = m;
  document.getElementById("cb_overruns").textContent = m.cb_overruns;
  document.getElementById("dsp_drops").textContent = m.dsp_drops;
  document.getElementById("fft_drops").textContent = m.fft_drops;
  renderPerfPanel();
});

onError((reason) => {
  const log = document.getElementById("err-log");
  const ts = new Date().toLocaleTimeString();
  log.textContent = `[${ts}] ${reason}\n` + log.textContent;
  if (log.textContent.length > 4000) log.textContent = log.textContent.slice(0, 4000);
  controls.snapBackOnError(reason);
});

// ----- Perf panel rendering -----
const PERF_ROWS = [
  { key: "cb",  label: "cb"  },
  { key: "dsp", label: "dsp" },
  { key: "fft", label: "fft" },
  { key: "ws",  label: "ws"  },
];
const BROWSER_ROWS = ["raf", "lines", "bars", "scene", "fft"];
const perfContainer = document.getElementById("perf-rows");

function ensurePerfRows() {
  if (perfContainer.children.length > 0) return;
  for (const r of PERF_ROWS) addPerfRow(r.key, r.label);
  for (const k of BROWSER_ROWS) addPerfRow("b_" + k, k);
}

function addPerfRow(id, label) {
  const row = document.createElement("div");
  row.className = "perf-row";
  row.id = "perf-" + id;
  row.innerHTML = `<span>${label}</span><div class="perf-bar"><div class="perf-bar-fill"></div></div><span class="perf-num">- / -</span>`;
  perfContainer.appendChild(row);
}

function setPerfRow(id, avg_ms, p95_ms, load_pct, disabled) {
  const row = document.getElementById("perf-" + id);
  if (!row) return;
  row.classList.toggle("disabled", !!disabled);
  const fill = row.querySelector(".perf-bar-fill");
  const num  = row.querySelector(".perf-num");
  fill.style.width = `${Math.min(100, load_pct)}%`;
  fill.classList.remove("amber", "red");
  if (load_pct >= 80) fill.classList.add("red");
  else if (load_pct >= 50) fill.classList.add("amber");
  num.textContent = `${avg_ms.toFixed(2)} / ${p95_ms.toFixed(2)} ms`;
}

function renderPerfPanel() {
  ensurePerfRows();
  const p = store.status?.perf;
  if (p) {
    for (const r of PERF_ROWS) {
      const stage = p[r.key] || {};
      setPerfRow(r.key, stage.avg_ms || 0, stage.p95_ms || 0, stage.load_pct || 0,
                 r.key === "fft" && stage.enabled === false);
    }
  }
  // Browser side. raf_ms here is the inter-DRAW interval (we throttle draws
  // to the UI refresh rate). Compare against the target period rather than a
  // fixed 60 fps budget so reducing the refresh rate doesn't light the bar red.
  const targetFps = Math.max(1, store.target_ui_fps || 60);
  const targetPeriod = 1000 / targetFps;
  const raf_avg = avgRing(store.raf_ms_ring);
  const raf_p95 = p95Ring(store.raf_ms_ring);
  const raf_load = Math.max(0, (raf_avg - targetPeriod) / targetPeriod * 100);
  setPerfRow("b_raf", raf_avg, raf_p95, raf_load, false);
  for (const k of ["lines", "bars", "scene", "fft"]) {
    const v = store.viz_perf[k];
    if (!v) continue;
    const a = avgRing(v.ring), p95 = p95Ring(v.ring);
    setPerfRow("b_" + k, a, p95, (a / targetPeriod) * 100, false);
  }
}

// ----- RAF loop -----
// We always run at requestAnimationFrame cadence (driven by the monitor) but
// only redraw when at least `1000 / target_ui_fps` ms have elapsed since the
// previous draw. The badge measures the actual draw rate, so changing the UI
// refresh rate slider is reflected directly in the "ui X fps" indicator.
let lastDrawT = performance.now();

// Tooltips for the top-right badges so it's clear what each number means.
{
  const srv = document.getElementById("server-fps");
  if (srv) srv.title =
    "Server snapshot rate — how often the server is pushing L/M/H snapshot " +
    "JSON over the WebSocket. Tracks the UI refresh rate slider. Independent " +
    "of the FFT enable toggle (FFT frames are sent as separate binary " +
    "messages and are NOT counted here).";
  const ui = document.getElementById("ui-fps");
  if (ui) ui.title =
    "Browser render rate — how often the canvases are actually being " +
    "redrawn. The render loop is throttled to the UI refresh rate slider, " +
    "so this should track that value (capped by the monitor's refresh rate).";
}

setupTooltips();

function frame(now) {
  const targetFps = Math.max(1, store.target_ui_fps || 60);
  const minPeriod = 1000 / targetFps - 0.5; // small slack so 60 fps RAF hits 60
  const elapsed = now - lastDrawT;
  if (elapsed >= minPeriod) {
    // Record the inter-draw interval (not the inter-RAF interval) so the
    // "ui fps" badge reflects actual draw cadence, which the UI refresh rate
    // slider controls.
    store.raf_ms_ring[store.raf_idx % store.raf_ms_ring.length] = elapsed;
    store.raf_idx++;
    lastDrawT = now;

    if ((store.raf_idx & 15) === 0) {
      const avg = avgRing(store.raf_ms_ring);
      const fps = avg > 0 ? Math.round(1000 / avg) : 0;
      document.getElementById("ui-fps").textContent = `ui ${fps} fps`;
      // Server snapshot rate (snapshot JSON only; binary FFT frames excluded
      // so the FFT enable toggle doesn't move this number).
      const ts = store.snapshotTimestamps;
      if (ts.length >= 2) {
        const span = (ts[ts.length - 1] - ts[0]) / 1000;
        const sfps = span > 0 ? Math.round((ts.length - 1) / span) : 0;
        document.getElementById("server-fps").textContent = `srv ${sfps} Hz`;
      }
    }
    lines.draw();
    bars.draw();
    scene.draw();
    fft.draw();
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
connect();
