// Entry point. Wires WS handlers -> store, sets up controls, runs RAF loop.

import { connect, onMessage, onError } from "./ws.js";
import { store, avgRing, p95Ring } from "./store.js";
import { setupControls } from "./controls.js";
import { makeLines } from "./viz/lmh_lines.js";
import { makeBars }  from "./viz/lmh_bars.js";
import { makeScene } from "./viz/lmh_scene.js";
import { makeFft }   from "./viz/fft_2d.js";

const controls = setupControls();

const lines = makeLines(document.getElementById("viz-lines"));
const bars  = makeBars(document.getElementById("viz-bars"));
const scene = makeScene(document.getElementById("viz-scene"));
const fft   = makeFft(document.getElementById("viz-fft"));

// ----- WS handlers -----
onMessage("snapshot", (m) => {
  store.low = m.low; store.mid = m.mid; store.high = m.high;
  store.low_raw = m.low_raw; store.mid_raw = m.mid_raw; store.high_raw = m.high_raw;
  // raw bars in the controls panel
  const fmt = (v) => `${Math.min(100, v * 100).toFixed(0)}%`;
  document.getElementById("raw-low").style.width  = fmt(m.low_raw);
  document.getElementById("raw-mid").style.width  = fmt(m.mid_raw);
  document.getElementById("raw-high").style.width = fmt(m.high_raw);
});

onMessage("meta", (m) => {
  store.meta = { ...store.meta, ...m };
  if (m.fft_db_floor !== undefined) store.fft_db_floor = m.fft_db_floor;
  if (m.fft_db_ceiling !== undefined) store.fft_db_ceiling = m.fft_db_ceiling;
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
  // Browser side. raf_ms is the inter-frame interval (~16.67ms at 60fps),
  // not work time — so display "% over budget" instead of "% of budget",
  // which lights up only when frames are actually dropped.
  const raf_avg = avgRing(store.raf_ms_ring);
  const raf_p95 = p95Ring(store.raf_ms_ring);
  const raf_load = Math.max(0, (raf_avg - 16.667) / 16.667 * 100);
  setPerfRow("b_raf", raf_avg, raf_p95, raf_load, false);
  for (const k of ["lines", "bars", "scene", "fft"]) {
    const v = store.viz_perf[k];
    if (!v) continue;
    const a = avgRing(v.ring), p95 = p95Ring(v.ring);
    setPerfRow("b_" + k, a, p95, (a / 16.667) * 100, false);
  }
}

// ----- RAF loop -----
let lastFrameT = performance.now();
function frame(now) {
  const dt = now - lastFrameT;
  lastFrameT = now;
  // raf delta ring
  store.raf_ms_ring[store.raf_idx % store.raf_ms_ring.length] = dt;
  store.raf_idx++;
  // ui fps badge — rolling 60 frames
  if ((store.raf_idx & 15) === 0) {
    const avg = avgRing(store.raf_ms_ring);
    const fps = avg > 0 ? Math.round(1000 / avg) : 0;
    document.getElementById("ui-fps").textContent = `ui ${fps} fps`;
    // server fps from msgTimestamps
    const ts = store.msgTimestamps;
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
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
connect();
