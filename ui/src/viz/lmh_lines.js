// Three rolling polylines for low/mid/high.
//
// - Horizontal alpha gradient: line goes 1.0 (right, newest) → 0.1 (left, oldest).
// - Subtle per-band area fill at ~0.10 alpha on the right, fading to ~0.02 on the left.
// - Faint y-axis labels (0..1) on the left edge.
// - History window (2..30s, log-scale) via store.lines_history_s. Each sample is
//   plotted at x = w * (1 - age_ms / window_ms), so the time axis is absolute:
//   a half-empty buffer leaves the left portion of the canvas blank, and new
//   samples flow in from the right rather than re-stretching the existing curve.

import { store, recordVizPerf } from "../store.js";
import { LMH } from "../colors.js";

// Ring big enough to cover 30s of history at the top UI refresh rate (120 fps)
// with comfortable headroom — 4096 * 4 bytes * 4 arrays ≈ 64 KB total.
const N_MAX = 4096;

const FILL_ALPHA_RIGHT   = 0.10;
const FILL_ALPHA_LEFT    = 0.0;
const STROKE_ALPHA_RIGHT = 1.00;
const STROKE_ALPHA_LEFT  = 0.0;

export function makeLines(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const buf = {
    low:  new Float32Array(N_MAX),
    mid:  new Float32Array(N_MAX),
    high: new Float32Array(N_MAX),
  };
  const ts = new Float32Array(N_MAX); // ms since epoch0 (Float32 precision is fine for windows < ~30s)
  const epoch0 = performance.now();
  let head = 0, count = 0;

  // Cached per-resize.
  let cw = 0, ch = 0;
  let strokeGrads = null, fillGrads = null;

  // rAF pauses while the tab is hidden, so the ring's newest sample is
  // timestamped from when we last drew. On refocus, drop the stale history
  // and let the curve fill in from the right edge again.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { head = 0; count = 0; }
  });

  function hexToRgb(hex) {
    const m = hex.replace("#", "");
    if (m.length === 3) return m.split("").map((c) => parseInt(c + c, 16));
    return [parseInt(m.slice(0, 2), 16), parseInt(m.slice(2, 4), 16), parseInt(m.slice(4, 6), 16)];
  }

  function buildGradients(w) {
    const mk = (hex, a0, a1) => {
      const [r, g, b] = hexToRgb(hex);
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, `rgba(${r},${g},${b},${a0})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},${a1})`);
      return grad;
    };
    strokeGrads = {
      low:  mk(LMH.low.hex,  STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
      mid:  mk(LMH.mid.hex,  STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
      high: mk(LMH.high.hex, STROKE_ALPHA_LEFT, STROKE_ALPHA_RIGHT),
    };
    fillGrads = {
      low:  mk(LMH.low.hex,  FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
      mid:  mk(LMH.mid.hex,  FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
      high: mk(LMH.high.hex, FILL_ALPHA_LEFT, FILL_ALPHA_RIGHT),
    };
  }

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
    if (w !== cw || h !== ch) {
      cw = w; ch = h;
      buildGradients(w);
    }
  }

  function update() {
    buf.low[head]  = store.low;
    buf.mid[head]  = store.mid;
    buf.high[head] = store.high;
    ts[head] = performance.now() - epoch0;
    head = (head + 1) % N_MAX;
    if (count < N_MAX) count++;
  }

  // Number of most-recent samples that fall within [now - histS, now].
  function effectiveK(histMs, nowRel) {
    if (count === 0) return 0;
    const cutoff = nowRel - histMs;
    const latest = (head - 1 + N_MAX) % N_MAX;
    let k = 1;
    while (k < count) {
      const idx = (latest - k + N_MAX) % N_MAX;
      if (ts[idx] < cutoff) break;
      k++;
    }
    return k;
  }

  function tracePolyline(arr, K, w, h, histMs, nowRel) {
    const latest = (head - 1 + N_MAX) % N_MAX;
    for (let i = 0; i < K; i++) {
      // i=0 → oldest (leftmost in window); i=K-1 → newest (rightmost, x≈w).
      const idx = (latest - (K - 1 - i) + N_MAX) % N_MAX;
      const v = arr[idx];
      const x = w * (1 - (nowRel - ts[idx]) / histMs);
      const y = h - (v < 0 ? 0 : v > 1 ? 1 : v) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  }

  function drawSeries(arr, fillGrad, strokeGrad, K, w, h, dpr, histMs, nowRel) {
    if (K < 2) return;
    const latest = (head - 1 + N_MAX) % N_MAX;
    const oldestIdx  = (latest - (K - 1) + N_MAX) % N_MAX;
    const firstX = w * (1 - (nowRel - ts[oldestIdx]) / histMs);
    const lastX  = w * (1 - (nowRel - ts[latest])    / histMs);

    // Filled area: polyline closed back to baseline at the same x range as the
    // visible data — empty portion of the canvas stays empty.
    ctx.beginPath();
    tracePolyline(arr, K, w, h, histMs, nowRel);
    ctx.lineTo(lastX,  h);
    ctx.lineTo(firstX, h);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Stroke (no bottom edge).
    ctx.beginPath();
    tracePolyline(arr, K, w, h, histMs, nowRel);
    ctx.strokeStyle = strokeGrad;
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function drawGrid(w, h) {
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const f of [0.5]) {
      const y = Math.round(f * h) + 0.5;
      ctx.moveTo(0, y); ctx.lineTo(w, y);
    }
    ctx.stroke();
  }

  function drawAxisLabels(h, dpr) {
    ctx.fillStyle = "rgba(255,255,255,0.32)";
    ctx.font = `${10 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    const padL = 4 * dpr;
    for (const [f, label] of [[0.0, "1.0"], [0.5, "0.5"], [1.0, "0"]]) {
      const y = f * h;
      const yy = f === 0.0 ? y + 7 * dpr : f === 1.0 ? y - 7 * dpr : y;
      ctx.fillText(label, padL, yy);
    }
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    update();
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0a0b0d";
    ctx.fillRect(0, 0, w, h);
    drawGrid(w, h);

    const histS = Math.max(2, Math.min(30, store.lines_history_s ?? 5));
    const histMs = histS * 1000;
    const nowRel = performance.now() - epoch0;
    const K = effectiveK(histMs, nowRel);
    drawSeries(buf.low,  fillGrads.low,  strokeGrads.low,  K, w, h, dpr, histMs, nowRel);
    drawSeries(buf.mid,  fillGrads.mid,  strokeGrads.mid,  K, w, h, dpr, histMs, nowRel);
    drawSeries(buf.high, fillGrads.high, strokeGrads.high, K, w, h, dpr, histMs, nowRel);

    drawAxisLabels(h, dpr);
    recordVizPerf("lines", performance.now() - t0);
  }
  return { draw };
}
