// Three rolling polylines for low/mid/high.

import { store, recordVizPerf } from "../store.js";

const N = 300;

export function makeLines(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const buf = {
    low:  new Float32Array(N),
    mid:  new Float32Array(N),
    high: new Float32Array(N),
  };
  let head = 0;

  function update() {
    buf.low[head]  = store.low;
    buf.mid[head]  = store.mid;
    buf.high[head] = store.high;
    head = (head + 1) % N;
  }

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }

  function drawSeries(arr, color) {
    const w = canvas.width, h = canvas.height;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const idx = (head + i) % N;
      const x = (i / (N - 1)) * w;
      const y = h - Math.max(0, Math.min(1, arr[idx])) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    update();
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0a0b0d";
    ctx.fillRect(0, 0, w, h);
    // Faint baseline grid
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.beginPath();
    for (const f of [0.25, 0.5, 0.75]) {
      ctx.moveTo(0, f * h); ctx.lineTo(w, f * h);
    }
    ctx.stroke();

    drawSeries(buf.low,  "#5a8dee");
    drawSeries(buf.mid,  "#79d17a");
    drawSeries(buf.high, "#e8a857");
    recordVizPerf("lines", performance.now() - t0);
  }
  return { draw };
}
