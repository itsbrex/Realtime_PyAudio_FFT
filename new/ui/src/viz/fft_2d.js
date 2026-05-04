// FFT bars with viridis-ish LUT, optional peak hold.

import { store, recordVizPerf } from "../store.js";

const LUT = (() => {
  // Cheap viridis approximation: blue -> teal -> green -> yellow.
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = Math.round(255 * Math.min(1, Math.max(0, -0.2 + 1.6 * t)));
    const g = Math.round(255 * (0.1 + 0.85 * Math.sin(Math.PI * t)));
    const b = Math.round(255 * Math.max(0, 1 - 1.6 * t + 0.5 * Math.pow(t, 4)));
    out[i*3] = r; out[i*3+1] = g; out[i*3+2] = b;
  }
  return out;
})();

export function makeFft(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  let peaks = null;
  let lastT = performance.now();

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    const now = performance.now();
    const dt = (now - lastT) / 1000;
    lastT = now;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#0a0b0d";
    ctx.fillRect(0, 0, w, h);

    const bins = store.fft_bins;
    const n = bins ? bins.length : 0;
    if (!bins || n === 0) {
      ctx.fillStyle = "#444";
      ctx.font = "12px ui-sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("FFT disabled", w / 2, h / 2);
      recordVizPerf("fft", performance.now() - t0);
      return;
    }

    if (!peaks || peaks.length !== n) peaks = new Float32Array(n);
    const floor = store.fft_db_floor || -80;
    const ceiling = store.fft_db_ceiling || 0;
    const span = Math.max(1, ceiling - floor);

    const barW = w / n;
    for (let i = 0; i < n; i++) {
      const db = bins[i];
      const v = Math.max(0, Math.min(1, (db - floor) / span));
      // peak hold
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - 0.6 * dt);

      const barH = v * h;
      const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
      const r = LUT[cidx*3], g = LUT[cidx*3+1], b = LUT[cidx*3+2];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
      // peak tick
      const py = h - peaks[i] * h;
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillRect(i * barW, py - 1, Math.max(1, barW - 1), 1);
    }
    recordVizPerf("fft", performance.now() - t0);
  }
  return { draw };
}
