// Three vertical bars with peak-hold ticks decaying at ~1.5/sec.

import { store, recordVizPerf } from "../store.js";

const COLORS = ["#5a8dee", "#79d17a", "#e8a857"];
const LABELS = ["low", "mid", "high"];

export function makeBars(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const peaks = [0, 0, 0];
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
    const vals = [store.low, store.mid, store.high];
    const dpr = window.devicePixelRatio || 1;
    const padX = 16 * dpr, padY = 14 * dpr;
    const slot = (w - 2 * padX) / 3;
    const barW = slot * 0.6;
    for (let i = 0; i < 3; i++) {
      const v = Math.max(0, Math.min(1, vals[i]));
      // peak hold
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - 1.5 * dt);

      const x = padX + slot * i + (slot - barW) / 2;
      const baseY = h - padY;
      const barH = (h - 2 * padY) * v;
      ctx.fillStyle = "#1f2227";
      ctx.fillRect(x, padY, barW, h - 2 * padY);
      ctx.fillStyle = COLORS[i];
      ctx.fillRect(x, baseY - barH, barW, barH);
      // peak tick
      const peakY = baseY - (h - 2 * padY) * peaks[i];
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, peakY - 1, barW, 2);
      // label
      ctx.fillStyle = "#8b939c";
      ctx.font = `${10 * dpr}px ui-sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(LABELS[i], x + barW / 2, h - 2);
    }
    recordVizPerf("bars", performance.now() - t0);
  }
  return { draw };
}
