// Single rectangle: hue/lightness from low, overlay alpha from mid,
// noise speckle from high.

import { store, recordVizPerf } from "../store.js";

export function makeScene(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  let imgData = null;

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      imgData = null;
    }
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    const w = canvas.width, h = canvas.height;
    const lo = Math.max(0, Math.min(1, store.low));
    const md = Math.max(0, Math.min(1, store.mid));
    const hi = Math.max(0, Math.min(1, store.high));

    const hue = 200 + lo * 160;       // 200..360
    const light = 14 + lo * 38;       // 14..52
    ctx.fillStyle = `hsl(${hue}, 60%, ${light}%)`;
    ctx.fillRect(0, 0, w, h);

    // mid overlay
    ctx.fillStyle = `rgba(255, 220, 130, ${md * 0.5})`;
    ctx.fillRect(0, 0, w, h);

    // hi-frequency speckle
    if (hi > 0.05) {
      if (!imgData || imgData.width !== w || imgData.height !== h) {
        imgData = ctx.createImageData(w, h);
      }
      const data = imgData.data;
      const density = Math.floor(hi * w * h * 0.06);
      for (let i = 0; i < density; i++) {
        const x = (Math.random() * w) | 0;
        const y = (Math.random() * h) | 0;
        const off = (y * w + x) * 4;
        data[off] = 255; data[off+1] = 255; data[off+2] = 255; data[off+3] = (hi * 220) | 0;
      }
      // Composite via temp canvas alpha — simpler: just putImageData with srcOver semantics by re-fill on next frame.
      // Use globalCompositeOperation to overlay sparkles only
      ctx.globalCompositeOperation = "lighter";
      ctx.putImageData(imgData, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      // wipe imgData for next frame so old sparkles don't persist
      data.fill(0);
    }
    recordVizPerf("scene", performance.now() - t0);
  }
  return { draw };
}
