// Spatially-separated L/M/H scene:
//   low  -> central glowing disc (radius + alpha grow with low)
//   mid  -> full-screen background color hue (alpha grows with mid)
//   high -> bright random noise sprinkled across the screen (count + alpha
//           grow with high), resampled fresh every frame
// Layering (bottom -> top): dark base, mid hue tint, low disc, high noise.

import { store, recordVizPerf } from "../store.js";
import { LMH } from "../colors.js";

// ─────────────────────────────────────────────────────────────────────────
// Tuning. All visual knobs live here — tweak freely.
// ─────────────────────────────────────────────────────────────────────────
const CONFIG = {
  bgColor: "#0a0b0d",

  low: {
    // Radius scales with low: lowR = baseR * (radiusBase + radiusGain*lo) * radiusScale
    // baseR = 0.5 * min(w, h). radiusScale=1.25 makes the disc 25% larger.
    radiusScale: 1.25,
    radiusBase:  0.15,
    radiusGain:  0.40,
    // Outer alpha of the disc gradient. lo=0 -> alphaMin (invisible),
    // lo=1 -> alphaMax. Inner stop scales by midStopAlphaScale.
    alphaMin: 0.1,
    alphaMax: 0.9,
    saturation:    80,   // %
    lightnessMin:  25,   // % at lo=0
    lightnessMax:  75,   // % at lo=1
    midStop:           0.65,  // gradient stop position [0..1]
    midStopAlphaScale: 0.5,
  },

  mid: {
    // Full-screen tint behind everything else. Alpha = alphaMin..alphaMax
    // mapped from md=0..1. Even at alpha=1, low/high paint over it so it
    // never visually dominates.
    alphaMin: 0.0,
    alphaMax: 0.85,
    saturation: 45,   // %
    lightness:  20,   // % — kept dim so a full-alpha tint doesn't blind
  },

  high: {
    // Random sample count scales with hi.
    minPoints: 100,
    maxPoints: 500,
    // Per-point alpha scales with hi (alphaMin at hi=0, alphaMax at hi=1).
    alphaMin: 0.0,
    alphaMax: 1.0,
    // Visual point size in CSS pixels (multiplied by devicePixelRatio).
    pointSize: 2.0,
    // 0 = uniform across screen, 1 = strong push toward edges.
    // We bias outward by rejection sampling against an acceptance prob
    // that grows with normalized distance from center.
    edgeBias: 1.5,
    // Color: bright/whitish on the high hue so they read as sparkles.
    saturation: 50,  // %
    lightness:  90,   // %
  },
};
// ─────────────────────────────────────────────────────────────────────────

export function makeScene(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });

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
    const w = canvas.width, h = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    const lo = Math.max(0, Math.min(1, store.low));
    const md = Math.max(0, Math.min(1, store.mid));
    const hi = Math.max(0, Math.min(1, store.high));

    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.5;

    // --- Layer 0: solid dark background. ---
    ctx.fillStyle = CONFIG.bgColor;
    ctx.fillRect(0, 0, w, h);

    // --- Layer 1: MID full-screen tint (base hue, behind everything). ---
    const midAlpha = CONFIG.mid.alphaMin + (CONFIG.mid.alphaMax - CONFIG.mid.alphaMin) * md;
    if (midAlpha > 0.001) {
      ctx.fillStyle = `hsla(${LMH.mid.hue}, ${CONFIG.mid.saturation}%, ${CONFIG.mid.lightness}%, ${midAlpha})`;
      ctx.fillRect(0, 0, w, h);
    }

    // --- Layer 2: LOW central disc with radial gradient. ---
    const lowR = baseR * (CONFIG.low.radiusBase + CONFIG.low.radiusGain * lo) * CONFIG.low.radiusScale;
    const lowAlpha = CONFIG.low.alphaMin + (CONFIG.low.alphaMax - CONFIG.low.alphaMin) * lo;
    if (lowAlpha > 0.001 && lowR > 0.5) {
      const light = CONFIG.low.lightnessMin + (CONFIG.low.lightnessMax - CONFIG.low.lightnessMin) * lo;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, lowR);
      grad.addColorStop(0,                  `hsla(${LMH.low.hue}, ${CONFIG.low.saturation}%, ${light}%, ${lowAlpha})`);
      grad.addColorStop(CONFIG.low.midStop, `hsla(${LMH.low.hue}, ${CONFIG.low.saturation}%, ${light}%, ${lowAlpha * CONFIG.low.midStopAlphaScale})`);
      grad.addColorStop(1,                  `hsla(${LMH.low.hue}, ${CONFIG.low.saturation}%, ${light}%, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, lowR, 0, Math.PI * 2);
      ctx.fill();
    }

    // --- Layer 3: HIGH bright random noise (resampled every frame). ---
    const hiAlpha = CONFIG.high.alphaMin + (CONFIG.high.alphaMax - CONFIG.high.alphaMin) * hi;
    if (hiAlpha > 0.001) {
      const nPoints = Math.round(CONFIG.high.minPoints + (CONFIG.high.maxPoints - CONFIG.high.minPoints) * hi);
      const sz = Math.max(1, Math.round(CONFIG.high.pointSize * dpr));
      const halfW = w * 0.5, halfH = h * 0.5;
      const bias = CONFIG.high.edgeBias;
      ctx.fillStyle = `hsla(${LMH.high.hue}, ${CONFIG.high.saturation}%, ${CONFIG.high.lightness}%, ${hiAlpha})`;
      ctx.beginPath();
      for (let i = 0; i < nPoints; i++) {
        // Rejection sample with edge-biased acceptance, capped attempts so
        // we never loop forever in pathological cases.
        let x = 0, y = 0;
        for (let attempt = 0; attempt < 4; attempt++) {
          x = Math.random() * w;
          y = Math.random() * h;
          const dx = (x - cx) / halfW;
          const dy = (y - cy) / halfH;
          const d = Math.min(1, Math.sqrt(dx * dx + dy * dy));
          const accept = (1 - bias) + bias * d;
          if (Math.random() < accept) break;
        }
        ctx.rect(x | 0, y | 0, sz, sz);
      }
      ctx.fill();
    }

    recordVizPerf("scene", performance.now() - t0);
  }
  return { draw };
}
