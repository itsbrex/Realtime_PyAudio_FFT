// FFT bars — thin renderer.
//
// All post-processing happens server-side (server/dsp/fft_postprocess.py).
// The wire payload is whatever the server is sending RIGHT NOW, which is
// also what's going out over OSC on /audio/fft:
//
//   meta.fft_send_raw_db === false  (default):
//     post-processed [0..1] values — server has already done sentinel
//     interpolation, smoothing, peak normalization, gate, tanh and the
//     strength blend. Render directly as bar height; y-axis is 0..1.
//
//   meta.fft_send_raw_db === true:
//     raw wire dB. Sentinel bins (-1000) render as gaps. Y-axis is in dB
//     between [meta.fft_db_floor, meta.fft_db_ceiling].
//
// X-axis is log frequency in either mode. Tick labels at decade-ish anchors.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER } from "../colors.js";

const LUT = (() => {
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

const SENTINEL_THRESHOLD = -500;
const MIN_BAR_PX = 1;

// X-axis tick candidates (Hz). We'll show only those that fall inside the
// active [f_min, sr/2] range AND have enough horizontal room.
const X_TICK_CANDIDATES = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

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
    return dpr;
  }

  function draw() {
    const t0 = performance.now();
    const dpr = fitCanvas();
    const now = performance.now();
    const dt = Math.max(1e-3, Math.min(0.1, (now - lastT) / 1000));
    lastT = now;
    const peakDecay = store.peak_decay_per_s ?? 0.6;
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#0a0b0d";
    ctx.fillRect(0, 0, W, H);

    // Layout: leave room for axis labels (in CSS px scaled by dpr).
    const padL = Math.round(38 * dpr);
    const padR = Math.round(6  * dpr);
    const padT = Math.round(4  * dpr);
    const padB = Math.round(16 * dpr);
    const plotX = padL, plotY = padT;
    const plotW = Math.max(1, W - padL - padR);
    const plotH = Math.max(1, H - padT - padB);

    const bins = store.fft_bins;
    const n = bins ? bins.length : 0;
    if (!bins || n === 0) {
      ctx.fillStyle = "#444";
      ctx.font = `${Math.round(12 * dpr)}px ui-sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("FFT disabled", W / 2, H / 2);
      recordVizPerf("fft", performance.now() - t0);
      return;
    }
    if (!peaks || peaks.length !== n) peaks = new Float32Array(n);

    const meta = store.meta || {};
    const rawDb = !!meta.fft_send_raw_db;
    const fMin = meta.fft_f_min ?? meta.fft?.f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);

    // ---------------- bars ----------------
    const barW = plotW / n;
    if (rawDb) {
      const floor = meta.fft_db_floor ?? -60;
      const ceiling = meta.fft_db_ceiling ?? 0;
      const span = Math.max(1, ceiling - floor);
      for (let i = 0; i < n; i++) {
        const raw = bins[i];
        if (raw < SENTINEL_THRESHOLD) {
          peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
          continue; // gap — honest "no rfft bin mapped here"
        }
        const v = Math.max(0, Math.min(1, (raw - floor) / span));
        if (v > peaks[i]) peaks[i] = v;
        else peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
        paintBar(i, v, plotX, plotY, plotH, barW, /*minPx*/ 0);
      }
    } else {
      for (let i = 0; i < n; i++) {
        const v = Math.max(0, Math.min(1, bins[i]));
        if (v > peaks[i]) peaks[i] = v;
        else peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
        paintBar(i, v, plotX, plotY, plotH, barW, MIN_BAR_PX);
      }
    }

    // ---------------- L/M/H band overlay ----------------
    const bands = meta.bands;
    if (bands && fMax > fMin) {
      const freqToX = (f) => {
        if (f <= fMin) return plotX;
        if (f >= fMax) return plotX + plotW;
        return plotX + ((Math.log10(f) - logFmin) / logSpan) * plotW;
      };
      for (const name of LMH_ORDER) {
        const b = bands[name];
        if (!b) continue;
        const x0 = freqToX(b.lo_hz);
        const x1 = freqToX(b.hi_hz);
        if (x1 <= x0) continue;
        const c = LMH[name].rgb;
        ctx.fillStyle = `rgba(${c},0.15)`;
        ctx.fillRect(x0, plotY, x1 - x0, plotH);
        ctx.fillStyle = `rgba(${c},0.7)`;
        ctx.fillRect(x0, plotY, 1, plotH);
        ctx.fillRect(x1 - 1, plotY, 1, plotH);
      }
    }

    // ---------------- axes ----------------
    drawYAxis(rawDb, meta, plotX, plotY, plotH, dpr);
    drawXAxis(fMin, fMax, logFmin, logSpan, plotX, plotY, plotW, plotH, padB, dpr);

    recordVizPerf("fft", performance.now() - t0);
  }

  function paintBar(i, v, plotX, plotY, plotH, barW, minPx) {
    const barH = Math.max(minPx, v * plotH);
    const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
    const r = LUT[cidx*3], g = LUT[cidx*3+1], b = LUT[cidx*3+2];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(plotX + i * barW, plotY + plotH - barH, Math.max(1, barW - 1), barH);
    const py = plotY + plotH - peaks[i] * plotH;
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillRect(plotX + i * barW, py - 1, Math.max(1, barW - 1), 2);
  }

  function drawYAxis(rawDb, meta, plotX, plotY, plotH, dpr) {
    ctx.save();
    ctx.font = `${Math.round(10 * dpr)}px ui-sans-serif`;
    ctx.fillStyle = "#7a8088";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    let labels;
    if (rawDb) {
      const floor = meta.fft_db_floor ?? -60;
      const ceiling = meta.fft_db_ceiling ?? 0;
      const mid = Math.round((floor + ceiling) / 2);
      labels = [
        { v: 0, text: `${ceiling}` },
        { v: 0.5, text: `${mid}` },
        { v: 1, text: `${floor}` },
      ];
    } else {
      labels = [
        { v: 0, text: "1.0" },
        { v: 0.5, text: "0.5" },
        { v: 1, text: "0.0" },
      ];
    }
    const xLabel = plotX - Math.round(4 * dpr);
    ctx.strokeStyle = "rgba(120, 120, 120, 0.15)";
    ctx.lineWidth = 1;
    for (const lbl of labels) {
      const y = plotY + lbl.v * plotH;
      ctx.fillText(lbl.text, xLabel, y);
      // soft horizontal gridline
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + (canvas.width - plotX), y);
      ctx.stroke();
    }
    // Unit caption: dB if raw, scaled if processed (axis only — no bar chrome)
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#5a6068";
    ctx.fillText(rawDb ? "dB" : "scaled", plotX + Math.round(4 * dpr), plotY + Math.round(2 * dpr));
    ctx.restore();
  }

  function drawXAxis(fMin, fMax, logFmin, logSpan, plotX, plotY, plotW, plotH, padB, dpr) {
    ctx.save();
    ctx.font = `${Math.round(10 * dpr)}px ui-sans-serif`;
    ctx.fillStyle = "#7a8088";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const yLabel = plotY + plotH + Math.round(2 * dpr);
    const minSpacingPx = Math.round(40 * dpr);
    let lastTickPx = -Infinity;
    for (const f of X_TICK_CANDIDATES) {
      if (f < fMin || f > fMax) continue;
      const x = plotX + ((Math.log10(f) - logFmin) / logSpan) * plotW;
      if (x - lastTickPx < minSpacingPx) continue;
      lastTickPx = x;
      const text = f >= 1000 ? `${f / 1000}k` : `${f}`;
      ctx.fillText(text, x, yLabel);
      // tiny tick mark
      ctx.fillRect(x, plotY + plotH, 1, Math.round(3 * dpr));
    }
    ctx.restore();
  }

  return { draw };
}
