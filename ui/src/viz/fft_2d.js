// FFT bars with viridis-ish LUT, optional peak hold.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER } from "../colors.js";

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
    // Display-only spectral tilt: +4.5 dB/oct pivoted at 1 kHz. When on, we
    // also widen the y-axis (lower floor, higher ceiling) to absorb the
    // tilt's swing across the band so highs don't clip and lows don't sink.
    // OSC/WS payloads are untouched.
    const tiltOn = store.fft_tilt_enabled;
    const metaPre = store.meta || {};
    const fMinTilt = metaPre.fft?.f_min ?? 30;
    const fMaxTilt = (metaPre.sr ?? 48000) / 2;
    const logFminTilt = Math.log10(fMinTilt);
    const logSpanTilt = Math.max(1e-6, Math.log10(fMaxTilt) - logFminTilt);
    const TILT_DB_PER_OCT = 4.5;
    const LOG10_2 = Math.log10(2);

    let floor = store.fft_db_floor || -60;
    let ceiling = store.fft_db_ceiling || 0;
    if (tiltOn) {
      const octBelow1k = Math.max(0, (3 - logFminTilt) / LOG10_2); // log2(1000/fMin)
      const octAbove1k = Math.max(0, (Math.log10(fMaxTilt) - 3) / LOG10_2); // log2(fMax/1000)
      floor -= TILT_DB_PER_OCT * octBelow1k;
      ceiling += TILT_DB_PER_OCT * octAbove1k;
    }
    const span = Math.max(1, ceiling - floor);

    // Server uses a deeply-negative sentinel (-1000 dB) for log bins with no
    // rfft bin mapped to them; real measurements clamp at -240 dB worst case.
    const SENTINEL_THRESHOLD = -500;
    const barW = w / n;
    for (let i = 0; i < n; i++) {
      const raw = bins[i];
      if (raw < SENTINEL_THRESHOLD) {
        peaks[i] = Math.max(0, peaks[i] - 0.6 * dt);
        continue;
      }
      let db = raw;
      if (tiltOn) {
        // log-spaced bin centers: f = 10^(logFmin + (i+0.5)/n * logSpan)
        const logF = logFminTilt + ((i + 0.5) / n) * logSpanTilt;
        const octavesFrom1k = (logF - 3) / LOG10_2; // log2(f/1000)
        db += TILT_DB_PER_OCT * octavesFrom1k;
      }
      const v = Math.max(0, Math.min(1, (db - floor) / span));
      // peak hold
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - 0.6 * dt);

      const barH = v * h;
      const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
      const r = LUT[cidx*3], g = LUT[cidx*3+1], b = LUT[cidx*3+2];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
      // peak tick (slow bar) — brighter and a bit thicker so it reads clearly
      const py = h - peaks[i] * h;
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillRect(i * barW, py - 1, Math.max(1, barW - 1), 2);
    }

    // L/M/H band region overlay — match the FFT's log-frequency axis
    const meta = store.meta || {};
    const fMin = meta.fft?.f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;
    const bands = meta.bands;
    if (bands && fMax > fMin) {
      const logFmin = Math.log10(fMin);
      const logSpan = Math.log10(fMax) - logFmin;
      const freqToX = (f) => {
        if (f <= fMin) return 0;
        if (f >= fMax) return w;
        return ((Math.log10(f) - logFmin) / logSpan) * w;
      };
      for (const name of LMH_ORDER) {
        const b = bands[name];
        if (!b) continue;
        const x0 = freqToX(b.lo_hz);
        const x1 = freqToX(b.hi_hz);
        if (x1 <= x0) continue;
        const c = LMH[name].rgb;
        ctx.fillStyle = `rgba(${c},0.15)`;
        ctx.fillRect(x0, 0, x1 - x0, h);
        // edges — slightly more opaque so the boundaries pop
        ctx.fillStyle = `rgba(${c},0.7)`;
        ctx.fillRect(x0, 0, 1, h);
        ctx.fillRect(x1 - 1, 0, 1, h);
      }
    }

    recordVizPerf("fft", performance.now() - t0);
  }
  return { draw };
}
