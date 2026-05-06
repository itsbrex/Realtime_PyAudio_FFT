// FFT bars — visualizer designed to mirror the L/M/H pipeline so the spectrum
// can be used to tune the IIR band-pass + smoother + auto-scaler chain that
// feeds OSC.
//
// The wire payload is always honest dB (per-log-bin) regardless of mode below.
//
// Two modes:
//
//   "Raw dB" (checkbox ON): bypass everything — render the wire-format dB
//   directly on a fixed [db_floor, db_ceiling] range. Sentinel bins (-1000 dB
//   "no rfft bin mapped") show as gaps. Honest engineering view.
//
//   Default (checkbox OFF): per-bin port of server/dsp/features.py:AutoScaler,
//   driven by the same control values that the L/M/H chain uses:
//
//     1. Sentinel interpolation so the low-end staircase becomes a curve.
//     2. dB → linear amplitude (10^(db/20)).
//     3. Per-bin EMA smoothing. The tau is piecewise-linear-interpolated in
//        log-frequency space from `meta.tau.{low,mid,high}` anchored at each
//        L/M/H band's geometric-mean center frequency. Bins below the lowest
//        band center get tau_low; above the highest center get tau_high.
//     4. Per-bin asymmetric peak follower (fast attack 50 ms = AutoScaler
//        default, slow release `meta.autoscale.tau_release_s`).
//     5. AutoScaler core: subtract noise_floor, divide by max(peak, floor),
//        tanh — bit-for-bit identical to the server-side scaling math.
//     6. Strength blend: `v = s * scaled + (1-s) * raw_db_mapped` where
//        raw_db_mapped is the wire dB clamped to [db_floor, db_ceiling] (with
//        the same noise gate so silent bars stay pinned at the bottom rather
//        than disappearing).
//
// Tilt was dropped on purpose — per-bin peak normalization handles spectral
// tilt naturally, and dropping it keeps strength=0 honest (you see the actual
// pink-ish music spectrum instead of a fake-flattened version).

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

const SENTINEL_THRESHOLD = -500;       // server emits -1000 for empty log bins
const TAU_ATTACK_S = 0.05;             // matches server AutoScaler default
const PEAK_DECAY_PER_S = 0.6;          // peak-marker fade rate (display only)
const MIN_BAR_PX = 1;                  // floor bars at 1 px so silence stays visible

function interpolateSentinels(bins, n, out) {
  let nextIdx = -1;
  let nextVal = NaN;
  for (let i = n - 1; i >= 0; i--) {
    const v = bins[i];
    if (v >= SENTINEL_THRESHOLD) {
      out[i] = v;
      nextIdx = i;
      nextVal = v;
      continue;
    }
    let lIdx = i - 1;
    while (lIdx >= 0 && bins[lIdx] < SENTINEL_THRESHOLD) lIdx--;
    if (lIdx >= 0 && nextIdx >= 0) {
      const lVal = bins[lIdx];
      const t = (i - lIdx) / (nextIdx - lIdx);
      out[i] = lVal + t * (nextVal - lVal);
    } else if (nextIdx >= 0) {
      out[i] = nextVal;
    } else if (lIdx >= 0) {
      out[i] = bins[lIdx];
    } else {
      out[i] = -120;
    }
  }
}

// Piecewise-linear interpolation of (log-f, tau) anchored at L/M/H band
// geometric-mean centers. Bins outside the [low_center, high_center] range
// clamp to the nearest band's tau.
function buildPerBinTau(n, fMin, fMax, bands, tau) {
  const out = new Float32Array(n);
  const anchors = [
    { logF: Math.log10(Math.sqrt(bands.low.lo_hz  * bands.low.hi_hz)),  tau: tau.low  },
    { logF: Math.log10(Math.sqrt(bands.mid.lo_hz  * bands.mid.hi_hz)),  tau: tau.mid  },
    { logF: Math.log10(Math.sqrt(bands.high.lo_hz * bands.high.hi_hz)), tau: tau.high },
  ].sort((a, b) => a.logF - b.logF);
  const logFmin = Math.log10(fMin);
  const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);
  for (let i = 0; i < n; i++) {
    const logF = logFmin + ((i + 0.5) / n) * logSpan;
    if (logF <= anchors[0].logF) {
      out[i] = anchors[0].tau;
    } else if (logF >= anchors[2].logF) {
      out[i] = anchors[2].tau;
    } else {
      const j = logF <= anchors[1].logF ? 0 : 1;
      const a0 = anchors[j], a1 = anchors[j + 1];
      const t = (logF - a0.logF) / (a1.logF - a0.logF);
      out[i] = a0.tau + t * (a1.tau - a0.tau);
    }
  }
  return out;
}

// 5-tap binomial smoothing [1,4,6,4,1]/16, edge-replicate boundary.
function smoothSpectrum(arr, n, scratch) {
  for (let i = 0; i < n; i++) {
    const im2 = arr[Math.max(0, i - 2)];
    const im1 = arr[Math.max(0, i - 1)];
    const ic  = arr[i];
    const ip1 = arr[Math.min(n - 1, i + 1)];
    const ip2 = arr[Math.min(n - 1, i + 2)];
    scratch[i] = (im2 + 4 * im1 + 6 * ic + 4 * ip1 + ip2) * (1 / 16);
  }
  arr.set(scratch);
}

export function makeFft(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });

  let interpDb       = null; // post-sentinel-fill dB values
  let smoothLin      = null; // per-bin smoothed linear amplitude (matches LMH ExpSmoother._values per band)
  let peakLin        = null; // per-bin peak follower (matches AutoScaler._peak per band)
  let dispBuf        = null; // post-blend display values 0..1
  let smoothScratch  = null; // scratch for spectral smoothing
  let peaks          = null; // peak-hold display markers, 0..1

  let perBinTau      = null;
  let perBinTauKey   = "";

  let primed         = false;
  let lastRawDb      = null;
  let lastT          = performance.now();

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
  }

  function ensureBuffers(n) {
    if (!interpDb       || interpDb.length       !== n) interpDb       = new Float32Array(n);
    if (!dispBuf        || dispBuf.length        !== n) dispBuf        = new Float32Array(n);
    if (!smoothScratch  || smoothScratch.length  !== n) smoothScratch  = new Float32Array(n);
    if (!peaks          || peaks.length          !== n) peaks          = new Float32Array(n);
    if (!smoothLin      || smoothLin.length      !== n) { smoothLin    = new Float32Array(n); primed = false; }
    if (!peakLin        || peakLin.length        !== n) { peakLin      = new Float32Array(n); primed = false; }
  }

  function ensurePerBinTau(n, fMin, fMax, bands, tau) {
    const key = `${n}|${fMin}|${fMax}|${bands.low.lo_hz},${bands.low.hi_hz}|${bands.mid.lo_hz},${bands.mid.hi_hz}|${bands.high.lo_hz},${bands.high.hi_hz}|${tau.low},${tau.mid},${tau.high}`;
    if (key !== perBinTauKey) {
      perBinTau = buildPerBinTau(n, fMin, fMax, bands, tau);
      perBinTauKey = key;
    }
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    const now = performance.now();
    const dt = Math.max(1e-3, Math.min(0.1, (now - lastT) / 1000));
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
    ensureBuffers(n);

    const rawDb = !!store.fft_raw_db;
    if (rawDb !== lastRawDb) {
      primed = false;       // reseed peak/smoother state on toggle
      lastRawDb = rawDb;
    }

    const meta = store.meta || {};
    const fMin = meta.fft?.f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);

    if (rawDb) {
      drawRawDb(bins, n, w, h, dt);
    } else {
      const bands = meta.bands || {
        low:  { lo_hz: 30,   hi_hz: 250   },
        mid:  { lo_hz: 250,  hi_hz: 4000  },
        high: { lo_hz: 4000, hi_hz: 16000 },
      };
      const tau = meta.tau || { low: 0.15, mid: 0.06, high: 0.02 };
      const autoscale = meta.autoscale || { tau_release_s: 60, noise_floor: 0.001, strength: 1.0 };
      ensurePerBinTau(n, fMin, fMax, bands, tau);
      drawScaled(bins, n, w, h, dt, autoscale);
    }

    // L/M/H band overlay (same in both modes)
    const bands = meta.bands;
    if (bands && fMax > fMin) {
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
        ctx.fillStyle = `rgba(${c},0.7)`;
        ctx.fillRect(x0, 0, 1, h);
        ctx.fillRect(x1 - 1, 0, 1, h);
      }
    }

    recordVizPerf("fft", performance.now() - t0);
  }

  function drawRawDb(bins, n, w, h, dt) {
    // Honest readout: wire dB on fixed range, sentinels = gaps.
    const floor = store.fft_db_floor || -60;
    const ceiling = store.fft_db_ceiling || 0;
    const span = Math.max(1, ceiling - floor);
    const barW = w / n;
    for (let i = 0; i < n; i++) {
      const raw = bins[i];
      if (raw < SENTINEL_THRESHOLD) {
        peaks[i] = Math.max(0, peaks[i] - PEAK_DECAY_PER_S * dt);
        continue; // gap (no bar at all — honest indication of "no data")
      }
      const v = Math.max(0, Math.min(1, (raw - floor) / span));
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - PEAK_DECAY_PER_S * dt);
      paintBar(i, v, w, h, barW, /*minPx*/ 0);
    }
  }

  function drawScaled(bins, n, w, h, dt, autoscale) {
    interpolateSentinels(bins, n, interpDb);

    // ---- 1. dB -> linear, then per-bin EMA smoother (mirrors ExpSmoother) ----
    if (!primed) {
      // Seed smoother and peak from current frame so we don't see a slow fade-in.
      for (let i = 0; i < n; i++) {
        const lin = Math.pow(10, interpDb[i] / 20);
        smoothLin[i] = lin;
        peakLin[i] = Math.max(lin, autoscale.noise_floor);
      }
      primed = true;
    } else {
      for (let i = 0; i < n; i++) {
        const lin = Math.pow(10, interpDb[i] / 20);
        const a = 1 - Math.exp(-dt / Math.max(1e-3, perBinTau[i]));
        smoothLin[i] += a * (lin - smoothLin[i]);
      }
    }

    // ---- 2. Per-bin asymmetric peak follower (mirrors AutoScaler._peak) ----
    const aAtk = 1 - Math.exp(-dt / TAU_ATTACK_S);
    const aRel = 1 - Math.exp(-dt / Math.max(1e-3, autoscale.tau_release_s || 60));
    for (let i = 0; i < n; i++) {
      const sm = smoothLin[i];
      const a = sm > peakLin[i] ? aAtk : aRel;
      peakLin[i] += a * (sm - peakLin[i]);
    }

    // ---- 3. AutoScaler core + strength blend ----
    const noiseFloor = Math.max(0, autoscale.noise_floor || 0);
    const strength = Math.max(0, Math.min(1, autoscale.strength ?? 1));
    const noiseFloorDb = noiseFloor > 0 ? 20 * Math.log10(noiseFloor) : -Infinity;
    const dispFloor = store.fft_db_floor || -60;
    const dispCeiling = store.fft_db_ceiling || 0;
    const dispSpan = Math.max(1, dispCeiling - dispFloor);

    for (let i = 0; i < n; i++) {
      const sm = smoothLin[i];
      const denom = Math.max(peakLin[i], noiseFloor, 1e-12);
      const gated = Math.max(0, sm - noiseFloor);
      const scaled = Math.tanh(gated / denom);

      // Raw-side render (used at strength<1): wire dB on the fixed display
      // range, with the same noise gate so silent bins flatten to zero rather
      // than vanishing.
      const dbI = interpDb[i];
      let raw = (dbI - dispFloor) / dispSpan;
      if (raw < 0) raw = 0; else if (raw > 1) raw = 1;
      if (dbI < noiseFloorDb) raw = 0;

      dispBuf[i] = strength * scaled + (1 - strength) * raw;
    }

    // ---- 4. Light spectral smoothing for visual stability ----
    smoothSpectrum(dispBuf, n, smoothScratch);

    // ---- 5. Render with peak hold ----
    const barW = w / n;
    for (let i = 0; i < n; i++) {
      const v = Math.max(0, Math.min(1, dispBuf[i]));
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - PEAK_DECAY_PER_S * dt);
      paintBar(i, v, w, h, barW, MIN_BAR_PX);
    }
  }

  function paintBar(i, v, w, h, barW, minPx) {
    const barH = Math.max(minPx, v * h);
    const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
    const r = LUT[cidx*3], g = LUT[cidx*3+1], b = LUT[cidx*3+2];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
    const py = h - peaks[i] * h;
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillRect(i * barW, py - 1, Math.max(1, barW - 1), 2);
  }

  return { draw };
}
