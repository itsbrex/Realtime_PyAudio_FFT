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
//
// Interactive band overlay (when setInteractive(true)): click a colored band
// to select it; drag the body to shift, drag an edge handle to move that
// edge. Mirrors the freq_axis side-panel widget — same set_band messages,
// same selection/drag model — so when FFT is enabled it replaces the
// side-panel "Bandpass edges" UI.

import { store, recordVizPerf } from "../store.js";
import { LMH, LMH_ORDER } from "../colors.js";
import { send } from "../ws.js";

const LUT_STR = (() => {
  const out = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = Math.round(255 * Math.min(1, Math.max(0, -0.2 + 1.6 * t)));
    const g = Math.round(255 * (0.1 + 0.85 * Math.sin(Math.PI * t)));
    const b = Math.round(255 * Math.max(0, 1 - 1.6 * t + 0.5 * Math.pow(t, 4)));
    out[i] = `rgb(${r},${g},${b})`;
  }
  return out;
})();

const SENTINEL_THRESHOLD = -500;
const MIN_BAR_PX = 1;

const X_TICK_CANDIDATES = [30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

// Band-edit constants — match server validators / freq_axis.js.
const F_AXIS_MIN = 20;
const MIN_GAP_HZ = 50;
const HANDLE_W_PX = 10;        // CSS-px hit-width for edge handles
const ORDER = ["low", "mid", "high"];

// Mirrors snapHz in controls.js / freq_axis.js so the overlay and sliders
// agree on values.
function snapHz(f) {
  const step = Math.min(128, Math.max(8, Math.pow(2, Math.round(Math.log2(f / 40)))));
  return Math.max(step, Math.round(f / step) * step);
}

function fmtHz(v) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)} kHz` : `${Math.round(v)} Hz`;
}

export function makeFft(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  let peaks = null;
  let _vCache = null;
  let lastT = performance.now();

  // Band-edit state — mirrors meta.bands; only updated from meta when not
  // mid-drag (so server echoes don't fight the user).
  const bandState = {
    low:  { lo_hz: 30,   hi_hz: 250   },
    mid:  { lo_hz: 250,  hi_hz: 4000  },
    high: { lo_hz: 4000, hi_hz: 16000 },
  };
  let interactive = false;
  let selected = null;
  let drag = null;
  // CSS-px layout from the most recent draw, used by pointer hit-testing.
  let layout = null;

  // Cached CSS-px size, updated by ResizeObserver — avoids forcing a layout
  // read (getBoundingClientRect) every frame.
  let cssW = 0, cssH = 0;
  {
    const r0 = canvas.getBoundingClientRect();
    cssW = r0.width; cssH = r0.height;
    const ro = new ResizeObserver((entries) => {
      const e = entries[entries.length - 1];
      const cr = e.contentRect;
      cssW = cr.width; cssH = cr.height;
    });
    ro.observe(canvas);
  }

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(cssW * dpr));
    const h = Math.max(1, Math.floor(cssH * dpr));
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

    // Pad values are in CSS px — multiply by dpr for canvas-internal coords.
    const padL_css = 38, padR_css = 6, padT_css = 4, padB_css = 16;
    const padL = Math.round(padL_css * dpr);
    const padR = Math.round(padR_css * dpr);
    const padT = Math.round(padT_css * dpr);
    const padB = Math.round(padB_css * dpr);
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
      layout = null;
      recordVizPerf("fft", performance.now() - t0);
      return;
    }
    if (!peaks || peaks.length !== n) {
      peaks = new Float32Array(n);
      _vCache = new Float32Array(n);
    }

    const meta = store.meta || {};
    const rawDb = !!meta.fft_send_raw_db;
    const fMin = meta.fft_f_min ?? meta.fft?.f_min ?? 30;
    const sr = meta.sr ?? 48000;
    const fMax = sr / 2;
    const logFmin = Math.log10(fMin);
    const logSpan = Math.max(1e-6, Math.log10(fMax) - logFmin);

    // ---------------- bars ----------------
    // Two-pass paint: bar pass switches fillStyle per bin (precomputed LUT
    // strings — no per-frame allocation, browser caches the parsed colors),
    // peak-tick pass uses a single fillStyle for all ticks.
    const barW = plotW / n;
    const drawW = Math.max(1, barW - 1);
    const minBarPx = rawDb ? 0 : MIN_BAR_PX;
    let drewAnyBar = false;
    if (rawDb) {
      const floor = meta.fft_db_floor ?? -60;
      const ceiling = meta.fft_db_ceiling ?? 0;
      const span = Math.max(1, ceiling - floor);
      for (let i = 0; i < n; i++) {
        const raw = bins[i];
        if (raw < SENTINEL_THRESHOLD) {
          peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
          // Mark as sentinel so the peak-tick pass skips it.
          // Using NaN is unambiguous since v is always finite [0,1].
          _vCache[i] = NaN;
          continue;
        }
        const v = Math.max(0, Math.min(1, (raw - floor) / span));
        if (v > peaks[i]) peaks[i] = v;
        else peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
        const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
        ctx.fillStyle = LUT_STR[cidx];
        const barH = Math.max(minBarPx, v * plotH);
        ctx.fillRect(plotX + i * barW, plotY + plotH - barH, drawW, barH);
        _vCache[i] = v;
        drewAnyBar = true;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const v = Math.max(0, Math.min(1, bins[i]));
        if (v > peaks[i]) peaks[i] = v;
        else peaks[i] = Math.max(0, peaks[i] - peakDecay * dt);
        const cidx = Math.min(255, Math.max(0, (v * 255) | 0));
        ctx.fillStyle = LUT_STR[cidx];
        const barH = Math.max(minBarPx, v * plotH);
        ctx.fillRect(plotX + i * barW, plotY + plotH - barH, drawW, barH);
        _vCache[i] = v;
        drewAnyBar = true;
      }
    }

    // Peak ticks — one fillStyle for all of them.
    if (drewAnyBar) {
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      for (let i = 0; i < n; i++) {
        if (_vCache[i] !== _vCache[i]) continue; // NaN check: skip sentinels
        const py = plotY + plotH - peaks[i] * plotH;
        ctx.fillRect(plotX + i * barW, py - 1, drawW, 2);
      }
    }

    // ---------------- band overlay (interactive when `interactive`) ---------
    const fMaxHard = Math.min(22000, 0.45 * sr);
    const cssPlotX = padL_css;
    const cssPlotY = padT_css;
    const cssPlotW = Math.max(1, cssW - padL_css - padR_css);
    const cssPlotH = Math.max(1, cssH - padT_css - padB_css);
    layout = {
      cssPlotX, cssPlotY, cssPlotW, cssPlotH,
      fMin, fMax, logFmin, logSpan, sr, fMaxHard,
    };

    drawBandsOverlay(meta, plotX, plotY, plotW, plotH, dpr, fMin, fMax, logFmin, logSpan);

    // ---------------- axes ----------------
    drawYAxis(rawDb, meta, plotX, plotY, plotH, dpr);
    drawXAxis(fMin, fMax, logFmin, logSpan, plotX, plotY, plotW, plotH, padB, dpr);

    recordVizPerf("fft", performance.now() - t0);
  }

  function drawBandsOverlay(meta, plotX, plotY, plotW, plotH, dpr, fMin, fMax, logFmin, logSpan) {
    if (fMax <= fMin) return;
    // Pick which band geometry to draw: live drag state when interactive,
    // server's meta.bands otherwise (passive overlay).
    const src = interactive ? bandState : (meta.bands || null);
    if (!src) return;

    const freqToCanvasX = (f) => {
      if (f <= fMin) return plotX;
      if (f >= fMax) return plotX + plotW;
      return plotX + ((Math.log10(f) - logFmin) / logSpan) * plotW;
    };

    for (const name of LMH_ORDER) {
      const b = src[name];
      if (!b) continue;
      const x0 = freqToCanvasX(b.lo_hz);
      const x1 = freqToCanvasX(b.hi_hz);
      if (x1 <= x0) continue;
      const c = LMH[name].rgb;
      const isSel = interactive && (name === selected);
      const fillA   = isSel ? 0.30 : (interactive ? 0.10 : 0.15);
      const edgeA   = isSel ? 1.00 : 0.55;
      ctx.fillStyle = `rgba(${c},${fillA})`;
      ctx.fillRect(x0, plotY, x1 - x0, plotH);
      ctx.fillStyle = `rgba(${c},${edgeA})`;
      ctx.fillRect(x0, plotY, isSel ? 2 : 1, plotH);
      ctx.fillRect(x1 - (isSel ? 2 : 1), plotY, isSel ? 2 : 1, plotH);

      if (isSel) {
        // Edge handles — small filled bars, full plot height.
        const handleW = Math.max(2, Math.round(4 * dpr));
        ctx.fillStyle = `rgba(${c},0.95)`;
        ctx.fillRect(x0 - handleW / 2, plotY, handleW, plotH);
        ctx.fillRect(x1 - handleW / 2, plotY, handleW, plotH);

        // Edge frequency labels — top of plot.
        ctx.font = `${Math.round(10 * dpr)}px ui-sans-serif`;
        ctx.fillStyle = "#d6d9dc";
        ctx.textBaseline = "top";
        ctx.textAlign = "center";
        const ty = plotY + Math.round(2 * dpr);
        // Background chip so the text is readable over bars.
        const drawChip = (cx, text) => {
          const w = ctx.measureText(text).width + Math.round(8 * dpr);
          const h = Math.round(13 * dpr);
          ctx.fillStyle = "rgba(0,0,0,0.65)";
          ctx.fillRect(cx - w / 2, ty - 1, w, h);
          ctx.fillStyle = "#d6d9dc";
          ctx.fillText(text, cx, ty + 1);
        };
        drawChip(x0, fmtHz(b.lo_hz));
        drawChip(x1, fmtHz(b.hi_hz));
      }
    }
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
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + (canvas.width - plotX), y);
      ctx.stroke();
    }
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
      ctx.fillRect(x, plotY + plotH, 1, Math.round(3 * dpr));
    }
    ctx.restore();
  }

  // ---------------- Interactive band-edit ----------------

  function evtToCss(evt) {
    const r = canvas.getBoundingClientRect();
    return { x: evt.clientX - r.left, y: evt.clientY - r.top };
  }

  function cssXToFreq(cssX) {
    const { cssPlotX, cssPlotW, logFmin, logSpan } = layout;
    const t = (cssX - cssPlotX) / cssPlotW;
    return Math.pow(10, logFmin + t * logSpan);
  }

  function cssFreqToX(f) {
    const { cssPlotX, cssPlotW, fMin, fMax, logFmin, logSpan } = layout;
    if (f <= fMin) return cssPlotX;
    if (f >= fMax) return cssPlotX + cssPlotW;
    return cssPlotX + ((Math.log10(f) - logFmin) / logSpan) * cssPlotW;
  }

  function clampEdges(lo, hi) {
    const fmaxHard = layout ? layout.fMaxHard : 22000;
    if (lo < F_AXIS_MIN) lo = F_AXIS_MIN;
    if (hi > fmaxHard) hi = fmaxHard;
    if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
    return [lo, hi];
  }

  function inPlotY(cssY) {
    return layout && cssY >= layout.cssPlotY && cssY <= layout.cssPlotY + layout.cssPlotH;
  }

  function hitTest(cssX, cssY) {
    if (!layout || !inPlotY(cssY)) return null;
    // Selected band wins: check its handles, then its body.
    if (selected) {
      const s = bandState[selected];
      const xLo = cssFreqToX(s.lo_hz);
      const xHi = cssFreqToX(s.hi_hz);
      if (Math.abs(cssX - xLo) <= HANDLE_W_PX / 2) return { name: selected, role: "lo" };
      if (Math.abs(cssX - xHi) <= HANDLE_W_PX / 2) return { name: selected, role: "hi" };
      if (cssX >= xLo && cssX <= xHi) return { name: selected, role: "body" };
    }
    // Otherwise: walk in reverse paint order so the visually-on-top band
    // wins overlapping hits (matches freq_axis SVG draw order).
    for (let i = ORDER.length - 1; i >= 0; i--) {
      const name = ORDER[i];
      if (name === selected) continue;
      const s = bandState[name];
      const xLo = cssFreqToX(s.lo_hz);
      const xHi = cssFreqToX(s.hi_hz);
      if (cssX >= xLo && cssX <= xHi) return { name, role: "body" };
    }
    return null;
  }

  function startDrag(name, mode, cssX) {
    drag = {
      name, mode,
      startX: cssX,
      startLo: bandState[name].lo_hz,
      startHi: bandState[name].hi_hz,
    };
  }

  function onPointerDown(evt) {
    if (!interactive || !layout) return;
    const { x, y } = evtToCss(evt);
    const hit = hitTest(x, y);
    if (!hit) {
      // Click outside any band → deselect.
      if (selected !== null) selected = null;
      return;
    }
    if (hit.role === "body") {
      if (selected !== hit.name) {
        selected = hit.name;
        evt.preventDefault();
        return; // first click only selects
      }
      startDrag(hit.name, "body", x);
    } else {
      // Edge handle — only reachable when its band is already selected.
      startDrag(hit.name, hit.role, x);
    }
    evt.preventDefault();
    try { canvas.setPointerCapture(evt.pointerId); } catch (e) {}
  }

  function onPointerMove(evt) {
    if (!interactive || !layout) return;
    const { x, y } = evtToCss(evt);
    if (drag) {
      let lo = drag.startLo;
      let hi = drag.startHi;
      if (drag.mode === "lo") {
        lo = snapHz(cssXToFreq(x));
        if (lo > hi - MIN_GAP_HZ) lo = hi - MIN_GAP_HZ;
      } else if (drag.mode === "hi") {
        hi = snapHz(cssXToFreq(x));
        if (hi < lo + MIN_GAP_HZ) hi = lo + MIN_GAP_HZ;
      } else {
        // Body drag preserves log-width.
        const ratio = cssXToFreq(x) / cssXToFreq(drag.startX);
        lo = drag.startLo * ratio;
        hi = drag.startHi * ratio;
        const fmin = F_AXIS_MIN, fmax = layout.fMaxHard;
        if (lo < fmin) { const k = fmin / lo; lo *= k; hi *= k; }
        if (hi > fmax) { const k = fmax / hi; lo *= k; hi *= k; }
        lo = snapHz(lo);
        hi = snapHz(hi);
      }
      [lo, hi] = clampEdges(lo, hi);
      bandState[drag.name] = { lo_hz: lo, hi_hz: hi };
      send({ type: "set_band", band: drag.name, lo_hz: lo, hi_hz: hi, commit: false });
      return;
    }
    // Hover: update cursor based on what we're over.
    const hit = hitTest(x, y);
    let cursor = "default";
    if (hit) {
      if (hit.role === "lo" || hit.role === "hi") cursor = "ew-resize";
      else cursor = (hit.name === selected) ? "grab" : "pointer";
    }
    canvas.style.cursor = cursor;
  }

  function onPointerUp(evt) {
    if (!drag) return;
    const { name } = drag;
    const s = bandState[name];
    drag = null;
    try { canvas.releasePointerCapture(evt.pointerId); } catch (e) {}
    send({ type: "set_band", band: name, lo_hz: s.lo_hz, hi_hz: s.hi_hz, commit: true });
  }

  canvas.addEventListener("pointerdown",   onPointerDown);
  canvas.addEventListener("pointermove",   onPointerMove);
  canvas.addEventListener("pointerup",     onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("pointerleave",  () => { canvas.style.cursor = "default"; });

  // Page-level deselect when clicking outside the canvas.
  document.addEventListener("pointerdown", (evt) => {
    if (!interactive || selected === null) return;
    if (evt.target === canvas) return;
    selected = null;
  });

  return {
    draw,
    /** Mirror server-confirmed band geometry. Ignored mid-drag so the user's
     *  in-flight values aren't clobbered by the echo. */
    syncBands(metaBands) {
      if (drag) return;
      if (!metaBands) return;
      for (const name of ORDER) {
        const b = metaBands[name];
        if (b && typeof b.lo_hz === "number" && typeof b.hi_hz === "number") {
          bandState[name] = { lo_hz: b.lo_hz, hi_hz: b.hi_hz };
        }
      }
    },
    setInteractive(on) {
      interactive = !!on;
      if (interactive) {
        // Mirror the sidebar "Bandpass edges" tooltip onto the canvas so
        // users discover the bands are clickable. Pulled live from the
        // sidebar element so the two stay in sync; falls back to a hardcoded
        // copy if the element isn't reachable.
        const src = document.getElementById("freq-axis");
        const tip = (src && src.getAttribute("data-tooltip"))
          || "Click a colored band to select it. Once selected, drag its body to shift the whole band, or drag its left/right handle to move just that edge. Click outside any band to deselect. Bands may overlap.";
        canvas.setAttribute("data-tooltip", tip);
      } else {
        selected = null;
        drag = null;
        canvas.style.cursor = "default";
        canvas.removeAttribute("data-tooltip");
      }
    },
  };
}
