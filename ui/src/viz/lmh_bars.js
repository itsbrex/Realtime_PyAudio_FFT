// Three vertical bars with peak-hold ticks decaying at ~1.5/sec, plus an
// onset indicator (square, same width as the bar) directly above each band
// that flashes on the corresponding onset (low / mid / high).

import { store, recordVizPerf } from "../store.js";
import { LMH_HEX, LMH_ORDER } from "../colors.js";

const COLORS = LMH_HEX;
const LABELS = LMH_ORDER;

// Onset-flash decay time (seconds): time for the flash to fade from 1 → 0
// after an onset. Longer = flashes linger visually; shorter = each pulse is
// a sharper blink. Matches the perceived snappiness of typical drum hits.
const ONSET_FLASH_DECAY_S = 0.20;

export function makeBars(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const peaks = [0, 0, 0];
  let lastT = performance.now();

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
    const onsetTs = [
      store.low_onset_pulse_t,
      store.mid_onset_pulse_t,
      store.high_onset_pulse_t,
    ];
    const dpr = window.devicePixelRatio || 1;
    const padX = 16 * dpr, padY = 14 * dpr;
    const showOnsets = !!store.show_onsets;

    // Bar geometry. When the onset toggle is OFF, bars fill the full
    // vertical space (no reserved square area, no visible difference from
    // the original peak-hold-only layout). When ON, a square of the same
    // width as the bar sits above each band's column and the bars shorten
    // to fit beneath.
    const slotW = (w - 2 * padX) / 3;
    const barW = slotW * 0.6;
    const onsetSize = showOnsets ? barW : 0;
    const onsetGap = showOnsets ? 6 * dpr : 0;
    const onsetTop = padY;
    const barsTop = padY + onsetSize + onsetGap;
    const barsUsableH = (h - padY) - barsTop;

    for (let i = 0; i < 3; i++) {
      const v = Math.max(0, Math.min(1, vals[i]));
      // peak hold over the bar zone only (not the onset square)
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - (store.peak_decay_per_s ?? 0.6) * dt);

      const x = padX + slotW * i + (slotW - barW) / 2;
      const baseY = h - padY;

      // ---- Onset square (above the bar) — only when toggle is on ----
      if (showOnsets) {
        const sinceOnset = (now - onsetTs[i]) / 1000;
        const onsetFill = sinceOnset >= 0 && sinceOnset < ONSET_FLASH_DECAY_S
          ? 1.0 - (sinceOnset / ONSET_FLASH_DECAY_S)
          : 0.0;
        ctx.fillStyle = "#1f2227";
        ctx.fillRect(x, onsetTop, onsetSize, onsetSize);
        if (onsetFill > 0) {
          // Tint the band's signature color toward white at peak so the
          // onset reads as "this band fired" while still color-coded.
          const baseHex = COLORS[i];
          const r = parseInt(baseHex.slice(1, 3), 16);
          const g = parseInt(baseHex.slice(3, 5), 16);
          const b = parseInt(baseHex.slice(5, 7), 16);
          const mix = (c) => Math.round(c + (255 - c) * onsetFill);
          ctx.fillStyle = `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
          ctx.fillRect(x, onsetTop, onsetSize, onsetSize);
        }
      }

      // ---- L/M/H bar ----
      const barH = barsUsableH * v;
      ctx.fillStyle = "#1f2227";
      ctx.fillRect(x, barsTop, barW, barsUsableH);
      ctx.fillStyle = COLORS[i];
      ctx.fillRect(x, baseY - barH, barW, barH);
      // peak tick
      const peakY = baseY - barsUsableH * peaks[i];
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
