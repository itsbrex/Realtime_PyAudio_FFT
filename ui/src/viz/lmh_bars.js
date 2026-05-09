// Three vertical bars with peak-hold ticks decaying at ~1.5/sec, plus a
// fourth "beat" bar on the very left that flashes on each detected onset.

import { store, recordVizPerf } from "../store.js";
import { LMH_HEX, LMH_ORDER } from "../colors.js";

const COLORS = LMH_HEX;
const LABELS = LMH_ORDER;

// Beat-flash decay time (seconds): time for the flash to fade from 1 → 0
// after a beat onset. Longer = beats linger visually; shorter = each pulse
// is a sharper blink. Matches the perceived snappiness of typical kick hits.
const BEAT_FLASH_DECAY_S = 0.20;

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
    const dpr = window.devicePixelRatio || 1;
    const padX = 16 * dpr, padY = 14 * dpr;

    // Beat flash fill level (1.0 right after onset, decays linearly).
    const sinceBeat = (now - store.beat_pulse_t) / 1000;
    const beatFill = sinceBeat >= 0 && sinceBeat < BEAT_FLASH_DECAY_S
      ? 1.0 - (sinceBeat / BEAT_FLASH_DECAY_S)
      : 0.0;

    // Layout: 4 slots — beat (narrower, slightly separated) + L + M + H.
    // Beat bar takes ~half a normal slot so it visually reads as "extra"
    // rather than crowding the L/M/H trio.
    const beatSlotFrac = 0.5;
    const totalSlots = 3 + beatSlotFrac;
    const slotUnit = (w - 2 * padX) / totalSlots;
    const beatSlotW = slotUnit * beatSlotFrac;
    const lmhSlotW = slotUnit;
    const lmhBarW = lmhSlotW * 0.6;
    const beatBarW = beatSlotW * 0.7;

    // ---- Beat flash bar (leftmost) ----
    {
      const x = padX + (beatSlotW - beatBarW) / 2;
      const baseY = h - padY;
      const usableH = h - 2 * padY;
      // Track background.
      ctx.fillStyle = "#1f2227";
      ctx.fillRect(x, padY, beatBarW, usableH);
      // Filled portion (full-height when fresh, decaying down).
      if (beatFill > 0) {
        const fillH = usableH * beatFill;
        // Bright accent — distinct from the L/M/H palette so it reads as a
        // trigger event, not a fourth band. Slight color cue: pure white at
        // peak fading to a cooler blueish hue.
        const intensity = Math.round(255 * beatFill);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, 255)`;
        ctx.fillRect(x, baseY - fillH, beatBarW, fillH);
      }
      ctx.fillStyle = "#8b939c";
      ctx.font = `${10 * dpr}px ui-sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("beat", x + beatBarW / 2, h - 2);
    }

    // ---- L/M/H bars ----
    const lmhStartX = padX + beatSlotW;
    for (let i = 0; i < 3; i++) {
      const v = Math.max(0, Math.min(1, vals[i]));
      // peak hold
      if (v > peaks[i]) peaks[i] = v;
      else peaks[i] = Math.max(0, peaks[i] - (store.peak_decay_per_s ?? 0.6) * dt);

      const x = lmhStartX + lmhSlotW * i + (lmhSlotW - lmhBarW) / 2;
      const baseY = h - padY;
      const barH = (h - 2 * padY) * v;
      ctx.fillStyle = "#1f2227";
      ctx.fillRect(x, padY, lmhBarW, h - 2 * padY);
      ctx.fillStyle = COLORS[i];
      ctx.fillRect(x, baseY - barH, lmhBarW, barH);
      // peak tick
      const peakY = baseY - (h - 2 * padY) * peaks[i];
      ctx.fillStyle = "#fff";
      ctx.fillRect(x, peakY - 1, lmhBarW, 2);
      // label
      ctx.fillStyle = "#8b939c";
      ctx.font = `${10 * dpr}px ui-sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(LABELS[i], x + lmhBarW / 2, h - 2);
    }
    recordVizPerf("bars", performance.now() - t0);
  }
  return { draw };
}
