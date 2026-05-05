// Spatially-separated L/M/H scene:
//   low  -> central glowing disc (radius + brightness)
//   mid  -> ring around the disc  (radius offset + thickness + alpha)
//   high -> particles radiating outward from the disc (speed + alpha)
// Each band gets its own region so the eye can read them independently,
// while motion (particles) carries the high band — easier to perceive
// than dot density alone.

import { store, recordVizPerf } from "../store.js";
import { LMH } from "../colors.js";

const N_PARTICLES = 200;

export function makeScene(canvas) {
  const ctx = canvas.getContext("2d", { alpha: false });

  // Particle pool, pre-allocated.
  const px  = new Float32Array(N_PARTICLES);
  const py  = new Float32Array(N_PARTICLES);
  const pvx = new Float32Array(N_PARTICLES);
  const pvy = new Float32Array(N_PARTICLES);
  let initialized = false;
  let lastT = performance.now();

  function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width  * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      initialized = false;
    }
  }

  function spawn(i, cx, cy, r0) {
    const a = Math.random() * Math.PI * 2;
    const r = r0 * (0.6 + Math.random() * 0.4);
    px[i] = cx + Math.cos(a) * r;
    py[i] = cy + Math.sin(a) * r;
    const speed = 30 + Math.random() * 90;
    pvx[i] = Math.cos(a) * speed;
    pvy[i] = Math.sin(a) * speed;
  }

  function initParticles(w, h) {
    const cx = w / 2, cy = h / 2;
    const r0 = Math.min(w, h) * 0.15;
    for (let i = 0; i < N_PARTICLES; i++) spawn(i, cx, cy, r0);
    initialized = true;
  }

  function draw() {
    const t0 = performance.now();
    fitCanvas();
    const w = canvas.width, h = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    if (!initialized) initParticles(w, h);

    const now = performance.now();
    // Cap dt so backgrounded tabs don't fling all particles off-screen.
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;

    const lo = Math.max(0, Math.min(1, store.low));
    const md = Math.max(0, Math.min(1, store.mid));
    const hi = Math.max(0, Math.min(1, store.high));

    // Background.
    ctx.fillStyle = "#0a0b0d";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2;
    const baseR = Math.min(w, h) * 0.5;

    // --- LOW: central disc with radial gradient. ---
    const lowR = baseR * (0.15 + 0.40 * lo);
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, lowR);
    const light = 28 + 50 * lo;                // 28..78
    const alpha = 0.35 + 0.55 * lo;            // 0.35..0.90
    grad.addColorStop(0,    `hsla(${LMH.low.hue}, 80%, ${light}%, ${alpha})`);
    grad.addColorStop(0.65, `hsla(${LMH.low.hue}, 80%, ${light}%, ${alpha * 0.5})`);
    grad.addColorStop(1,    `hsla(${LMH.low.hue}, 80%, ${light}%, 0)`);
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, lowR, 0, Math.PI * 2);
    ctx.fill();

    // --- MID: ring just outside the disc. ---
    if (md > 0.02) {
      const midR  = lowR + baseR * (0.04 + 0.18 * md);
      const ringW = Math.max(1, baseR * 0.06 * md);
      ctx.strokeStyle = `hsla(${LMH.mid.hue}, 55%, 60%, ${0.3 + 0.6 * md})`;
      ctx.lineWidth = ringW;
      ctx.beginPath();
      ctx.arc(cx, cy, midR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // --- HIGH: particles drifting outward from the disc. ---
    // Speed scales with hi; particles always exist so position state is stable
    // when hi drops back up — they just slow to a near-stop.
    const speedScale = 0.25 + 3.5 * hi;
    const maxR2 = (Math.hypot(w, h) * 0.5 + 8) ** 2;
    const r0    = lowR;                                  // respawn radius
    const sz    = Math.max(1, Math.round(1.5 * dpr));
    ctx.fillStyle = `rgba(${LMH.high.rgb}, ${0.25 + 0.7 * hi})`;
    ctx.beginPath();
    for (let i = 0; i < N_PARTICLES; i++) {
      px[i] += pvx[i] * dt * speedScale;
      py[i] += pvy[i] * dt * speedScale;
      const dx = px[i] - cx, dy = py[i] - cy;
      if (dx * dx + dy * dy > maxR2) spawn(i, cx, cy, r0);
      ctx.rect(px[i] | 0, py[i] | 0, sz, sz);
    }
    ctx.fill();

    recordVizPerf("scene", performance.now() - t0);
  }
  return { draw };
}
