// Single source of truth for live data and meta. WS handlers update; UI reads.

export const store = {
  // Live signal
  low: 0, mid: 0, high: 0,
  low_raw: 0, mid_raw: 0, high_raw: 0,
  fft_bins: null, // Float32Array
  fft_db_floor: -80,
  fft_db_ceiling: 0,

  // Meta
  meta: {
    sr: 48000,
    blocksize: 256,
    n_fft_bins: 128,
    low_hz: 250,
    high_hz: 4000,
    fft_enabled: false,
    tau: { low: 0.15, mid: 0.06, high: 0.02 },
    autoscale: { tau_release_s: 60, noise_floor: 0.001, strength: 1.0 },
    ws_snapshot_hz: 60,
    device: { index: null, name: null }
  },

  devices: [],
  presets: [],

  status: {
    cb_overruns: 0, dsp_drops: 0, fft_drops: 0,
    perf: null
  },

  // Server-side message rate measurement
  msgTimestamps: [], // last 60 perf.now() of any inbound msg

  // Browser-side perf
  raf_ms_ring: new Array(60).fill(0),
  raf_idx: 0,

  // Per-viz render time rings (ms)
  viz_perf: {
    lines: { ring: new Array(60).fill(0), idx: 0 },
    bars:  { ring: new Array(60).fill(0), idx: 0 },
    scene: { ring: new Array(60).fill(0), idx: 0 },
    fft:   { ring: new Array(60).fill(0), idx: 0 }
  },

  // Last-confirmed values for slider snap-back on error
  lastConfirmed: {}
};

export function recordVizPerf(name, ms) {
  const v = store.viz_perf[name];
  if (!v) return;
  v.ring[v.idx % v.ring.length] = ms;
  v.idx++;
}

export function avgRing(ring) {
  let s = 0, n = 0;
  for (const v of ring) { if (v > 0) { s += v; n++; } }
  return n ? s / n : 0;
}

export function p95Ring(ring) {
  const sorted = ring.filter(v => v > 0).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
}
