# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The design is documented in detail in `realtime_audio_server_plan.md`; if behavior in code seems surprising, the plan is the authoritative source for *why*. **Note:** the plan predates the FFT post-processor — for FFT processing semantics, the code (`server/dsp/fft_postprocess.py`) is the source of truth.

```
server/      # Python audio-server package (entry point: server.main:main)
  audio/     # PortAudio callback, ring buffer, stream lifecycle, device probing
  dsp/       # FilterBank, ExpSmoother, AutoScaler, FFTWorker, DSPWorker,
             # FFTPostProcessor (per-bin port of AutoScaler for the FFT bins)
  control/   # WS message dispatcher + pure validators
  io/        # WSServer, OSC sender task, FeatureStore/FFTStore, static HTTP
  config.py  # Dataclass schema, YAML load, debounced atomic Persister
  main.py    # App orchestrator: builds and wires everything together
ui/          # Vanilla-JS browser UI (ES modules, no build step). Pure viz —
             # no DSP. What you see in the FFT graph is byte-identical to the
             # OSC payload.
config.yaml  # Persisted runtime state — written automatically when UI changes settings
```

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the server (reads ./config.yaml; opens WS on 8765, UI HTTP on 8766)
audio-server                          # entry point from pyproject [project.scripts]
python -m server.main                 # equivalent
audio-server --open                   # also launches the bundled UI in default browser
audio-server --no-ws                  # headless OSC-only mode
audio-server --device 2               # override input device index
audio-server --config /path/cfg.yaml --log-level DEBUG

# Tests (pytest dirs exist but are empty — no tests yet)
pytest tests/
pytest tests/unit/test_foo.py::test_bar    # single test
```

The server *must* be launched with `config.yaml`'s parent directory writable — the Persister atomically rewrites it on every UI control change, and presets are saved alongside as `preset-<name>.yaml`.

## Architecture: the threading contract is the design

This is a realtime audio system. The single most important invariant: **the PortAudio callback never allocates, never logs, never locks, never sends a packet, never calls SciPy.** Its job is mono-mix → ring write → set two events. All DSP runs in worker threads. Violating this — e.g. doing a `sosfilt` call or a `print` from the callback — will cause audio dropouts under load. See `realtime_audio_server_plan.md` §3.

### Data flow

```
PortAudio C thread ──► AudioCallback (server/audio/callback.py)
                          │  memcpy into SlotRing, set dsp_event + fft_event
                          ▼
                       SlotRing (server/audio/ringbuffer.py)
                       SPSC, block-aligned, seqlock publish
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
       DSPWorker (thread)       FFTWorker (thread)
       block-driven             hop-driven, optional
       FilterBank → RMS         windowed rfft → log-spaced bins (dB)
       → ExpSmoother            → FFTPostProcessor (per-bin port of AutoScaler:
       → AutoScaler                sentinel interp (precomputed LUT)
                                    → +tilt → dB→linear
       → FeatureStore.publish()    → per-bin EMA smoother (taus interpolated
              │                     from L/M/H band centers)
              │                     → asymmetric peak follower → spatial
              │                     Gaussian smear across bins (reflect mode)
              │                     → soft gate → tanh → strength blend with
              │                     raw-dB-mapped baseline)
              │                  → FFTStore.publish(raw_db, processed)
              │                    (refs into double-buffered output buffers
              │                     owned by FFTWorker / FFTPostProcessor —
              │                     no copy on the publish path)
              │                        │
              └────► sender_event (asyncio.Event, set via call_soon_threadsafe)
                            │
                            ▼
                  asyncio loop owns:
                    - osc_sender_task  (server/io/osc_sender.py)  → /audio/lmh, /audio/fft
                    - WSServer         (server/io/ws_server.py)   → JSON snapshots + binary FFT
                    - StaticHTTPServer (server/io/http_server.py) → serves ui/ over http
                    - Persister        (server/config.py)         → debounced atomic config.yaml
                    - Dispatcher       (server/control/dispatcher.py) → handles inbound WS control msgs

Browser (ui/) ──ws://127.0.0.1:8765──► Dispatcher (validate → mutate App state → persist → reply)
```

**Single source of truth for FFT.** Both OSC and WS read from the same `FFTStore`, picking either the `raw_db` or `processed` stream based on `cfg.fft.send_raw_db`. The UI reflects this flag from `meta` and renders whatever it receives — no client-side DSP. So the FFT graph in the browser is byte-identical to the `/audio/fft` OSC payload at any moment in time.

### App is the orchestrator

`server.main.App` owns every long-lived object: the stream, ring, workers, stores, sender, WS server, persister, perf rings, the `Config`, and `fft_postprocessor` (the per-bin sibling of `auto_scaler`). Cross-thread communication uses three patterns and only three:

1. **SlotRing**: SPSC seqlock for audio data (callback → DSP/FFT workers).
2. **threading.Event**: one-shot wakeups (`dsp_event`, `fft_event`, `fft_enabled`, `stop_flag`).
3. **FeatureStore / FFTStore**: lock-guarded publish/read (workers → asyncio sender / WS broadcaster). `FFTStore` holds a `(raw_db, processed)` pair; `read(kind)` returns the requested stream.

When the DSP or FFT worker publishes, it calls `App._signal_dsp_published`, which uses `loop.call_soon_threadsafe(sender_event.set)` to wake the OSC sender task. This is the only cross-thread asyncio interaction — never `asyncio.run_coroutine_threadsafe` from a worker.

### What runs where

- **Mutations to `App.cfg`, `filter_bank`, `smoother`, `auto_scaler`, `fft_worker`, `fft_postprocessor` config**: only on the asyncio loop, from `Dispatcher` handlers. Workers read these object attributes; mutators replace or call `set_*` / `update_*` methods. The post-processor has its own `_lock`; `process()` and all `update_*` methods take it.
- **Unified knobs drive both pipelines.** `set_smoothing` updates the L/M/H `ExpSmoother` AND the FFT per-bin smoother (taus interpolated piecewise-linearly in log-frequency from the band geometric centers). `set_autoscale` (`tau_attack_s`, `tau_release_s`, `noise_floor`, `strength`) updates both the L/M/H `AutoScaler` and the FFT `FFTPostProcessor`. `set_band` updates the IIR bandpass AND re-anchors the FFT smoothing-tau interpolation. So one slider = lockstep change across both pipelines.
- **FFT-only knobs.** `set_fft_send_raw_db` flips which `FFTStore` stream is sent (raw dB vs processed). `set_fft_peak_smear` controls a Gaussian smear (in octaves) of the per-bin peak follower so a single-tone bin doesn't fully self-normalize. `set_fft_tilt` (calls `FFTPostProcessor.update_tilt`) re-anchors the per-bin spectral tilt curve. None have an L/M/H equivalent (the L/M/H AutoScaler also has a `tilt_db_per_oct`, but its anchors are the three band centers, not log-spaced bins).
- **Filter retunes**: `App.schedule_filter_retune` debounces 50ms (collapses slider drags into one `FilterBank.retune` call).
- **Device hot-switch**: `App.hot_switch_device` (async, locked) — tears down the stream, resets ring/filter/scaler/smoother/FFT alignment, opens a fresh stream (sample rate may change), rebuilds DSP for the new sr, restarts. The post-processor's smear σ (in *bin* units) is recomputed because bins-per-octave depends on `sr`. Workers stay alive throughout (they idle on event timeouts while the stream is down).
- **Persistence**: every Dispatcher handler calls `app.persister.request(commit=...)`. `commit=False` is the drag path (1s debounce, capped at 250ms from first dirty). `commit=True` is for discrete events (50ms). Atomic write via tempfile + `os.replace`.

### Why FFT and DSP are separate workers

DSP runs once per audio block (256 samples ≈ 5.3ms at 48k). FFT runs once per `hop` samples (default 512) over a `window_size` window (default 1024). Decoupling them lets FFT skip work cleanly when disabled (`fft_enabled.clear()` → worker no-ops on its event tick) without affecting DSP latency.

### FFT post-processing (`server/dsp/fft_postprocess.py`)

Per-bin port of `features.AutoScaler`, intentionally structurally identical to the L/M/H pipeline so a single set of UI knobs tunes both. Runs inside `FFTWorker` immediately after the bincount, on the same hop-rate clock:

1. **Sentinel interpolation (precomputed LUT).** Empty-log-bin sentinels (`-1000 dB`, generated at the low end where rfft Δf > log-bin width) are filled with a linear blend of their nearest valid neighbors so the spectrum is continuous. The blend coefficients (`_left_idx`, `_right_idx`, `_w_left`, `_w_right`) are computed once on the first hop after `_allocate()` (bin layout is stable until the next `reconfigure()`), so the hot path is a pure `np.take` + multiply-add + scatter — no `np.interp` call, no per-hop temporaries. Edge handling: empty bins below the lowest valid bin clamp to that bin's value (w_left=0); empty bins above the highest valid bin clamp to that bin's value (w_right=0). Note: clamping sentinels to `db_floor` instead of interpolating produces visible gaps at the low end because the gate downstream zeros bins below `noise_floor`, and a sentinel-clamped low bin sits well below it — interpolation propagates the loud first valid bin's value across the empty gap so the gate fires uniformly.
2. **Pre-tilt + dB → linear.** `_lin = 10^((interp_db + tilt_db) / 20)`, where `tilt_db[k] = tilt_db_per_oct · log2(f_center[k] / 1 kHz)`. Calibration to L/M/H RMS² scale lives in `FFTWorker._win_power_corr = 2 / sum(hann)²`, applied at bincount time: a sine of amplitude A reads as bin-power A²/2 = sine RMS², so the same `noise_floor` value gates similarly on both pipelines without any extra dB shift.
3. **Per-bin EMA smoother.** Tau is piecewise-linear interpolation of `(log10(f_center), tau_band)` anchors at the L/M/H band geometric-mean centers. Outside the lowest/highest center, clamp to the nearest band's tau. Same `set_smoothing` slider drives both this and the L/M/H smoother.
4. **Asymmetric peak follower.** Fast attack (`tau_attack_s`, default 50 ms) on rising values, slow release (`tau_release_s`, default 60 s). Both taus are shared with the L/M/H `AutoScaler`. Implemented alloc-free as `fill(release) → np.copyto(attack, where=mask) → multiply → add`, all into preallocated buffers (`_diff`, `_alpha`, `_mask`).
5. **Spatial smear of the peak.** `gaussian_filter1d(peak_lin, σ, output=peak_smoothed, mode='reflect')` where σ is `peak_smear_oct × bins_per_octave`. Without this, a sustained single-frequency tone drives ITS bin's peak to fully self-normalize, so the tone bin reads *smaller* than its quiet neighbors. Sharing the divisor across a Gaussian neighborhood preserves local frequency contour while still flattening the long-term spectral envelope. `mode='reflect'` is self-normalising at the edges (mirrored neighbours carry the same kernel mass as the interior), so no per-bin normalisation divide is needed. Default 0.3 oct.
6. **AutoScaler core.** `tanh(max(0, smooth - noise_floor) / max(peak_smoothed, noise_floor))`. Bit-for-bit identical math to the L/M/H side, except divisor is the smeared peak.
7. **Strength blend.** `output = strength × scaled + (1 − strength) × raw_db_mapped`, where `raw_db_mapped = clamp((wire_dB − db_floor) / span, 0, 1)` with the same noise gate (in calibrated units) so silent bins flatten to 0 instead of disappearing. At `strength=0` the output is an honest dB readout; at `strength=1` it's fully equalized.

The output is always in `[0, 1]` (same range as L/M/H scaled). `process()` writes into one of two preallocated f32 output buffers and returns the just-written ref, toggling on each call; the returned ref stays valid until two further `process()` calls. `FFTWorker.run` does the same trick for the wire-format `bins_f32` (two preallocated buffers, alternated). Both refs are passed straight into `FFTStore.publish` — no copy on the publish path. The two consumers (OSC sender, WS encoder) drain the array synchronously inside their handler (via `.tolist()` and `.tobytes()` respectively) so the ref is always dropped well before the producer cycles back ~2 hops later.

**`update_*` methods** (called from the asyncio loop on Dispatcher events) all take `_lock` and recompute affected derived state: `update_smoothing` rebuilds `tau_per_bin`; `update_bands` re-anchors it; `update_autoscale` re-derives the attack/release alphas; `update_smear` recomputes σ; `update_tilt` rebuilds the per-bin tilt offsets; `reconfigure` reallocates everything (used on `n_bins` / `sr` / `f_min` / hot-switch). Hop-rate `process()` calls don't allocate — buffers (`smooth_lin`, `peak_lin`, `peak_smoothed`, `interp_db`, two `_processed_buffers`, the sentinel-interp LUT `_empty_idx / _left_idx / _right_idx / _w_left / _w_right / _left_vals / _right_vals`, plus four scratches `_lin / _diff / _alpha / _scratch` and a bool `_mask`) are preallocated.

### Wire formats

- **WS JSON**: `meta`, `snapshot` (L/M/H raw + scaled), `devices`, `presets`, `server_status`, `error`. Inbound types are listed in `Dispatcher._handlers`.
- **WS binary** (FFT): `[type=1:u8][reserved:u8][n_bins:u16 LE][float32 * n_bins LE]` — see `encode_fft_binary` / `decodeFftBinary` in `ui/src/ws.js`. **Float interpretation depends on `meta.fft_send_raw_db`**: `false` (default) → post-processed `[0..1]`; `true` → raw wire dB with `-1000` sentinels for empty log bins.
- **OSC**: `/audio/meta [sr, blocksize, n_fft_bins, low_lo, low_hi, mid_lo, mid_hi, high_lo, high_hi]` — three independent bandpasses, edges in Hz. `/audio/lmh [low, mid, high]` (scaled, per audio block). `/audio/fft [...bins]` (only when `osc.send_fft` is true and FFT is enabled). **Same `send_raw_db` flag controls FFT semantics here**: `false` → `[0..1]` post-processed (matches L/M/H semantics on OSC); `true` → raw dB (sentinels rewritten to `db_floor` so consumers see in-range values).

### Config / validation

`server/control/validate.py` is the single source of truth for value ranges (band cutoffs, tau, n_fft_bins, ws snapshot hz, autoscale params, preset name regex, device index). Both inbound WS messages (`Dispatcher`) and YAML loading (`config.load_config`) route through these validators, so invalid values from either source produce a fallback to defaults rather than crashing.

## Coding conventions specific to this codebase

- **No allocation in the callback.** Every buffer used by `AudioCallback`, `DSPWorker`, `FFTWorker`, `FFTPostProcessor`, and the ring is preallocated in `__init__` / `_allocate`. New code in those files must use in-place numpy ufuncs (`np.add(..., out=)`, `np.multiply(..., out=)`, `np.fft.rfft(..., out=)`, etc.) — never expressions that allocate. Subtle traps: `np.where(...)` does not accept `out=`, so prefer `fill + np.copyto(..., where=mask)` with a preallocated bool mask; `arr * scalar` and `arr1 + arr2` always allocate (use the ufunc with `out=`); `np.maximum(arr, scalar)` allocates unless given `out=` (repurpose a scratch buffer, e.g. `_alpha`, when its content is no longer needed); `np.square(x)` allocates a length-`x` temp (use `np.dot(x, x)` for RMS²); `bool_arr = arr > 0` allocates (use `np.greater(..., out=mask)` into a preallocated bool mask). The plan's §3.1 has the audit reasoning.
- **Hop-rate FFT output is double-buffered, not copied.** `FFTWorker._bins_f32_buffers` (×2) and `FFTPostProcessor._processed_buffers` (×2) are preallocated; each producer alternates which buffer it writes into and publishes the just-written ref directly. This holds because the two consumers (OSC `tolist()`, WS `tobytes()`) drop the ref synchronously inside their handler — well before the producer cycles back ~2 hops later. If you add a third consumer that holds the ref across an `await` or hands it to another coroutine, take a copy at that boundary or extend the buffer count.
- **Float32 audio path, float64 control state.** Ring slots, mono buffer, FFT window, FFT wire output are float32. Filter `zi` state, smoother values, autoscaler peak/scratch, post-processor internals are float64 for IIR numerical stability and accumulation precision. Don't mix without thinking about it.
- **Worker-thread reconfiguration goes through a lock or a setter**, not direct attribute assignment from the asyncio loop. `FFTWorker.reconfigure()` and `FFTPostProcessor.{reconfigure, update_*}` take `self._lock`; `ExpSmoother.set_tau()` and `AutoScaler.set_*()` are atomic single-attribute swaps that workers re-read on next iteration.
- **Drop-oldest, never block.** WS per-client outbound is a `_BoundedDropOldest` (maxsize=4); the broadcast loop and dispatcher reply path both use `put_nowait_drop_oldest`. Never await a slow client.

## UI

Plain ES modules, no build step, no framework. `ui/index.html` loads `src/main.js`, which wires WS handlers (`ws.js`) into a tiny store (`store.js`) and four canvas visualizers (`viz/`). The browser must reach the UI over `http://` (not `file://`) for ES modules to load — that's why `StaticHTTPServer` exists. When iterating on the UI, just refresh the browser; the server picks up the new files since it's serving them statically.

**Zero DSP in the UI.** The browser is a thin renderer + control panel. All signal processing happens server-side. The only computations on the UI side are pure visualization: peak-hold markers (decay-only — they fall toward the incoming signal, never reshape it), the L/M/H scene (disc + tint + sparkles indexed by `store.{low,mid,high}`), and the rolling polylines. `viz/fft_2d.js` paints whatever bins it receives from the WS binary frame, with axis labels chosen by mode (`dB` axis when `meta.fft_send_raw_db === true`, `0..1` axis otherwise). This is by design: **what you see in the FFT graph is byte-identical to the `/audio/fft` OSC payload**, so the viz can be used to tune all the IIR / smoother / autoscaler / smear knobs and trust that what's shipping to downstream consumers matches the picture.

When toggling the "raw dB" checkbox or any other server-bound control, the UI sends a WS message and reflects the *server's* state back from the next `meta` payload — the UI never holds local state that could drift from the server.

## Things not to do

- Don't add tests under `tests/` blindly — the directories exist but no fixtures or shared conftest is set up yet. Ask before scaffolding test infra.
- Don't add a build step to `ui/`. The whole point is that it's hackable static files.
