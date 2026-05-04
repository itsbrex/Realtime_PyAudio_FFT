# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The design is documented in detail in `realtime_audio_server_plan.md`; if behavior in code seems surprising, the plan is the authoritative source for *why*.

```
server/      # Python audio-server package (entry point: server.main:main)
  audio/     # PortAudio callback, ring buffer, stream lifecycle, device probing
  dsp/       # FilterBank, ExpSmoother, AutoScaler, FFTWorker, DSPWorker
  control/   # WS message dispatcher + pure validators
  io/        # WSServer, OSC sender task, FeatureStore/FFTStore, static HTTP
  config.py  # Dataclass schema, YAML load, debounced atomic Persister
  main.py    # App orchestrator: builds and wires everything together
ui/          # Vanilla-JS browser UI (ES modules, no build step)
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
       → ExpSmoother            → FFTStore.publish()
       → AutoScaler                    │
       → FeatureStore.publish()        │
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

### App is the orchestrator

`server.main.App` owns every long-lived object: the stream, ring, workers, stores, sender, WS server, persister, perf rings, and the `Config`. Cross-thread communication uses three patterns and only three:

1. **SlotRing**: SPSC seqlock for audio data (callback → DSP/FFT workers).
2. **threading.Event**: one-shot wakeups (`dsp_event`, `fft_event`, `fft_enabled`, `stop_flag`).
3. **FeatureStore / FFTStore**: lock-guarded publish/read (workers → asyncio sender / WS broadcaster).

When the DSP or FFT worker publishes, it calls `App._signal_dsp_published`, which uses `loop.call_soon_threadsafe(sender_event.set)` to wake the OSC sender task. This is the only cross-thread asyncio interaction — never `asyncio.run_coroutine_threadsafe` from a worker.

### What runs where

- **Mutations to `App.cfg`, `filter_bank`, `smoother`, `auto_scaler`, `fft_worker` config**: only on the asyncio loop, from `Dispatcher` handlers. Workers read these object attributes; mutators replace or call `set_*` methods.
- **Filter retunes**: `App.schedule_filter_retune` debounces 50ms (collapses slider drags into one `FilterBank.retune` call).
- **Device hot-switch**: `App.hot_switch_device` (async, locked) — tears down the stream, resets ring/filter/scaler/smoother/FFT alignment, opens a fresh stream (sample rate may change), rebuilds DSP for the new sr, restarts. Workers stay alive throughout (they idle on event timeouts while the stream is down).
- **Persistence**: every Dispatcher handler calls `app.persister.request(commit=...)`. `commit=False` is the drag path (1s debounce, capped at 250ms from first dirty). `commit=True` is for discrete events (50ms). Atomic write via tempfile + `os.replace`.

### Why FFT and DSP are separate workers

DSP runs once per audio block (256 samples ≈ 5.3ms at 48k). FFT runs once per `hop` samples (default 512) over a `window_size` window (default 1024). Decoupling them lets FFT skip work cleanly when disabled (`fft_enabled.clear()` → worker no-ops on its event tick) without affecting DSP latency.

### Wire formats

- **WS JSON**: `meta`, `snapshot` (L/M/H raw + scaled), `devices`, `presets`, `server_status`, `error`. Inbound types are listed in `Dispatcher._handlers`.
- **WS binary** (FFT): `[type=1:u8][reserved:u8][n_bins:u16 LE][float32 * n_bins LE]` — see `encode_fft_binary` / `decodeFftBinary` in `ui/src/ws.js`.
- **OSC**: `/audio/meta [sr, blocksize, n_fft_bins, low_lo, low_hi, mid_lo, mid_hi, high_lo, high_hi]` — three independent bandpasses, edges in Hz. `/audio/lmh [low, mid, high]` (scaled, per audio block). `/audio/fft [...bins]` (only when `osc.send_fft` is true and FFT is enabled).

### Config / validation

`server/control/validate.py` is the single source of truth for value ranges (band cutoffs, tau, n_fft_bins, ws snapshot hz, autoscale params, preset name regex, device index). Both inbound WS messages (`Dispatcher`) and YAML loading (`config.load_config`) route through these validators, so invalid values from either source produce a fallback to defaults rather than crashing.

## Coding conventions specific to this codebase

- **No allocation in the callback.** Every buffer used by `AudioCallback`, `DSPWorker`, `FFTWorker`, and the ring is preallocated in `__init__`. New code in those files must use in-place numpy ufuncs (`np.add(..., out=)`, `np.multiply(..., out=)`, `np.fft.rfft(..., out=)`, etc.) — never expressions that allocate. The plan's §3.1 has the audit reasoning.
- **Float32 audio path, float64 control state.** Ring slots, mono buffer, FFT window are float32. Filter `zi` state, smoother values, autoscaler peak/scratch are float64 for IIR numerical stability and accumulation precision. Don't mix without thinking about it.
- **Worker-thread reconfiguration goes through a lock or a setter**, not direct attribute assignment from the asyncio loop. `FFTWorker.reconfigure()` uses `self._lock`; `ExpSmoother.set_tau()` and `AutoScaler.set_*()` are atomic single-attribute swaps that workers re-read on next iteration.
- **Drop-oldest, never block.** WS per-client outbound is a `_BoundedDropOldest` (maxsize=4); the broadcast loop and dispatcher reply path both use `put_nowait_drop_oldest`. Never await a slow client.

## UI

Plain ES modules, no build step, no framework. `ui/index.html` loads `src/main.js`, which wires WS handlers (`ws.js`) into a tiny store (`store.js`) and four canvas visualizers (`viz/`). The browser must reach the UI over `http://` (not `file://`) for ES modules to load — that's why `StaticHTTPServer` exists. When iterating on the UI, just refresh the browser; the server picks up the new files since it's serving them statically.

## Things not to do

- Don't add tests under `tests/` blindly — the directories exist but no fixtures or shared conftest is set up yet. Ask before scaffolding test infra.
- Don't add a build step to `ui/`. The whole point is that it's hackable static files.
