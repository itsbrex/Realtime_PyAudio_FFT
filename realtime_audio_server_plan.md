# Realtime Audio Feature Server — Implementation Plan

A localhost Python server that captures system audio (mic input, cable input, soundcard, ...), computes perceptually-tuned low/mid/high band energies and an optional FFT spectrum, and publishes both to (a) external apps over OSC/UDP and (b) a browser visualizer over WebSocket. The browser additionally sends control messages (toggle FFT, change band crossovers, switch input device, change smoothing) back to the server, all of which are persisted to a `config.yaml` file so the next launch comes up in the same state.

This plan integrates three rounds of review feedback. The most important change relative to earlier drafts: **all DSP is moved out of the audio callback**. The callback's job is now a pure SPSC ring-buffer write. This makes the real-time contract trivially satisfiable in pure Python and removes the allocation hazards inherent in `scipy.signal.sosfilt(…)`, which (verified against scipy main, `_signaltools.py::sosfilt`) computes `dtype = np.result_type(sos, x, zi)` and then `np.array(x, dtype, order='C')` — i.e. it always allocates a fresh output array, has no `out=`, and with our `x: float32` + `zi: float64` combo returns float64 buffers.

---

## 1. Goals & Non-Goals

**Goals**
- End-to-end input-to-OSC latency in the **8–15 ms** range at 48 kHz, blocksize 256. The system is **block-synchronous**, not sample-accurate — features are produced once per audio block, and downstream delivery (OSC/WebSocket/browser) adds its own scheduling slop.
- Zero audio dropouts under normal desktop load (callback never blocks, never allocates, never logs, never holds a lock).
- Tunable low/high crossover frequencies and per-band smoothing — controllable live from the browser UI.
- Optional FFT pipeline (128 log-spaced bins, default) toggleable at runtime.
- Robust device discovery + hot-switching of input device.
- All UI-controllable settings persisted to `config.yaml`; server boots into the last-used state.
- Single-process Python server; no external bridges.

**Non-Goals (v1)**
- 3D / Three.js FFT visualizer (planned for v2).
- Multi-client OSC fanout beyond a configurable list of UDP destinations.
- Recording to disk, file playback, offline batch processing.
- Pop-free filter retune (we accept a brief click on parameter change in v1).

---

## 2. High-level architecture

```
                              ┌── OSC/UDP ──→ external consumers (TouchDesigner, …)
                              │
sounddevice ──► Audio CB ─────┘                ┌──► OSC sender ──► UDP socket
(C thread)      │                              │
   memcpy only  ▼                              │
       SPSC ring buffer (raw mono float32) ────┴──► DSP worker (block-driven)
                              │                      │
                              │                      │  filter (LP/BP/HP, SOS)
                              │                      │  RMS / exp smoothing
                              │                      ▼
                              │              Latest features (low,mid,high)
                              │                      │
                              │                      └──► WS broadcaster ──► browser UI
                              │
                              └──► FFT worker (hop-driven, optional)
                                            │
                                            ├──► OSC /audio/fft
                                            └──► WS broadcaster (binary frame)

Browser UI ──► WS control channel ──► Control dispatcher ──► (filter retune,
                                                              FFT toggle,
                                                              device switch,
                                                              smoothing params,
                                                              persist config.yaml)
```

Threads doing real work:

| Thread                     | Owner               | Priority    | Responsibility                                        |
|----------------------------|---------------------|-------------|-------------------------------------------------------|
| Audio callback             | PortAudio           | RT (C)      | Mono-mix into preallocated buffer, ring write, signal |
| DSP worker                 | `threading.Thread`  | Normal      | Filter + RMS + smoothing per block; publish features  |
| FFT worker (optional)      | `threading.Thread`  | Normal      | FFT + log-bin per hop; publish spectrum               |
| Sender / WS broadcaster    | asyncio event loop  | Normal      | OSC send, WS fan-out, control dispatch                |

Plus a main thread that orchestrates startup/shutdown and runs the asyncio loop.

The cost of pulling DSP out of the callback is **one extra block hop of latency**: the audio thread signals an event and a worker wakes up to consume the new block. On Python 3.12 / macOS this measures in low hundreds of microseconds — well within the latency budget — and buys a trivially provable RT-safety story for the callback.

---

## 3. Threading & data-flow contract

The cardinal rule: **the audio callback never waits, never allocates, never logs, never sends a packet, never takes an explicit Python lock.** Every step is a memcpy or an integer increment.

A note on the GIL: because `sounddevice` invokes our Python callback via CFFI from PortAudio's C thread, the callback implicitly acquires the GIL on entry — we cannot avoid this in pure-Python land. What we *can* avoid is taking any *additional* explicit locks and any allocation that would force a GC pause while the GIL is held. The mitigation is to keep worker threads GIL-friendly: the DSP and FFT workers spend nearly all their time inside NumPy/SciPy C kernels (`sosfilt`, `rfft`, `bincount`, in-place ufuncs), all of which release the GIL, so the callback rarely contends. We avoid pure-Python loops in any worker thread for the same reason.

### 3.1 Audio callback (hot path)

Per call (blocksize = 256 samples):

1. PortAudio passes `status` (a `sounddevice.CallbackFlags` instance) on every call. We test `status.input_overflow` — set when PortAudio's input buffer was full and frames were dropped *before* this callback ran — and bump `cb_overruns` (one int store, no allocation, no log call). All other flag bits are ignored on the hot path; the flags are sticky-cleared between callbacks. This is the only producer of `cb_overruns`; the counter is read by the asyncio sender when assembling `server_status` (§3.2 counter-visibility note).
2. If stereo, mono-mix into a **preallocated** `mono_buf` using two in-place ufunc calls:
   ```python
   np.add(in_data[:, 0], in_data[:, 1], out=mono_buf)
   np.multiply(mono_buf, 0.5, out=mono_buf)
   ```
   We use this rather than `np.mean(in_data, axis=1, out=mono_buf)` because the two-ufunc form is the smallest auditable primitive — each call is unambiguously a single in-place ufunc and we don't have to reason about what the reduction wrapper does internally. (For the record: with float32 input *and* float32 `out=`, current NumPy does **not** promote to a float64 accumulator — `numpy.mean`'s docs explicitly note that float32 input keeps float32 precision unless `dtype=` says otherwise — so `np.mean` would not actually allocate a hidden float64 intermediate in our specific dtype combination. The argument for the ufunc pair is auditability and version-stability across the dep range, not a concrete allocation we observed.)

   **Channel-order caveat.** `in_data` arrives as PortAudio's interleaved frames-major buffer (shape `(blocksize, n_channels)`, C-contiguous), and `in_data[:, 0]` / `in_data[:, 1]` are **strided views** with stride = `n_channels · 4` bytes. For typical stereo input devices (built-in mic, USB stereo) channel 0 = left, channel 1 = right and a simple sum-and-halve is correct. For **aggregate devices and multichannel pro interfaces** (e.g. RME with non-LR pairing, Audio MIDI Setup–assembled aggregates) the channel order is whatever the device exposes, and naïvely summing channels 0 and 1 may pick a non-stereo pair. We sidestep this in v1 by exposing only `channels=1` and `channels=2` and letting users pick the mono channel via `sounddevice`'s `mapping=` if their stereo pair is non-default — documented in the README, not the UI.

   On performance: with strided views, an explicit reduction (`np.sum(in_data, axis=1, out=mono_buf)` followed by `np.multiply(mono_buf, 0.5, out=mono_buf)`) does a single pass over the contiguous buffer and is typically faster than two strided reads + a multiply. We've kept the two-ufunc form for auditability; if profiling at blocksize=256 ever shows the callback exceeding 50 µs we'll switch to the `np.sum(out=)` form.
3. Copy `mono_buf` into the next slot of the block-aligned SPSC slot ring with a single `np.copyto(ring.slots[wi & mask], mono_buf)`. There is no wrap branch and no partial-block handling because the ring is block-aligned (§4.2).
4. Publish in order: write the per-slot ready seq (`ring.block_seq[wi & mask] = wi + 1`), *then* the global `ring.write_idx = wi + 1`. The two integer stores happen in a single Python thread under the GIL, after the data copy is complete — this is the publish boundary consumers rely on (§4.2).
5. `dsp_event.set()` and `fft_event.set()` (cheap; no allocation). Both events are set unconditionally on every block — the FFT worker no-ops if `fft_enabled` is clear or if the backlog is below `n_blocks_per_window`, so it is safe to wake it every block. Without setting `fft_event` here, the FFT worker would only advance on its 100 ms shutdown-poll timeout, which would visibly stutter the spectrum view.

That's it. No filtering, no RMS, no smoothing, no networking. We never call into SciPy from this thread, so we never have to worry about scipy's internal allocation behavior.

**Dtype contract.** `in_data` arrives as float32 from PortAudio (we open the stream with `dtype='float32'`). `mono_buf` and the ring slots are float32. Float32 is the right choice for everything that touches the audio data path — audio is effectively 24-bit, well within float32's 7-decimal-digit precision. The single exception is filter `zi` state in §6.2, which is float64 for IIR numerical stability; that's confined to the filter object and never enters the ring buffer or the cross-thread stores.

Cost budget: **< 50 µs** per callback. The callback's CPU cost is dominated by the two ufuncs for stereo input; mono input skips them entirely.

### 3.2 DSP worker (block-driven)

A `threading.Thread` running:

```
loop:
    if not dsp_event.wait(timeout=0.1):              # periodic wakeup for shutdown
        if stop_flag.is_set(): break
        continue
    dsp_event.clear()
    if stop_flag.is_set(): break

    # Snap forward if we've fallen behind by more than the ring can hold safely;
    # increment dsp_drops by however many blocks we skipped. Otherwise consume
    # exactly one block per wake — let the next event drive the next iteration.
    wi = ring.write_idx                              # single int load under GIL
    if wi - read_block_idx > MAX_DSP_BACKLOG_BLOCKS: # MAX_DSP_BACKLOG_BLOCKS = 4 (≈21 ms @ 48k/256)
        skipped = (wi - 1) - read_block_idx
        dsp_drops += skipped
        read_block_idx = wi - 1

    if read_block_idx < wi and ring.try_read_block(read_block_idx, dsp_in):
        lo, md, hi = filter_bank.process(dsp_in)         # preallocated outputs
        rms_lo, rms_md, rms_hi = rms3(lo, md, hi)        # preallocated scratch
        smoother.update(rms_lo, rms_md, rms_hi)          # exp smoothing per band
        auto_scaler.update(smoother.values, scaled_buf)  # rolling normalize → ~[0,1] (§6.3.2)
        features_store.publish(smoother.values, scaled_buf)  # raw + scaled, mutex-guarded
        sender_event.set()                                # asyncio.Event via call_soon_threadsafe
        read_block_idx += 1
    elif read_block_idx < wi:
        dsp_drops += 1                               # try_read_block returned False ⇒ producer lapped us mid-read
        read_block_idx = max(read_block_idx + 1, wi - 1)
```

The `timeout=0.1` matters: during a device hot-switch (§7.2) the audio stream is closed and the callback stops firing, so without a timeout the worker would block on `dsp_event.wait()` forever and never observe the stop flag. With it, the worker wakes at least every 100 ms to check lifecycle state.

If the worker briefly falls behind (i.e. multiple events queued), `dsp_event.wait()` returns once and we process exactly one block, then loop. We never accumulate; the ring buffer is large enough to absorb a few blocks of slop, and we read from a tracked `read_block_idx` so we don't skip data unless the ring genuinely overruns. Overrun is logged as `dsp_drops`.

**On counter visibility.** `dsp_drops` (and `fft_drops`, `cb_overruns`) are written from worker threads and read from the asyncio sender thread when assembling `server_status`. Under the CPython GIL these int writes are atomic, but a reader may observe a value one or two increments stale. That is fine — these are diagnostics, not correctness gates — and we deliberately *do not* introduce `threading.Lock` or `atomic` machinery for them. If we ever need exact counts (e.g. a regression test), the test should drive the server to a known steady state and read the counters at quiesce, not race them live.

This thread is allowed to allocate, log, and call SciPy freely. It's a normal Python thread.

### 3.3 FFT worker (hop-driven, optional)

A separate `threading.Thread`. All arithmetic is in **block indices** (not samples), matching the slot ring in §4.2: `n_blocks_per_window = window_size // blocksize` (e.g. 4 for 1024/256) and `hop_blocks = hop // blocksize` (e.g. 2 for 512/256).

```
loop:
    fft_event.wait(timeout=0.1)          # wakeup hint from audio callback / shutdown poll
    if stop_flag.is_set(): break
    if not fft_enabled.is_set():
        continue

    wi = ring.write_idx                                # single int load under GIL

    # Drop-then-process: bound the backlog before doing any work, so a slow
    # frame can't snowball into a permanent CPU peg. MAX_BACKLOG_HOPS = 2.
    if wi - read_block_idx > n_blocks_per_window + MAX_BACKLOG_HOPS * hop_blocks:
        # Snap forward to the most recent aligned hop that still fits a full window.
        target_block = wi - n_blocks_per_window
        skipped_hops = (target_block - read_block_idx) // hop_blocks
        if skipped_hops > 0:
            fft_drops += skipped_hops
            read_block_idx += skipped_hops * hop_blocks

    # Process at most one frame per wake; rely on the next event to drive the next hop.
    if wi - read_block_idx >= n_blocks_per_window:
        if not ring.try_read_window(read_block_idx, n_blocks_per_window, window_buf):
            fft_drops += 1                              # producer lapped us mid-read
            read_block_idx = max(read_block_idx + hop_blocks, wi - n_blocks_per_window)
            continue

        np.multiply(window_buf, hann, out=window_buf)   # in-place
        np.fft.rfft(window_buf, out=spectrum)           # in-place into preallocated complex64 spectrum (rfft `out=` supported since NumPy 2.0)
        np.abs(spectrum, out=mag_buf)                   # in-place; mag_buf is float32
        np.add(mag_buf, EPS, out=mag_buf)               # in-place
        np.log10(mag_buf, out=db_buf)                   # in-place
        np.multiply(db_buf, 20.0, out=db_buf)           # in-place
        # bin_idx_valid and bin_valid_mask are precomputed in dsp/fft.py (§6.4):
        #   bin_idx_valid = bin_assign[bin_valid_mask]   # rfft bins that map to a real log bin
        #   bin_valid_mask = (bin_assign >= 0)            # masks out DC + above-Nyquist edges
        # bincount's C implementation casts weights to float64 unconditionally and
        # allocates a float64 result — there is no dtype param and no out=. We keep
        # FFTStore in float64; the WS binary encoder casts to float32 once on send.
        bins = np.bincount(bin_idx_valid,               # ALLOCATES float64 ndarray (~1 KB at N_BINS=128)
                           weights=db_buf[bin_valid_mask],
                           minlength=N_BINS)
        fft_store.publish(bins)                         # publish-by-handoff: see §4.3
        read_block_idx += hop_blocks
```

**Allocation contract on this thread.** Only one of the calls above genuinely allocates per frame, and that's intentional: the FFT worker is a normal Python thread, not the audio callback, and Python-level GC pressure here cannot cause audio dropouts.

| Call                                                 | Allocates? | Why we accept (or don't)                                                                  |
|------------------------------------------------------|------------|-------------------------------------------------------------------------------------------|
| `np.multiply(window_buf, hann, out=window_buf)`      | No         | Preallocated `window_buf`; in-place ufunc.                                                |
| `np.fft.rfft(window_buf, out=spectrum)`              | No         | `out=` parameter added in NumPy 2.0; we pin `numpy>=2.0` in `pyproject.toml`. `spectrum` is preallocated complex64, length `window_size//2 + 1`. |
| `np.abs(spectrum, out=mag_buf)`                      | No         | Preallocated `mag_buf`.                                                                   |
| `np.add(mag_buf, EPS, out=mag_buf)`                  | No         | In-place; avoids the `mag_buf + EPS` temporary that an earlier draft created.             |
| `np.log10(mag_buf, out=db_buf)`                      | No         | Preallocated `db_buf`.                                                                    |
| `np.multiply(db_buf, 20.0, out=db_buf)`              | No         | In-place; avoids the `20 * log10(...)` temporary.                                          |
| `np.bincount(bin_idx_valid, weights=…, minlength=N_BINS)` | **Yes** | `bincount`'s signature is `(x, /, weights=None, minlength=0)` — no `out=`, no dtype control. Verified against `numpy/_core/src/multiarray/compiled_base.c`: weights are cast to NPY_DOUBLE and the result is allocated NPY_DOUBLE unconditionally. So the per-frame alloc is a **~1 KB float64 ndarray** (at `N_BINS=128`), not a 512-byte float32 one. Off-thread; still trivially handled. |
| `fft_store.publish(bins)`                            | No (re-uses `bins`) | We hand the just-allocated `bincount` result over to the store directly — no extra copy. |

So the per-frame allocation budget is exactly **one NumPy array**: `bins` (`bincount`, float64, ~1 KB at N_BINS=128). At ≈ 94 frames/s this is short-lived and trivially handled by CPython's small-object pools. If profiling later shows GC pressure on the FFT path, the remaining allocation can be replaced by an in-place accumulator into a preallocated float32 buffer (`bins_out_f32[:] = 0; np.add.at(bins_out_f32, bin_idx_valid, db_buf[bin_valid_mask])`) — that also eliminates the float64→float32 cast on the WS encode path. Listed as a v1.1 item (§16).

**Dtype on the WS path.** Because `bins` is float64 and the WS binary frame is float32, the WS encoder does a single `np.asarray(bins, dtype=np.float32)` (preallocated) before writing the frame's `fft_data` payload. OSC consumers also need float32 (the python-osc `f` tag is a 32-bit float per the OSC 1.0 spec); we do the same cast in the OSC sender. Both casts are off the audio thread.

Two design points worth being explicit about:

- **Drop policy.** An earlier draft had `while ring.has_at_least_one_hop(): … process … read_pos += hop`, which silently turned the "drop frames under load" rule into "process all queued frames sequentially" — exactly the failure mode we wanted to prevent (the worker would peg CPU draining the backlog and never recover). The new shape checks the backlog *first*, snaps `read_block_idx` forward if it exceeds `n_blocks_per_window + MAX_BACKLOG_HOPS · hop_blocks`, increments `fft_drops` by however many hops were skipped, and only then processes one frame. Stable hop spacing is preserved because `read_block_idx` is always advanced by an integer multiple of `hop_blocks`.
- **One frame per wake.** Combined with the drop check above, this keeps the worker's per-iteration cost bounded and lets the next `fft_event.set()` from the audio callback drive forward progress naturally.

Toggling FFT: clearing `fft_enabled` makes the worker idle on the next iteration. We never tear the thread down between toggles.

### 3.4 Sender & WebSocket broadcaster

A single asyncio loop hosts:
- the `websockets` server (control + data fan-out),
- an **OSC sender task** waiting on a `loop`-side `asyncio.Event` set by the DSP worker via `loop.call_soon_threadsafe(event.set)`,
- a **WS broadcaster task** running on its own fixed-rate timer that polls *both* `FeatureStore` and `FFTStore` by seq — there is no event-driven hand-off into the WS path at all.

OSC and WebSocket run on **different cadences in v1**, deliberately:

| Sink            | Cadence                | Source of pacing                          | Why this rate                                                                 |
|-----------------|------------------------|-------------------------------------------|-------------------------------------------------------------------------------|
| OSC `/audio/lmh`| Full block rate (≈ 187 Hz at sr=48k/blocksize=256) | DSP worker → `sender_event` → OSC task    | OSC consumers (TouchDesigner, Max, custom apps) routinely want every block; throttling at the server would hide useful temporal detail from downstream DSP. |
| WS snapshot     | Coalesced to **60 Hz** (configurable; 60–120 Hz)   | Independent `asyncio.sleep(1/60)` loop; reads latest from `FeatureStore` by seq | The browser renders on `requestAnimationFrame`; sending faster than the display rate just means more JSON parses per redraw and more `call_soon_threadsafe` handle churn for nothing. |
| WS FFT (binary) | FFT publish rate (≈ 94 Hz at hop=512/sr=48k)       | Driven by FFT worker publishes            | Already slower than the WS snapshot path; no additional coalescing needed.   |

Concretely, the WS broadcaster is:

```python
async def ws_broadcast_loop():
    period = 1.0 / ws_snapshot_hz                # default 60 Hz; configurable
    last_feat_seq = 0
    last_fft_seq = 0
    while not stop:
        await asyncio.sleep(period)

        # Headless short-circuit: if no UI client is connected (the common case —
        # the UI is launched occasionally for tuning, not always-on), skip the
        # store reads and JSON encode entirely. Saves the per-tick encode work
        # (~60 Hz JSON snapshot + ~94 Hz binary FFT frame) when nobody is
        # listening. The loop still wakes on its own clock so that newly-
        # connected clients see fresh data on the next tick.
        if not clients:
            continue

        # L/M/H snapshot path
        seq, raw, scaled = features_store.read() # mutex held ~1 µs
        if seq != last_feat_seq:
            last_feat_seq = seq
            msg = encode_snapshot_json(seq, raw, scaled)  # scaled → low/mid/high; raw → low_raw/...
            for client in clients:
                client.outbound.put_nowait_drop_oldest(msg)

        # FFT path — same poll-by-seq pattern; only sent when FFT enabled
        if fft_enabled.is_set():
            seq, frame = fft_store.read()
            if seq != last_fft_seq and frame is not None:
                last_fft_seq = seq
                msg = encode_fft_binary(frame)   # binary frame per §6.6
                for client in clients:
                    client.outbound.put_nowait_drop_oldest(msg)
```

Key properties:
- The WS task does **not** wait on `sender_event`; it polls `FeatureStore` on its own clock. This decouples WS pacing from audio block pacing entirely, and removes per-block `call_soon_threadsafe` traffic on the WS path.
- The OSC task still wakes on `sender_event` for every block, because OSC is the one consumer that wants every block. The `call_soon_threadsafe` cost there is one allocation per block (~187 Hz) — measured, acceptable, and listed as a v1.1 optimization in §16 if profiling ever shows it.
- `seq` deduplication: if the WS clock fires before a new audio block was processed (rare at 60 Hz vs 187 Hz block rate, but possible during DSP starvation), we skip the broadcast entirely rather than re-sending the same values.
- `ws_snapshot_hz` is a config knob (`ws.snapshot_hz`, default `60`, range `[15, 240]`), exposed in `config.yaml` and in the live UI for users who want to drive 120/144 Hz displays at native rate.

---

## 4. Cross-thread data primitives

Two shared structures cross thread boundaries: the **feature store** (DSP worker → sender) and the **audio ring buffer** (audio callback → DSP & FFT workers). One additional slot holds the **latest FFT frame** (FFT worker → sender).

We do not lean on CPython GIL atomicity as a design primitive. Concurrency primitives are explicit: a tiny mutex, or an SPSC pattern, in each case.

### 4.1 Feature store (DSP worker → sender)

```python
class FeatureStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._raw    = (0.0, 0.0, 0.0)   # smoothed RMS, pre-autoscale (diagnostic)
        self._scaled = (0.0, 0.0, 0.0)   # post-autoscale, ~[0, 1] (§6.3.2)
        self._seq = 0

    def publish(self, raw, scaled):       # both length-3, copied into immutable tuples
        with self._lock:
            self._raw    = (float(raw[0]),    float(raw[1]),    float(raw[2]))
            self._scaled = (float(scaled[0]), float(scaled[1]), float(scaled[2]))
            self._seq += 1

    def read(self):
        with self._lock:
            return self._seq, self._raw, self._scaled
```

The lock is held for ~1 µs at a time. The DSP worker takes it ~187 times per second. Contention is essentially nil. We use a mutex rather than seqlock-style atomics to keep correctness obvious; if profiling ever shows contention we can swap in a double-buffer pointer flip. This is an explicit v1 design decision, not a "for now" caveat.

The store carries both raw smoothed RMS and post-autoscale values so consumers can pick: OSC `/audio/lmh` ships the scaled tuple (§6.5), the WS snapshot ships scaled in `low/mid/high` for VJ control plus raw in `low_raw/mid_raw/high_raw` for the diagnostic UI strip (§6.6). The tuple-copy in `publish` is six float boxes — cheap, and it ensures the tuple held under the lock is never aliased to a worker's mutable scratch buffer.

### 4.2 Audio ring buffer (audio callback → DSP & FFT workers)

A **block-aligned slot ring** rather than a byte-level ring. Earlier drafts used `write_pos` as the sole coordination point between the audio callback and consumers. The slot ring is more defensive than that: it makes the publish boundary first-class so we don't have to reason about NumPy's exact GIL-release pattern.

A note on what we're actually defending against. NumPy releases the GIL inside its inner copy/ufunc loops (`NPY_BEGIN_THREADS` in `PyArray_AssignArray`) and reacquires it before `np.copyto` returns. Under standard CPython, the GIL acquire/release path issues memory barriers (it's a pthread mutex on macOS), so a consumer that *acquires* the GIL after the callback has *released* it is guaranteed to observe completed writes — including data and the trailing `write_idx` store, in program order. So on today's GIL-ed CPython the older single-`write_pos` design was very likely already correct in practice.

What the slot ring actually buys us, in priority order:

- **Audit clarity.** The producer/consumer contract — "the consumer reads a stable `block_seq` around its data copy and counts a drop on mismatch" — is verifiable by reading the consumer code. The single-`write_pos` design's correctness, by contrast, requires arguing about CPython's mutex providing a release barrier and about NumPy's GIL-release pattern inside `PyArray_AssignArray`. Both arguments are correct today, but neither is a contract the project documents publicly.
- **No byte-level wrap reasoning.** A block-aligned ring has one canonical "publish boundary" per block; a byte-level ring with mid-block wraps doesn't. This is its own simplification, independent of memory ordering.
- **Free-threaded CPython forward compat (speculative).** Under free-threaded CPython (PEP 703, Phase II "supported but optional" in 3.14, *not* the default build) the GIL is no longer the synchronization mechanism and the per-slot seqlock becomes load-bearing. We do *not* treat free-threaded Python as a deployment target for v1 — `sounddevice` has no documented free-threading work as of 2026-05, and free-threaded CPython runs ~10–20% slower single-threaded — but the slot ring costs nothing extra to make this future-proof, so we take the win.

We adopt the slot ring for the first two reasons. The PEP 703 angle is a free side-benefit, not a justification.

The slot ring fixes this with two concrete changes:

1. **Block alignment.** Both consumers naturally consume in block multiples (DSP = 1 block; FFT = `window_size / blocksize` blocks). Aligning the ring on blocks turns the publish boundary into "one slot at a time" — there is no byte-level wrap mid-block to reason about.
2. **Per-slot ready flag** (`block_seq`), written *after* the slot's data and read *around* the consumer's data copy. This is a classic seqlock pattern: a consumer that observes a stable `block_seq` value before *and* after the data copy has provably read a self-consistent block.

The global `write_idx` is still there — it tells consumers "what's the latest block you could try to read?" — but it is no longer the safety boundary. The per-slot seq is.

```python
class SlotRing:
    """SPSC block-aligned ring with explicit per-slot publish.

    Producer (audio callback) writes one full block into slot[wi & mask],
    publishes the per-slot seq, then publishes the global write_idx.
    Consumers verify each read with a seqlock-style before/after check on
    block_seq, which rules out half-written or torn observations under any
    memory model.
    """

    def __init__(self, n_slots_pow2, blocksize):
        assert (n_slots_pow2 & (n_slots_pow2 - 1)) == 0
        assert n_slots_pow2 >= 8                                    # see headroom note below
        self.n = n_slots_pow2
        self.mask = n_slots_pow2 - 1
        self.blocksize = blocksize
        self.slots = np.zeros((n_slots_pow2, blocksize), dtype=np.float32)  # audio data is float32 throughout the ring path
        self.block_seq = np.zeros(n_slots_pow2, dtype=np.int64)     # 0 = never written
        self.write_idx = 0                                          # monotonic block count; published last

    # ----- producer (audio callback) -----
    def write_block(self, src):
        wi = self.write_idx
        slot = wi & self.mask
        np.copyto(self.slots[slot], src)            # 1. data write
        self.block_seq[slot] = wi + 1               # 2. per-slot publish (ready flag)
        self.write_idx = wi + 1                     # 3. global publish

    # ----- DSP consumer (one block per call) -----
    def try_read_block(self, read_idx, out):
        slot = read_idx & self.mask
        s1 = self.block_seq[slot]
        if s1 != read_idx + 1:
            return False                            # not yet written, OR already lapped
        np.copyto(out, self.slots[slot])
        return self.block_seq[slot] == s1           # False ⇒ producer lapped us mid-read

    # ----- FFT consumer (n_blocks consecutive blocks into a contiguous out) -----
    def try_read_window(self, start_block_idx, n_blocks, out):
        bs = self.blocksize
        for k in range(n_blocks):
            slot = (start_block_idx + k) & self.mask
            expected = start_block_idx + k + 1
            s1 = self.block_seq[slot]
            if s1 != expected:
                return False
            np.copyto(out[k * bs:(k + 1) * bs], self.slots[slot])
            if self.block_seq[slot] != s1:
                return False
        return True
```

**Why three writes in a specific order, not just `write_idx`.** The producer's three writes (`slot data`, `block_seq[slot]`, `write_idx`) all happen in the audio callback, which holds the GIL for the duration of the callback. Within the callback, Python statements execute in order. When the callback returns, the GIL is released — a release barrier on every reasonable interpretation of the CPython memory model — and a consumer that subsequently acquires the GIL sees all three writes in published order. The `block_seq[slot]` re-read *after* the consumer's data copy is what catches a producer lap mid-read; that check works regardless of whether the GIL is the synchronization mechanism, so the same code is correct under hypothetical free-threaded CPython without any audit changes.

**Defaults & sizing.** `n_slots = 32`, `blocksize = 256` → 32 × 256 = 8192 samples ≈ 170 ms at 48 kHz. The hard sizing constraint is `n_slots ≥ 2 · (window_size / blocksize) + safety_margin` so the producer cannot lap an in-progress FFT window read. With `n_slots = 32` and the default `window_size / blocksize = 4`, we have 28 slots of headroom — the FFT worker would have to be ≈ 150 ms behind for the producer to lap it. Anything that bad is a real-world problem (CPU starvation, GC pause), not a race condition; the seqlock check catches it and counts an `fft_drop` rather than silently corrupting a frame.

**Single producer, two consumers.** The producer is the audio callback; the two consumers are the DSP worker and the FFT worker. Each consumer maintains its own `read_block_idx`. Reads against `block_seq` and `slots` are independent across consumers — neither blocks the other. The producer never blocks, never reads `block_seq` outside its own slot, and never reads either consumer's `read_block_idx`.

**Producer-vs-consumer invariants** (enforced by the consumer-side check, not by the producer):
- "Slot freshness": `block_seq[slot] == read_block_idx + 1` ⇒ the slot holds the block we expect.
- "No lap mid-read": `block_seq[slot]` is unchanged across the data copy ⇒ the producer did not overwrite this slot during our read.

If either invariant fails, the consumer counts a drop and snaps `read_block_idx` forward to the most recent aligned position behind `write_idx`. We never block, never retry-spin, never let a torn read propagate.

Why not `collections.deque`? Per-element iteration on append; NumPy memcpy is dramatically faster for 256-sample blocks, and `deque` has no natural per-block publish boundary.
Why not `Queue`? It serializes producer/consumer with a lock per put — unacceptable in the callback.
Why not the byte-level ring from earlier drafts? It worked in practice but its safety story rested on store-store ordering between a NumPy memcpy and a separate `int` write, which is not a contract CPython provides cleanly. The slot ring makes the publish boundary first-class.

### 4.3 FFT result slot (FFT worker → sender)

A single-slot, drop-old hand-off. We define the policy once, in the FFT-store module, and never call `Queue.put_nowait` from a thread other than the sender loop:

```python
class FFTStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0

    def publish(self, frame):
        with self._lock:
            self._latest = frame   # frame is a freshly-allocated ndarray; old is dropped
            self._seq += 1

    def read(self):
        with self._lock:
            return self._seq, self._latest
```

The asyncio sender polls `read()` on a `loop_signal` set by `loop.call_soon_threadsafe`; if `seq` is unchanged it skips. This avoids the `loop.call_soon_threadsafe(queue.put_nowait, …)` pattern entirely — `put_nowait` can raise `QueueFull` from the loop thread, leaking exceptions into the event loop. The store-and-poll pattern is exception-free by construction.

---

## 5. Project layout

```
.
├── realtime_audio_server_plan.md   ← this file
├── pyproject.toml                  ← uv / pip-installable
├── config.yaml                     ← persisted UI state (created on first save)
├── server/
│   ├── __init__.py
│   ├── main.py                     ← entry point, asyncio orchestration
│   ├── config.py                   ← Config dataclass, load/save, CLI parsing
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── devices.py              ← list/select/probe input devices
│   │   ├── stream.py               ← sounddevice.InputStream lifecycle
│   │   ├── callback.py             ← the hot-path callback (memcpy + signal)
│   │   └── ringbuffer.py           ← SPSC numpy ring
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── filters.py              ← FilterBank, SOS design, swap on retune
│   │   ├── features.py             ← exp smoother, RMS, visual scaling
│   │   ├── worker.py               ← block-driven DSP worker thread
│   │   └── fft.py                  ← FFT worker thread, Hann, log binning
│   ├── io/
│   │   ├── __init__.py
│   │   ├── stores.py               ← FeatureStore, FFTStore
│   │   ├── osc_sender.py           ← python-osc client(s)
│   │   └── ws_server.py            ← websockets-based server
│   └── control/
│       ├── __init__.py
│       ├── dispatcher.py           ← decode WS control messages, mutate state
│       └── validate.py             ← input validation (crossovers, taus, …)
├── ui/
│   ├── index.html
│   ├── style.css
│   └── src/
│       ├── main.js                 ← module entry, WS bootstrap, RAF loop
│       ├── ws.js                   ← reconnect, framerate calc
│       ├── store.js                ← latest L/M/H, FFT, server fps
│       ├── controls.js             ← sliders, toggles, device picker
│       └── viz/
│           ├── lmh_lines.js        ← rolling line chart
│           ├── lmh_bars.js         ← instantaneous bars + peak hold
│           ├── lmh_scene.js        ← 3-color "audio-reactive sim"
│           └── fft_2d.js           ← FFT canvas renderer
└── tests/
    ├── conftest.py                 ← shared fixtures: synthesized signal generators, fake clock, in-memory stores
    ├── unit/
    │   ├── test_filters.py         ← SOS stability at edge cutoffs incl. near Nyquist; retune resets zi without NaNs
    │   ├── test_features.py        ← exp smoother math; AutoScaler warm-up snap, asymmetry, gate behaviour, reset()
    │   ├── test_ringbuffer.py      ← SPSC slot ring: in-order publish, seqlock detects mid-read lap, drop accounting
    │   ├── test_fft.py             ← log-bin map monotonicity, coverage of [f_min, sr/2], hop alignment
    │   ├── test_validate.py        ← every validator: accepts in-range, rejects out-of-range, finite checks, types
    │   └── test_config.py          ← YAML round-trip; unknown-key warning; invalid value falls back to default
    ├── integration/
    │   ├── test_dsp_pipeline.py    ← sine sweep 100 Hz → 1 kHz → 8 kHz: assert L/M/H envelope tracking within tol
    │   ├── test_dispatcher.py      ← inbound control messages drive correct mutations + persist; error replies on bad input
    │   ├── test_ws_protocol.py     ← in-process WS client; assert meta/snapshot/server_status framing + binary FFT layout
    │   ├── test_osc_protocol.py    ← in-process OSC receiver; assert /audio/lmh, /audio/meta, /audio/fft schemas
    │   ├── test_persistence.py     ← simulate kill mid-save: assert config.yaml is old-valid OR new-valid, never partial
    │   └── test_device_hot_switch.py  ← mock sounddevice; assert filter rebuild + autoscaler reset + ring zero on switch
    └── perf/
        ├── test_callback_alloc.py  ← tracemalloc on a 10 s captured stream; assert net allocations from callback frames are constant after warm
        ├── test_dsp_load.py        ← run DSP worker against a recorded buffer; assert avg < block_period, p95 < 1.5× block_period
        └── test_fft_load.py        ← same shape against the FFT worker; assert no fft_drops at steady state
```

---

## 6. Module-by-module design

### 6.1 `audio/devices.py`

- `list_input_devices()` → list of dicts `{index, name, hostapi, default_samplerate, max_input_channels}` filtered to those with `max_input_channels ≥ 1`.
- `signal_active_probe(devices, duration=0.2)` (renamed from "active sources"): for each device, opens a short capture, returns those whose RMS exceeded a small threshold during the probe window. **This is a heuristic** — it only proves the device produced signal during the probe, not that it always will. The UI surfaces probe results as "saw signal at 12:34:56" rather than as a property of the device.
- `default_input()` → fallback if no probe is requested.

The UI requests `{"type":"list_devices"}` over the WS control channel, gets back `{"type":"devices","items":[…]}`, and renders a dropdown. Selecting one sends `{"type":"set_device","index":N}`.

### 6.2 `dsp/filters.py`

```python
class FilterBank:
    """LP / BP / HP Butterworth bank, SOS form, stateful per-band."""

    def __init__(self, sr, low_hz, high_hz, blocksize, order=4):
        self.sr = sr
        self.order = order
        self.blocksize = blocksize
        self._design(low_hz, high_hz)
        self._init_state()

    def _design(self, low_hz, high_hz):
        # iirfilter(order, [low_hz/sr*2], btype='lowpass',  output='sos')  → sos_lp (float64)
        # iirfilter(order, [low_hz/sr*2, high_hz/sr*2], btype='bandpass', output='sos') → sos_bp
        # iirfilter(order, [high_hz/sr*2], btype='highpass', output='sos') → sos_hp
        ...

    def _init_state(self):
        # zi is float64 for IIR numerical stability — a recursive biquad accumulates
        # error every sample, and at 48 kHz over minutes float32 zi can drift audibly
        # (DC offset, slow envelope creep) for the low-cutoff band. zi is confined to
        # the FilterBank object; it never enters the ring buffer or the cross-thread
        # stores, so the rest of the data path stays float32. See §3.1 dtype contract.
        self.zi_lp = np.zeros((self.sos_lp.shape[0], 2), dtype=np.float64)
        self.zi_bp = np.zeros((self.sos_bp.shape[0], 2), dtype=np.float64)
        self.zi_hp = np.zeros((self.sos_hp.shape[0], 2), dtype=np.float64)

    def retune(self, low_hz, high_hz):
        """Called on the asyncio loop thread, NOT the DSP worker.
        Builds new SOS, then swaps under a tiny lock.
        Resets zi (brief click acceptable; rare event)."""
        ...

    def process(self, x):
        # sosfilt has no `out=`, so it allocates three new arrays per call. That's fine:
        # we are off the audio thread, and these are short-lived ~1 KB ndarrays.
        # With float32 `x` and float64 `zi`, sosfilt promotes and returns float64 outputs.
        # We do NOT cast back to float32 here — the next step in the DSP worker is RMS,
        # which collapses each band to a scalar and makes the dtype irrelevant from then on.
        out_lo, self.zi_lp = sosfilt(self.sos_lp, x, zi=self.zi_lp)
        out_md, self.zi_bp = sosfilt(self.sos_bp, x, zi=self.zi_bp)
        out_hi, self.zi_hp = sosfilt(self.sos_hp, x, zi=self.zi_hp)
        return out_lo, out_md, out_hi
```

The DSP worker — not the audio callback — owns the `FilterBank`. `retune()` runs on the asyncio loop thread; we serialize it against `process()` with a small per-bank lock (only held during the SOS pointer swap and `zi` zeroing, both nanoseconds-scale).

**Validation (see also §6.7):** before any retune we enforce:
- `MIN_HZ ≤ low_hz ≤ high_hz - MIN_GAP_HZ` (defaults: `MIN_HZ=20`, `MIN_GAP_HZ=50`)
- `high_hz ≤ 0.45 × sr` (margin from Nyquist; Butterworth band edges become unstable closer to fs/2)
- both finite and positive

A request that fails validation is rejected with a `{"type":"error","reason":"…"}` WS reply and **does not** mutate filter state.

**Why order 4:** order 2 (single biquad) gives only 12 dB/oct — bands bleed. Order 4 → 24 dB/oct, mainstream-audio standard. Cost is two SOS rows per filter, negligible.

**Why LP/BP/HP rather than 3× BP:** preserves sub-bass and air rather than rolling them off; matches plan_01 §3.

**Sample-rate discipline:** filters are designed against the actual `stream.samplerate` returned by sounddevice after open. We never hard-code 44100. On device switch we re-query `sr` and rebuild.

### 6.3 `dsp/features.py`

Two per-band stages run back-to-back in the DSP worker: an **exponential smoother** that absorbs block-rate jitter, then an **auto-scaler** that adapts to the long-term loudness of the room and emits values in a roughly `[0, 1]` band suitable for direct VJ control.

#### 6.3.1 Exponential smoother

Per-band exponential smoothing with per-band τ:
```
α_band = 1 - exp(-blocksize / (sr · τ_band))
y_band = α_band · rms_band + (1 - α_band) · y_prev_band
```
Defaults: τ_low = 150 ms, τ_mid = 60 ms, τ_high = 20 ms. This addresses the "highs feel jumpy / lows feel sluggish" pitfall.

The smoother config (`τ` per band) lives in a small dataclass updated from the dispatcher; reads are uncontended.

Smoothed band values are kept as Python floats (i.e. CPython float64 boxes), but the *information content* is float32 — RMS of a float64 filter output mapped to a single scalar carries far less precision than the underlying buffer. We don't down-cast: a `float` is what `FeatureStore` and `python-osc` and `json.dumps` all want anyway, and any explicit cast would just add a no-op.

#### 6.3.2 Rolling auto-scaler

VJ consumers want feature values in a roughly `[0, 1]` dynamic range, but raw RMS depends on whatever the room is doing right now: soft background music for an hour, then a DJ drops a beat and suddenly RMS jumps an order of magnitude. A static gain can't bridge that. Conversely, during silence the input is just sensor noise, and any aggressive normalizer would happily amplify that noise into spurious "music" output.

The auto-scaler solves both with a single asymmetric peak follower per band, plus a noise floor:

```
# Per block, per band:
α_attack  = 1 - exp(-blocksize / (sr · τ_attack))      # τ_attack  fixed at 50 ms
α_release = 1 - exp(-blocksize / (sr · τ_release))     # τ_release exposed: 5–300 s, default 60 s
α         = α_attack if value > peak else α_release
peak     += α · (value - peak)                          # asymmetric one-pole follower

denom    = max(peak, noise_floor)                       # prevent silence amplification
gated    = max(value - noise_floor, 0.0)                # soft gate: 0 at/below floor
out      = tanh(gated / denom)                          # parameter-free soft compressor
```

Why this shape:

- **τ_release is the rolling window.** A loud transient charges `peak` instantly (fast attack) and then bleeds off with time-constant τ_release. At τ_release = 60 s the follower has fallen to e⁻¹ ≈ 37 % after one minute, e⁻³ ≈ 5 % after three. That is the user-visible "window length" knob — exposed as `tau_release_s` in the UI/config.
- **Asymmetric attack/release** is what gives us robustness to outliers without giving up responsiveness. A symmetric tracker either chases every transient (no headroom for the next one) or smears slow enough that a DJ drop stays clipped for seconds. Fast attack + slow release means the follower captures the *envelope* of recent peaks; ratios above 1 happen only on a fresh peak that exceeds the current envelope.
- **`tanh` as the compressor.** Monotonic, smooth, parameter-free, asymptotes to 1, and identity-like for small inputs (`tanh(0.3) ≈ 0.29`, `tanh(1.0) ≈ 0.76`, `tanh(2.0) ≈ 0.96`). Average loudness lands around 0.4–0.7, peaks reach 0.8–0.95, brief outliers compress gracefully toward 1 without ever crossing it. We don't expose a "compression strength" knob because tanh's shape is already a good general-purpose match for music dynamics — the meaningful knobs are the time window and the noise floor.
- **Noise floor.** Single linear-RMS scalar (default `1e-3`, ≈ −60 dBFS), exposed in the UI/config. Two roles in one knob: (a) it floors the divisor (`denom = max(peak, floor)`) so during silence we divide by the floor rather than by an arbitrarily-decayed `peak`; (b) it's subtracted from the signal (clamped at zero), which is a soft gate — values at or below the floor produce 0 output, values well above produce ≈ ratio. The two halves cancel exactly when `value = peak = floor` (silence), giving `out = tanh(0) = 0`.

What the auto-scaler does **not** do:

- It does not run on the FFT path. The spectrum already has its own dB-based mapping (`fft_db_floor`/`fft_db_ceiling` in §6.4). Per-bin auto-scaling would need 128 trackers and is rarely what spectrum consumers want anyway.
- It does not expose τ_attack as a knob. Fast attack (50 ms) is what makes the follower an *envelope* rather than a smoother; making it slow defeats the design. Buried as a constant.
- It does not chain with a separate visual scaling. The auto-scaler's output **is** what we send to OSC and WS — a single, time-aware [0, 1] mapping, not a static log curve layered on top of raw RMS.

State and update cost per block: 3 doubles (`peak[low, mid, high]`) and a vectorized length-3 update — `np.where`, one `*=`, one `+=`, one `np.tanh`. Sub-microsecond on the DSP worker thread. No allocation: scratch is a preallocated length-3 array.

```python
class AutoScaler:
    """Rolling per-band normalizer: smoothed RMS → ~[0, 1] via asymmetric
    peak follower + soft noise gate + tanh compression. See §6.3.2."""

    def __init__(self, sr, blocksize, tau_release_s=60.0, noise_floor=1e-3):
        self.sr, self.blocksize = sr, blocksize
        self.noise_floor = float(noise_floor)
        self._peak = np.full(3, self.noise_floor, dtype=np.float64)
        self._scratch = np.zeros(3, dtype=np.float64)
        self._warmed = False                                # snap-to-envelope on first call (see update())
        self.set_taus(tau_attack_s=0.05, tau_release_s=tau_release_s)

    def set_taus(self, tau_attack_s, tau_release_s):
        dt = self.blocksize / self.sr
        self._a_atk = 1.0 - math.exp(-dt / max(tau_attack_s,  1e-3))
        self._a_rel = 1.0 - math.exp(-dt / max(tau_release_s, 1e-3))

    def set_noise_floor(self, floor):
        # Raise `_peak` if the user moved the gate above the current envelope; otherwise
        # `_peak` decays toward the new floor naturally over τ_release.
        self.noise_floor = max(float(floor), 0.0)
        np.maximum(self._peak, self.noise_floor, out=self._peak)

    def reset(self):
        # Called on device hot-switch (§7.2): new device may have very different gain.
        self._peak.fill(self.noise_floor)
        self._warmed = False

    def update(self, values_in, out):                     # both length-3 float64 arrays
        if not self._warmed:
            # First block (and first block after reset): snap `_peak` to the live
            # envelope. Without this, the EMA charge over one block leaves `_peak`
            # ~10× below `values_in` for ~τ_attack (≈ 50 ms ≈ 10 blocks), during which
            # `tanh(value/denom)` saturates to 1. Snap-once removes that startup glitch
            # without giving up fast-attack's outlier rejection on the steady-state path.
            np.maximum(self._peak, values_in, out=self._peak)
            self._warmed = True
        else:
            rising = values_in > self._peak
            a = np.where(rising, self._a_atk, self._a_rel, out=self._scratch)
            self._peak += a * (values_in - self._peak)
        denom = np.maximum(self._peak, self.noise_floor)
        np.subtract(values_in, self.noise_floor, out=out)
        np.maximum(out, 0.0, out=out)
        np.divide(out, denom, out=out)
        np.tanh(out, out=out)
        return out
```

`tau_release_s` and `noise_floor` are mutated from the asyncio dispatcher thread (control messages, §6.7). Both are single-float fields; we rely on the same "single-store-is-atomic-under-GIL" pattern the smoother config uses, no lock. `set_taus()` rewrites both `_a_atk` and `_a_rel`; the worker's read of those two floats one block later may briefly mix old-and-new α for a single block, which is musically inaudible (the follower's output is a continuous function of α).

**Tunable Strength.** This auto-scaler is very opinionated and could squash / hide useful signals. Therefore a single 0-1 float **auto_scaler_strength** sits in config.yaml (defaults to 1.0) that modulates between fully auto-scaled (1.0) outputs and raw, unscaled outputs (0.0). Make sure **auto_scaler_strength** is applied in the code appropriately (likely missing in current code samples).

**Warm-up.** A naive `_peak = 0` start (or even `_peak = noise_floor`) would saturate the output for ~τ_attack ≈ 50 ms (≈ 10 blocks at sr=48k/blocksize=256), because the EMA charge over one block leaves `_peak ≪ values_in`, the divisor stays small, and `tanh(value/denom) → 1`. The first-call snap in `update()` (`np.maximum(_peak, values_in)`) replaces that EMA charge with a direct copy: on block 1 we get `_peak = values_in`, `denom = max(values_in, noise_floor) = values_in`, and `out = tanh((values_in − floor) / values_in) ≈ tanh(1 − floor/value)` — bounded by `tanh(1) ≈ 0.76` for a typical first transient, not saturated. From block 2 onward the regular asymmetric EMA runs unchanged, so steady-state behaviour (fast attack rejects single-sample outliers, slow release defines the rolling window) is identical. The same snap fires after `reset()`, which §7.2 calls on device hot-switch — the new device may have very different sensitivity, so old envelope state is meaningless.

### 6.4 `dsp/fft.py`

- `WindowSize = 1024`, `Hop = 512` defaults → 50% overlap, FFT rate ≈ 94 Hz at 48 kHz.
- Pre-computed `hann = np.hanning(window_size).astype(np.float32)`.
- Log-bin map computed once (and recomputed on `set_n_fft_bins` / sr change):
  - `edges = np.logspace(log10(f_min), log10(f_max), n_bins + 1)`
  - `bin_assign[k] = log_bin_index_for_rfft_bin_k`, with `-1` for the DC/Nyquist edges we want to skip.
  - `bin_valid_mask = (bin_assign >= 0)` and `bin_idx_valid = bin_assign[bin_valid_mask]` are precomputed alongside `bin_assign` so the per-frame call is a single `np.bincount(bin_idx_valid, weights=db_buf[bin_valid_mask], minlength=n_bins)` — O(N) reduction, no Python loops.
  - Defaults: `f_min = 30 Hz`, `f_max = sr/2`.
- Output: `n_bins` float64s in dB scale (driven by `np.bincount`'s mandatory float64 output; cast to float32 once at the OSC/WS encode boundary — see §3.3).
- **dB floor / ceiling for the WS visualization mapping:** `-80 dB` floor, `0 dB` ceiling. The browser does `clamp((db + 80) / 80, 0, 1)` for visualization; the floor and ceiling are sent in the `meta` message (`fft_db_floor`, `fft_db_ceiling`) so they can be retuned from the server without a UI deploy. OSC consumers always receive raw dB and choose their own mapping.

Toggling the FFT worker is a `threading.Event`; the worker checks it inside its loop and idles otherwise. We do **not** stop/start the thread on every toggle.

Out-of-band events surfaced on the FFT path:
- `fft_drops` — incremented when the worker can't keep up and snaps `read_block_idx` forward (by however many hops it skipped), and incremented by 1 when `try_read_window` returns False because the producer lapped us mid-read. Both conditions mean "we missed an FFT frame"; we deliberately keep them in a single counter rather than splitting backlog-snap vs. mid-read-lap, since downstream operators care about "were FFT frames lost?", not which mechanism dropped them.

### 6.5 `io/osc_sender.py`

- One `python-osc` `SimpleUDPClient` per destination, pre-resolved at startup.
- Default destination: `127.0.0.1:9000`. Multiple destinations allowed via config.
- Message catalog:
  - `/audio/meta` — `[sr:i, blocksize:i, n_fft_bins:i, low_hz:f, high_hz:f]`
  - `/audio/lmh` — `[low:f, mid:f, high:f]`, **post-autoscale, ~[0, 1]** (the values VJ consumers want — see §6.3.2). Raw smoothed RMS is not exposed over OSC in v1; if a downstream consumer wants pre-autoscale values, the WS snapshot carries them as `low_raw/mid_raw/high_raw` and a `/audio/lmh_raw` channel can be added in v1.1 without breaking compatibility.
  - `/audio/fft` — `n_fft_bins` floats in dB, only sent when FFT enabled. Not auto-scaled (the FFT path has its own dB floor/ceiling mapping; see §6.4).

We send messages individually (no bundles); `python-osc` is fast enough at localhost UDP that overhead is negligible.

### 6.6 `io/ws_server.py`

- Server on `ws://127.0.0.1:8765`, started via `from websockets.asyncio.server import serve` (the modern asyncio impl). We do **not** use `websockets.legacy.server`, which has been deprecated since 14.0.
- **Headless mode (`--no-ws`).** The WS server (and the broadcast loop, and the inbound dispatcher consumer task) is gated by `ws.enabled` (config) / `--no-ws` (CLI override sets it false). When disabled, the asyncio loop hosts only the OSC sender — no socket bound on `ws.port`, no broadcast loop wakeups, no dispatcher task. This is the expected default for production-style runs where the UI is only launched occasionally to tune parameters and persist them to `config.yaml`. With `--no-ws`, runtime control mutation is unavailable for that process — change params by editing `config.yaml` and restarting, or by briefly running with WS enabled and using the UI. OSC output, FFT, autoscaler, device selection, and persistence all work identically regardless of this flag.
- Multi-client capable; each client has its own bounded outbound queue.
- Frame discrimination by WebSocket frame type:
  - **Text frames** carry JSON for control + low-rate data (snapshots, meta, devices, status).
  - **Binary frames** carry FFT spectra: `[type:u8=1, reserved:u8, n_bins:u16, fft_data:float32 × n_bins]` little-endian (4-byte header, 4-byte aligned for the float32 payload). `n_bins` is u16 so the validator's `[8, 1024]` range fits with headroom; an earlier draft had `n_bins:u8` which silently capped at 255 and would have broken the high-resolution case. Avoids the cost of JSON-parsing 128 floats ~94 times per second in the browser.
- Outbound JSON messages:
  - `{"type":"snapshot","seq":N,"low":f,"mid":f,"high":f,"low_raw":f,"mid_raw":f,"high_raw":f,"t":server_ms}` — `low/mid/high` are the auto-scaled values in `~[0, 1]` (same numbers OSC consumers see); `low_raw/mid_raw/high_raw` are the pre-autoscale smoothed RMS for the diagnostic UI strip.
  - `{"type":"meta","sr":48000,"blocksize":256,"n_fft_bins":128,"low_hz":250,"high_hz":4000,"fft_enabled":false,"fft_db_floor":-80,"fft_db_ceiling":0,"tau":{"low":0.15,"mid":0.06,"high":0.02},"autoscale":{"tau_release_s":60.0,"noise_floor":0.001},"ws_snapshot_hz":60,"device":{"index":3,"name":"…"}}` — `meta` is re-broadcast after any successful state mutation (incl. `set_ws_snapshot_hz`), so the UI's slider always reflects the authoritative server value.
  - `{"type":"devices","items":[{"index":2,"name":"…","probed_signal":true}, …]}`
  - `{"type":"server_status","cb_overruns":n,"dsp_drops":n,"fft_drops":n,"perf":{...}}` — emitted at 2 Hz; the `perf` block is detailed in §6.8.
  - `{"type":"presets","items":[{"name":"techno","saved_at":"2026-05-04T15:30:00Z"}, …]}` — full list of preset files detected in the config directory (§8). Sent in reply to `list_presets`, after a successful `save_preset`, and on every client connect alongside `meta`/`devices`/`server_status`.
  - `{"type":"error","reason":"low_hz must be < high_hz"}`
- Inbound (control) JSON, all validated before mutation:
  - `{"type":"set_fft","enabled":true}`
  - `{"type":"set_band_cutoffs","low_hz":250,"high_hz":4000}`
  - `{"type":"set_smoothing","tau":{"low":0.15,"mid":0.06,"high":0.02}}`
  - `{"type":"set_autoscale","tau_release_s":60.0,"noise_floor":0.001}` — either field optional; only the present field(s) are mutated. Slider-shaped, so the same `commit: bool` drag-aware persistence pattern as `set_band_cutoffs` / `set_smoothing` (§6.7).
  - `{"type":"list_devices","probe":true}` / `{"type":"set_device","index":N}`
  - `{"type":"set_n_fft_bins","n":128}`
  - `{"type":"set_ws_snapshot_hz","hz":120}` — change the WS snapshot loop's cadence live. Slider-shaped, so accepts the same `commit: bool` drag-aware persistence pattern as the other numeric controls (§6.7). Mutating it retunes `period = 1.0 / ws_snapshot_hz` on the next iteration of the WS broadcast loop (one float store; no buffers to rebuild). Range `[15, 240]` Hz.
  - `{"type":"list_presets"}` — enumerate `preset-*.yaml` files in the config directory (§8) and reply with `{"type":"presets",…}`.
  - `{"type":"save_preset","name":"techno"}` — snapshot the live tunable state (DSP cutoffs/τ, autoscale, FFT view settings — *not* device, OSC destinations, or WS server config; those are session/infrastructure) into `<config_dir>/preset-<name>.yaml`. Atomic write via tmp + `os.replace` (same path as `config.yaml`'s persister, §8). On success the server re-broadcasts the refreshed `presets` list. Validation: name passes through `validate_preset_name` (§6.7) — alphanumeric / dash / underscore / space, length 1–64 — to prevent path traversal and stray characters in filenames.
  - `{"type":"load_preset","name":"techno"}` — read `<config_dir>/preset-<name>.yaml`, run every field through the same validators a live control message would (§6.7), and apply each through the existing handlers so the retune / swap behavior is identical to a slider mutation. Then persist the applied state to `config.yaml` so a subsequent boot comes up in the loaded preset. The server re-broadcasts `meta` after the apply completes (existing convention — a single `meta` message is enough; we do not synthesize a separate `preset_loaded` event).
- Backpressure: each client's outbound queue is `asyncio.Queue(maxsize=4)`; on full we drop the oldest. Slow clients can never stall the broadcaster. Note: this is hand-rolled — `websockets`' built-in flow control (`max_queue`, transport high/low water marks) is *inbound only* for `max_queue` and *blocks `send()`* for outbound, neither of which gives us drop-oldest semantics. The broadcaster does `client.outbound.put_nowait_drop_oldest(msg)` and a per-client sender coroutine consumes the queue with `await client.send(msg)`. If `send()` blocks on a slow socket, the queue fills, and the broadcaster's drop-oldest logic kicks in — back-pressure stays bounded to the queue.

On every successful client connect we immediately send `meta`, `devices`, `presets`, and `server_status` so the UI is in sync without needing to poll.

### 6.7 `control/dispatcher.py` and `control/validate.py`

`validate.py` exposes pure functions:
```python
def validate_band_cutoffs(low_hz, high_hz, sr):
    if not (math.isfinite(low_hz) and math.isfinite(high_hz)):
        raise ValueError("non-finite cutoff")
    if low_hz < MIN_HZ:
        raise ValueError(f"low_hz must be ≥ {MIN_HZ}")
    if high_hz <= low_hz + MIN_GAP_HZ:
        raise ValueError(f"high_hz must be > low_hz + {MIN_GAP_HZ}")
    if high_hz > 0.45 * sr:
        raise ValueError(f"high_hz must be ≤ 0.45·sr ({0.45*sr:.0f} Hz)")

def validate_tau(tau_dict):
    for k, v in tau_dict.items():
        if k not in ("low", "mid", "high"):
            raise ValueError(f"unknown band {k!r}")
        if not (0.005 <= v <= 2.0):
            raise ValueError(f"τ[{k}] must be in [5 ms, 2 s]")

def validate_n_fft_bins(n):
    if not isinstance(n, int) or not (8 <= n <= 1024):
        raise ValueError("n_fft_bins must be an int in [8, 1024]")

def validate_autoscale(tau_release_s=None, noise_floor=None):
    # Either field may be absent (partial update); only present fields are validated.
    if tau_release_s is not None:
        if not (math.isfinite(tau_release_s) and 5.0 <= tau_release_s <= 300.0):
            raise ValueError("tau_release_s must be in [5 s, 300 s]")
    if noise_floor is not None:
        if not (math.isfinite(noise_floor) and 0.0 <= noise_floor <= 0.1):
            raise ValueError("noise_floor must be in [0, 0.1] linear RMS")

def validate_ws_snapshot_hz(hz):
    # Accept int or float; coerce to int after range check (loop uses 1.0 / hz, but
    # int avoids surprise with users sending "60.0000001" through a slider).
    if not (isinstance(hz, (int, float)) and not isinstance(hz, bool)):
        raise ValueError("hz must be numeric")
    if not math.isfinite(hz):
        raise ValueError("hz must be finite")
    if not (15 <= hz <= 240):
        raise ValueError("hz must be in [15, 240]")

_PRESET_NAME_RE = re.compile(r'^[a-zA-Z0-9_\- ]+$')

def validate_preset_name(name):
    # Path-traversal and filename-sanity guard. We construct the on-disk path as
    # `<config_dir>/preset-<name>.yaml`, so anything that lets `name` escape the
    # directory (slashes, backslashes, '..', NUL) or break the filename (control
    # chars, dots) is rejected. Spaces are allowed for readable names like
    # "club bass heavy" but get URL-encoded nowhere — we only ever write to disk.
    if not isinstance(name, str):
        raise ValueError("preset name must be a string")
    n = name.strip()
    if not (1 <= len(n) <= 64):
        raise ValueError("preset name must be 1–64 characters")
    if not _PRESET_NAME_RE.match(n):
        raise ValueError("preset name may only contain letters, digits, spaces, hyphens, underscores")
    return n   # returns the stripped form so the dispatcher writes a canonical filename
```

`dispatcher.py` is a `type → handler` registry. Handlers run on the asyncio loop thread:
- `set_band_cutoffs` → validate → debounce 50 ms → `filter_bank.retune(...)` → persist (drag-aware; see below).
- `set_fft` → set/clear `fft_enabled`; persist (immediate — discrete event).
- `set_smoothing` → validate → write new τ values to smoother config; persist (drag-aware).
- `set_autoscale` → validate → call `auto_scaler.set_taus(tau_attack_s=0.05, tau_release_s=…)` and/or `auto_scaler.set_noise_floor(…)` (only the present field(s)); persist (drag-aware).
- `set_device` → tear down + rebuild stream (see §7.2); persist (immediate — discrete event).
- `set_n_fft_bins` → validate → rebuild log-bin map atomically in FFT worker; persist (immediate).
- `set_ws_snapshot_hz` → validate → write the new rate to a single float on the WS broadcaster (read at the top of every loop iteration; no synchronization beyond the GIL); persist (drag-aware).
- `list_devices` → enumerate (and optionally probe) → reply.
- `list_presets` → list `preset-*.yaml` files in the config directory (filename glob; we do not read the bodies until a `load_preset` arrives) → reply with `{"type":"presets",…}` carrying name + mtime per entry.
- `save_preset` → validate name → snapshot the live tunable state into a YAML body (DSP, autoscale, FFT view; *not* device, OSC, or WS server config — see §8 "Preset files") → atomic tmp+rename write through the same `Persister` machinery as `config.yaml` → reply with refreshed `presets` list. Failure (disk full, permission denied) emits `{"type":"error",…}` and does **not** mutate live state — `save_preset` is read-only with respect to the running pipeline.
- `load_preset` → read file → run every field through its normal validator → for each field that validates, dispatch through the *existing* handler (e.g. `set_band_cutoffs`, `set_smoothing`, `set_autoscale`, `set_n_fft_bins`) so the retune / swap behavior is identical to a UI slider mutation → persist the applied state to `config.yaml` with `commit=true` (immediate flush). Validation failures on individual fields are logged and skipped (the rest of the preset still applies); a wholly-invalid preset replies with `{"type":"error",…}` and leaves the pipeline untouched.

All control mutations that succeed mark the live state dirty and either schedule a debounced persist or trigger an immediate one — see §8.

**Drag-aware persistence.** Slider-shaped controls (`set_band_cutoffs`, `set_smoothing`, `set_autoscale`) accept an optional `commit: bool` field on the inbound message. The UI sends `commit: false` on drag move and `commit: true` on drag release (or when no further change has arrived for 500 ms). The handler always applies the live mutation immediately (so the audio is responsive); persistence then follows one of two paths:
- `commit: false` → schedule a long-debounced save (1 s, coalescing) — covers the case where the user keeps the slider pressed indefinitely.
- `commit: true` → fire the persist immediately on the next loop tick (still atomic, still through the same `persist()` path).

This is layered on top of the unconditional 250 ms coalesce in §8: even if the UI never sends `commit: true`, the live state is still flushed within 1 s of the last change — we just avoid writing on every micro-adjustment during a drag.

Failed validation produces `{"type":"error","reason":"…"}` and **never mutates** server state. Sliders that send invalid values get snapped back by the UI on the error reply.

### 6.8 Performance instrumentation

We want both (a) before/after numbers for the v1.1 items in §16, and (b) a live UI indicator showing how much of each thread's per-block budget is consumed. v1 ships a small instrumentation layer that times each pipeline stage and reports it on the existing `server_status` channel at 2 Hz.

**What we measure** — wall-clock deltas via `time.perf_counter_ns()` on the relevant thread:

| Stream    | Where                                                                  | Cadence              | Ring |
|-----------|------------------------------------------------------------------------|----------------------|------|
| `cb_ns`   | Audio callback body                                                    | per callback (~187 Hz) | 128 |
| `dsp_ns`  | DSP worker: filter + RMS + smoother + auto_scaler + publish            | per block            | 128  |
| `fft_ns`  | FFT worker: window + rfft + abs + log + bincount + publish             | per hop (~94 Hz)     | 64   |
| `ws_ns`   | WS broadcaster: snapshot encode + per-client `put_nowait_drop_oldest`  | per tick (60 Hz)     | 32   |

Each stream owns a preallocated `np.int64` ring indexed by a counter mod ring-length: one int store per sample, no Python loop. Mean and p95 are computed at status-emission time (2 Hz), not per sample. We deliberately do **not** use EMAs — they hide tail latency, which is exactly the thing that produces audible glitches.

**Audio-callback honesty.** `time.perf_counter_ns()` on macOS resolves to `mach_absolute_time` and is fast, but it returns a Python int — values outside the small-int cache (`[-5, 256]`) are fresh PyLong allocations (~28 B). Two calls per callback × ~187 Hz ≈ 10 KB/s of short-lived allocation. This is well below CPython's gen0 GC threshold (700 allocations) so it does not provoke pauses, and it is the *only* allocation we permit inside the callback. If a future allocation audit (§15) wants the callback truly alloc-free, we can drop into `ctypes.CDLL(None).clock_gettime` writing into a preallocated `ctypes.c_uint64` — listed as a v1.1 micro-opt in §16.

**Load metric.** For each stage:
- `block_period_ns = blocksize / sr × 1e9`  (5.333 ms at 48k/256) — used for `cb`, `dsp`.
- `hop_period_ns   = hop / sr × 1e9`        (10.667 ms at hop=512/sr=48k) — used for `fft`.
- `ws_period_ns    = 1e9 / ws_snapshot_hz`  (16.67 ms at 60 Hz) — used for `ws`.
- `load_pct = avg_ns / period_ns × 100`.

This is per-thread wall-clock load, not whole-machine. 100% means a worker is exactly keeping pace with its source clock; persistent > 100% manifests as drops in the matching `*_drops` counter. A reading like "80% load with non-zero drops" indicates wasted wakeups (event set on a block that hasn't published yet) rather than CPU starvation.

**Wire format.** We extend `server_status` rather than introducing a new message type, and add a fixed 2 Hz emission timer in addition to the existing on-connect emission (§6.6) so drop counters and load are visible live:

```json
{
  "type": "server_status",
  "cb_overruns": 0, "dsp_drops": 0, "fft_drops": 0,
  "perf": {
    "block_period_ms": 5.333,
    "hop_period_ms": 10.667,
    "ws_period_ms": 16.667,
    "cb":  {"avg_ms": 0.018, "p95_ms": 0.041, "load_pct": 0.3},
    "dsp": {"avg_ms": 0.27,  "p95_ms": 0.42,  "load_pct": 5.1},
    "fft": {"avg_ms": 0.55,  "p95_ms": 0.81,  "load_pct": 5.2, "enabled": true},
    "ws":  {"avg_ms": 0.12,  "p95_ms": 0.20,  "load_pct": 0.7}
  }
}
```

When FFT is disabled, `fft.enabled` is false and the other `fft.*` fields hold the last observed values (or zeros if never enabled this session).

**Browser-side instrumentation.** The UI maintains its own perf state — the server has no visibility into render time. A length-60 ring per visualizer holds `performance.now()` deltas around each `draw()`, and the existing 60-frame RAF rolling window from §9.1 doubles as the overall frame-rate measurement. Browser load = `avg_render_ms / 16.667 × 100` (one 60 Hz frame is the budget). The browser exposes no per-process CPU API; this is wall-clock-only and fine for relative comparisons across optimization passes, not for cross-machine absolute claims.

**UI rendering.** §9.4 has the layout. A "Performance" panel:
- One row per stage (`cb`, `dsp`, `fft`, `ws`, `raf`, plus one per visualizer) with a horizontal load bar (green < 50% ≤ amber < 80% ≤ red) and `avg_ms` / `p95_ms` numerics next to it.
- The existing drop counters (`cb_overruns`, `dsp_drops`, `fft_drops`) sit beneath the bars — load bars answer "by how much margin?", the counters answer "did we actually keep up?".

**Why v1 not v1.1.** The instrumentation *is* the optimization workflow's measurement apparatus — adding it post-hoc means we have no baseline to compare v1.1 work against. The implementation is also small (one ring per stage, one ratio per emit, one extra block in the WS payload), and the only audit-relevant cost is the two `perf_counter_ns()` calls in the callback, characterised above.

---

## 7. Lifecycle

### 7.1 Startup

1. Parse CLI flags (override config file). Recognized flags: `--config <path>` (alternate config file location, §8), `--no-ws` (force-disable the WS server, §6.6), `--open` (open the bundled UI in the default browser after startup; see step 11 below).
2. Load `config.yaml` if present; merge over baked-in defaults.
3. Build empty stores (FeatureStore, FFTStore), ring buffer, smoother.
4. Resolve initial device: CLI flag > config > default-input. If config-named device is gone, fall back with a warning.
5. Open `sounddevice.InputStream(device=…, samplerate=None, blocksize=256, channels=1 or 2, dtype='float32', callback=…)`.
6. Read actual `stream.samplerate` *and* `stream.latency` and log them. PortAudio adapts our `blocksize=256` to the CoreAudio host buffer (§12); the reported latency tells us whether the host accepted our request cleanly. If `stream.latency > 2 × (blocksize / sr)` we surface a one-time WS warning ("device buffer larger than expected; see README").
7. Build `FilterBank` against the actual `stream.samplerate`.
8. Start DSP worker thread.
9. Start FFT worker thread (idling unless `fft_enabled`).
10. Start asyncio loop with OSC sender. Also start the WS server (broadcaster + inbound dispatcher) **unless** `ws.enabled` is false (set via the `--no-ws` CLI flag or the config field — see §6.6 "Headless mode"). In headless mode no socket is bound on `ws.port`, no broadcast loop runs, and no control-message dispatcher is consuming.
11. If `--open` was passed **and** the WS server is running, open the bundled UI in the default browser via `webbrowser.open('file://' + abspath('ui/index.html'))` (one-shot call; we do not track the spawned tab). With `--no-ws` (or `ws.enabled: false`), `--open` is silently ignored — there is no live data feed for the page to attach to, so opening it would just leave the user staring at a UI spinning on its WS reconnect loop.
12. Send `/audio/meta` once over OSC.

### 7.2 Device hot-switch

1. Pause sender task (last published features remain visible to consumers).
2. Stop & close stream.
3. Reset filter `zi` to zeros, call `auto_scaler.reset()` (new device may have very different sensitivity / input gain — old envelope is meaningless; the reset re-arms the first-block snap so the user sees a sane envelope on the very first post-switch block, see §6.3.2), and zero the ring buffer (`write_idx = 0`, `block_seq[:] = 0`, `slots[:] = 0`) — the new device may have a different sr, so old filter state is invalid anyway. Consumers' `read_block_idx` is also zeroed so they re-sync against the fresh stream.
4. Rebuild stream against new device.
5. Re-read `stream.samplerate`; rebuild `FilterBank` and FFT log-bin map if sr changed.
6. Start stream. Resume sender. Push fresh `/audio/meta`.
7. Persist `device` to config.yaml.

A click during switch is acceptable for v1; v1.1 can add a 50 ms gain ramp on the input side.

### 7.3 Shutdown

SIGINT/SIGTERM:
1. Stop the audio stream.
2. Set DSP and FFT stop events; join workers (with 1 s timeout).
3. Cancel asyncio tasks; close WS server.
4. Close OSC clients.
5. Persist current state to config.yaml (best-effort).

---

## 8. Persistence — `config.yaml`

config.yaml is part of the v1 spec, not deferred. It lives next to `pyproject.toml`. The path is overridable with `--config <path>`.

**Schema** (all fields optional; missing fields fall back to defaults):
```yaml
audio:
  device: { name: "BlackHole 2ch", index: 3 }   # name preferred, index advisory
  blocksize: 256
dsp:
  low_hz: 250.0
  high_hz: 4000.0
  tau: { low: 0.15, mid: 0.06, high: 0.02 }
autoscale:
  tau_release_s: 60.0   # rolling window for the per-band peak follower (§6.3.2). Range [5, 300].
  noise_floor:   0.001  # linear RMS gate; signal at/below this maps to 0 output. Range [0, 0.1].
fft:
  enabled: false
  n_bins: 128
  window_size: 1024
  hop: 512
  f_min: 30.0
osc:
  destinations:
    - { host: "127.0.0.1", port: 9000 }
  send_fft: false
ws:
  enabled: true         # set to false (or pass --no-ws on the CLI) for headless OSC-only runs (§6.6).
  host: "127.0.0.1"
  port: 8765
  snapshot_hz: 60       # WS L/M/H broadcast rate; OSC stays at full block rate. Range [15, 240].
```

**Save rules:**
- Save is requested by control-message handlers (asyncio loop thread). Handlers do not write to disk directly; they call `persist_request(commit: bool)` on a small `Persister` task that owns the file.
- Two debounce levels, picked by the handler:
  - **Drag (`commit=false`):** schedule a flush 1 s from now, replacing any earlier scheduled flush. Intent: do not write on every slider tick during an active drag.
  - **Commit (`commit=true`) and discrete events (`set_fft`, `set_device`, `set_n_fft_bins`):** schedule a flush 50 ms from now, *and* never let the existing flush slip past 250 ms total wait. Intent: persist promptly when the user means it.
- The 250 ms cap covers the long-debounce case: if a series of `commit=false` writes is followed by a `commit=true`, the next flush still fires within 250 ms even if the long timer was about to push it further out.
- Save is atomic: write to `config.yaml.tmp`, `os.replace()` to `config.yaml`. Never partial.
- Save errors are logged but do not propagate; the live server keeps running.
- On controlled shutdown we flush a final save synchronously before exiting (cancels any pending debounce timers; writes once).

**Load rules:**
- On startup we parse YAML with `yaml.safe_load`. Unknown keys are warned-on, not rejected (forward compatibility).
- Each loaded value is run through the same validators as a live control message; invalid values are dropped with a warning and replaced by defaults.
- Device matching: prefer `name`; fall back to `index`; if both fail, use system default.

This keeps config.yaml authoritative for "what's on the screen on next boot" without coupling it to runtime hot paths.

**Preset files** (`preset-<name>.yaml`, sibling files alongside `config.yaml`):

- A preset is a *named, reusable* slice of the tunable state. Saved and loaded via the UI's "save preset" / "load preset" buttons (§9.4), surfaced over WS as `save_preset` / `load_preset` / `list_presets` (§6.6).
- **Scope** — presets cover the fields a VJ user typically wants to swap mid-session: `dsp` (cutoffs, τ), `autoscale` (release, noise floor), and the `fft` view block (n_bins, window_size, hop, f_min). They deliberately *exclude* `audio.device`, `osc.destinations`, and `ws.*` — those are session/infrastructure choices, not creative ones; including them would mean loading a preset could yank your input device or change the WS port, which is never what the user means.
- **File shape** is the same YAML schema as the matching subset of `config.yaml`, plus a tiny header:
  ```yaml
  # preset-techno.yaml
  name: "techno"
  saved_at: "2026-05-04T15:30:00Z"
  dsp:        { low_hz: 250.0, high_hz: 4000.0, tau: { low: 0.15, mid: 0.06, high: 0.02 } }
  autoscale:  { tau_release_s: 60.0, noise_floor: 0.001 }
  fft:        { n_bins: 128, window_size: 1024, hop: 512, f_min: 30.0 }
  ```
  The `name` field is informational (the on-disk filename is the source of truth for matching). `saved_at` is for the UI's preset list ordering.
- **Save** reuses the same `Persister` task that owns `config.yaml`: the handler builds the YAML body in-memory and submits it as an atomic write (tmp + `os.replace`) to the preset's path. The persister's debounce machinery is *not* applied to preset saves — they are user-triggered, low-rate, and "immediate" is the correct semantic.
- **Load** is the inverse: parse with `yaml.safe_load`, run each value through its live-message validator, dispatch valid fields through the same handlers a slider would use. After all fields have been applied, persist the resulting live state to `config.yaml` with `commit=true` so a subsequent boot comes up in the loaded preset. A preset that references a field the current server doesn't recognize (forward-compat: e.g. a v1.1 preset loaded on v1) warns and ignores that field — same convention as `config.yaml` load.
- **Discovery**: the `list_presets` handler does a filename glob (`preset-*.yaml`) in the config directory; we do not read bodies until `load_preset` requests a specific name. Listing 100 presets is a directory scan, not 100 YAML parses.

---

## 9. Browser UI

### 9.1 Networking

- One WebSocket. Auto-reconnect with exponential backoff capped at 2 s.
- Server-side message rate measured client-side: ring of last 60 message timestamps, FPS = `60 / (last - first)`. Rendered as a small badge.
- UI FPS: `requestAnimationFrame` with rolling 60-frame window.

### 9.2 Render loop

Strict separation: **WebSocket handlers update the store; nothing else.** Each visualizer runs its own `requestAnimationFrame` loop, reads from the store, and applies UI-side smoothing/lerp toward the latest values. This decouples the ~94–187 Hz network rate from the 60–144 Hz display rate and prevents stutter when packets arrive in bursts (review §8).

### 9.3 Visualizers (Canvas2D, v1)

1. **L/M/H rolling lines.** Three colored polylines, last N=300 samples wide, redrawn each frame. Time axis = wall-clock seconds.
2. **L/M/H bars with peak hold.** Three vertical bars, instantaneous height + a peak-hold tick that decays at ~1.5 units/sec.
3. **L/M/H scene.** A single large rectangle. Background hue/lightness driven by `low`. Overlay alpha by `mid`. White-noise speckle (Canvas `ImageData`) gated by `high` for hi-hat sparkle.
4. **FFT 2D.** 128 bars on a log-x axis (bins are already log-spaced). Per-bin color from a 256-entry viridis-ish LUT. Optional peak-hold line. Toggle below the canvas drives the `set_fft` control message.

### 9.4 Controls panel

- Device dropdown populated from `devices` (with "probe now" button).
- FFT on/off toggle.
- Two sliders: low crossover (40 Hz–2 kHz, log-scaled) and high crossover (1 kHz–10 kHz, log-scaled), with debounced send (UI side: 50 ms during drag; release also sends).
- Three τ sliders.
- **Auto-scaler:** two sliders (sends `set_autoscale`, drag-aware like the others):
  - "Window" — `tau_release_s` in `[5 s, 300 s]`, log-scaled, default 60 s. Effective rolling window over which the peak follower forgets a transient.
  - "Noise gate" — `noise_floor` in `[0, 0.1]` linear RMS (UI shows dBFS too: `20·log10(floor)` when floor > 0), log-scaled with a `0` snap at the bottom for "off", default `0.001` (≈ −60 dBFS).
- Small live-monitor strip: per-band raw smoothed RMS bars (the `low_raw/mid_raw/high_raw` snapshot fields) shown next to the scaled bars, so the user can see what the auto-scaler is doing — useful when tuning the noise gate.
- **WS snapshot rate** slider: `ws_snapshot_hz` in `[15, 240]`, log-scaled (so 60/120/144 are evenly spaced visually), default 60. Sends `set_ws_snapshot_hz` with the same drag-aware `commit: bool` pattern as the other numeric controls. Affects the L/M/H WS broadcast rate only — OSC stays at full block rate, FFT WS frames stay on the FFT worker's clock.
- **Presets**: a small panel with two buttons and a name field:
  - **Save preset** — text input (placeholder "preset name") + a button. On click, the UI sends `{"type":"save_preset","name":<input value>}`. The button is disabled while the input is empty or fails the client-side mirror of `validate_preset_name` (so the user gets red-outline feedback on illegal characters before the round-trip). On success the input clears and the preset list refreshes from the server's `presets` reply.
  - **Load preset** — a dropdown populated from the latest `presets` message (server-authoritative; we never trust a stale client-side list) + a "Load" button. On click, the UI sends `{"type":"load_preset","name":<selected>}`. After the server applies the preset and re-broadcasts `meta`, all the existing sliders/toggles snap to the new values via the same `meta`-driven sync path the WS reconnect already uses (no special "preset loaded" UI animation in v1).
  - The dropdown shows preset names sorted by `saved_at` descending so the most recently saved preset is on top — matches the "I just saved this, now I want to switch back to it" workflow during a tuning session.
- Two badges: server fps, ui fps.
- **Performance panel** (see §6.8). One row per stage with a horizontal load bar (green < 50% ≤ amber < 80% ≤ red) and `avg_ms` / `p95_ms` numerics:
  - Server stages from `server_status.perf`: `cb`, `dsp`, `fft` (greyed when disabled), `ws`.
  - Browser stages from local `performance.now()` rings: `raf` plus one row per visualizer.
  - Drop counters (`cb_overruns / dsp_drops / fft_drops`) sit beneath the bars.

The UI snaps any slider back to last-server-confirmed value if it receives a `{"type":"error",…}` for that change.

---

## 10. Data formats — canonical schemas

```
OSC (UDP):
  /audio/meta   sr:i  blocksize:i  n_fft_bins:i  low_hz:f  high_hz:f
  /audio/lmh    low:f  mid:f  high:f       (post-autoscale, ~[0, 1] — see §6.3.2)
  /audio/fft    bin_0..bin_{N-1}:f          (only when FFT enabled and OSC fft enabled; raw dB)

WebSocket (text JSON):
  outbound: snapshot, meta, devices, presets, server_status, error
  inbound : set_fft, set_band_cutoffs, set_smoothing, set_autoscale, list_devices, set_device, set_n_fft_bins, set_ws_snapshot_hz, list_presets, save_preset, load_preset

WebSocket (binary):
  [type=1:u8][reserved:u8][n_bins:u16][fft_data:float32 × n_bins]   (little-endian, 4-byte header)
```

OSC addresses are flat (no nesting), per review §6.

---

## 11. Tradeoffs explicitly chosen

| Decision                              | Chosen                              | Alternative                         | Rationale                                                                       |
|---------------------------------------|-------------------------------------|-------------------------------------|---------------------------------------------------------------------------------|
| DSP location                          | Off-callback worker thread          | Inside callback                     | Trivially RT-safe; eliminates sosfilt allocation hazard. ~0.5 ms latency cost.  |
| OSC send location                     | Asyncio sender                      | Inside DSP worker                   | Keep network and DSP concerns separate; profilable boundary.                    |
| Snapshot synchronization              | Single mutex                        | Seqlock / GIL atomics               | Correctness is obvious; 1 µs lock holds, uncontended.                           |
| FFT result handoff                    | Polled latest-slot                  | `loop.call_soon_threadsafe(put_nowait)` | Exception-free by construction; drop-old policy in one place.               |
| Ring buffer                           | Block-aligned slot ring + per-slot seqlock | Byte-level ring with shared `write_pos` | Single-`write_pos` is very likely already correct under today's GIL. We pick the slot ring for **audit clarity** (consumer correctness is local, no GIL-semantics argument required) and **simpler wrap reasoning** (block-aligned, no mid-block torn writes). PEP 703 forward compat is a free side-benefit, not a deployment target. |
| Filter type                           | LP/BP/HP, order 4                   | 3× bandpass / order 2               | Better separation, captures sub-bass + air.                                     |
| Smoothing                             | Per-band exponential                | Sliding RMS / fixed τ               | O(1), perceptually superior asymmetry across bands.                             |
| Output normalization                  | Asymmetric peak follower (fast attack, slow release) + soft noise gate + tanh, applied per-band post-smoothing | Static `log1p` curve, sliding-window AGC, true rolling percentile | VJ consumers want stable [0, 1]; static gain can't bridge soft-music-vs-DJ-drop, true rolling stats need O(window) memory. The asymmetric follower is O(1) state, captures envelopes (not averages), and the noise floor stops silence amplification. tanh is parameter-free and good enough; v1.1 can swap in a softer knee if needed. |
| FFT scheduling                        | Fixed-hop loop                      | "Process when N samples arrived"    | Stable timing, predictable overlap.                                             |
| FFT log binning                       | Precomputed map + bincount          | Per-frame logspace + interp         | One-time cost; per-frame is a single `np.bincount` call.                        |
| FFT visual scaling                    | dB on UI                            | Linear magnitude                    | Music has 60 dB+ dynamic range; linear renders look "dead".                     |
| WS data format                        | JSON for control, binary for FFT    | Pure JSON                           | Avoids 128-float JSON parse ~94×/s in browser.                                  |
| WS update rate                        | Coalesced to 60 Hz (configurable 60–120) on a fixed-rate asyncio loop | 1:1 with audio block (~187 Hz) | Browser renders on RAF anyway; sending faster than display rate just burns JSON parses and `call_soon_threadsafe` handles. OSC remains at full block rate. |
| Filter retune validation              | Server-side, mandatory              | Trust the UI                        | UI may drift; server is authoritative; defense in depth.                        |
| Filter retune debounce                | 50 ms server-side                   | UI-side only                        | Defense in depth; cheap.                                                        |
| Sample rate                           | Device-driven                       | Force 44.1 kHz / 48 kHz             | Avoids filter drift; honors review §11.                                         |
| Channel handling                      | Mono-mix in callback w/ `out=`      | Per-channel features                | L/M/H is a perceptual summary; mono is the right abstraction.                   |
| Persistence                           | config.yaml, atomic write           | No persistence / SQLite             | YAML is human-readable, debuggable; tmp+rename is robust enough for a single user. |
| Performance instrumentation           | Per-stage `int64` timing rings + on-demand mean/p95, reported in `server_status.perf` at 2 Hz | EMA smoothing / external profiler / no instrumentation | Rings preserve tail latency that EMAs hide; built-in instrumentation gives v1.1 work a measurable baseline; the per-block cost is two `perf_counter_ns()` calls in the callback (~10 KB/s of short-lived allocation, well below GC thresholds — see §6.8). |

---

## 12. Performance / latency budget

At blocksize=256, sr=48000:

| Stage                                      | Typical time      |
|--------------------------------------------|-------------------|
| Acquire input block (PortAudio)            | 5.3 ms            |
| Callback memcpy + ring write + signal      | < 0.05 ms         |
| Event hop into DSP worker                  | 0.1–0.5 ms        |
| DSP filter (3× sosfilt + RMS + smooth)     | 0.1–0.3 ms        |
| Feature publish + sender wakeup            | 0.1–0.5 ms        |
| OSC encode + UDP loopback                  | 0.3–1 ms          |
| **Sub-total to OSC consumer**              | **~6–8 ms**       |
| WebSocket frame to browser                 | + 0.5–2 ms        |
| Browser RAF render lag                     | + 0–16 ms         |
| **Sub-total to pixels on screen**          | **~7–25 ms**      |

**The 5.3 ms input-acquire row depends on the macOS hardware buffer size.** On CoreAudio the device buffer size is system-controlled (Audio MIDI Setup → Audio Devices → Format). PortAudio adapts its `blocksize` request to the host buffer using the rule documented in `BufferingLatencyAndTimingImplementationGuidelines`: if `host_buffer = N · blocksize` (integer ratio), the callback fires N times per host tick with **no extra adapter buffer**; if the request and host don't divide cleanly, PA inserts an adapter ring buffer that adds latency. Concretely:

- **Hardware buffer 256 frames @ 48 kHz** → 5.3 ms acquire, callback fires once per host tick. Best case.
- **Hardware buffer 512 frames** → 10.7 ms hardware acquire; with `blocksize=256` PA fires the callback twice per host tick (2:1 ratio is clean), so the *first* callback in each pair sees ~5.3 ms of "fresh" data and the second sees data already 5.3 ms old. Average end-to-end OSC latency rises to ~10–13 ms.
- **Hardware buffer 1024 frames** → 21.3 ms acquire regardless of `blocksize`; we cannot beat the hardware tick.
- **Mismatched ratio (e.g. host=480, user=256)** → PA engages its adapter; expect another 5–10 ms of buffer-conversion latency. We avoid this by using power-of-two `blocksize` values.

For the latency-sensitive deployment we should: (a) document the Audio MIDI Setup expectation in the README ("set device buffer to 256 for best latency"), (b) read and log the actual stream latency reported by PortAudio at startup (`stream.latency` after open), and (c) consider exposing `blocksize=0` as an opt-in mode — it lets PA pass host blocks through directly (variable frame count per callback) and is generally lowest-latency on CoreAudio. The cost of `blocksize=0` for our DSP is real: variable block size means we lose the clean "one slot = one block" alignment in the SPSC ring (§4.2), so this is a v1.1 item, not v1.

FFT path is on a slower clock (one frame per 512 samples ≈ 10.7 ms hop); FFT visualization latency is `hop + frame_compute + transport ≈ 15–25 ms`. Acceptable for a spectrum view.

We surface per-second `cb_overruns`, `dsp_drops`, `fft_drops` on the WS status channel for diagnostics.

---

## 13. Risks & mitigations

| Risk                                                      | Mitigation                                                                |
|-----------------------------------------------------------|---------------------------------------------------------------------------|
| GC pause inside audio callback                            | Callback never allocates: pre-alloc mono buf, `np.copyto` only.           |
| Consumer observes a half-written block (torn read)        | Block-aligned slot ring with per-slot `block_seq` written *after* data; consumers verify `block_seq` before/after the data copy and count a drop on mismatch (§4.2). |
| Producer laps a slow consumer mid-window                  | Slot count sized so `n_slots ≥ 2 · (window_size / blocksize) + margin`; seqlock check catches the rare lap and snaps `read_block_idx` forward (§4.2). |
| GIL contention starving the audio callback                | Worker hot paths stay inside GIL-releasing C kernels (sosfilt, rfft, in-place ufuncs); no pure-Python loops in any worker. |
| Reduction wrapper hiding a hidden allocation (`np.mean` family) | Use the explicit `np.add` + `np.multiply` ufunc pair (§3.1). For our specific dtype combo (float32 in, float32 out) NumPy does not in fact allocate a float64 accumulator, but the ufunc pair is unambiguous on inspection and stable across the dep range. |
| Hidden allocations in scipy on hot path                   | DSP moved off the audio thread entirely; scipy allocations are tolerated. |
| `mag_buf + EPS` / `20 * log10(...)` creating per-frame temporaries in FFT worker | In-place ufunc chain (`np.add(out=)` → `np.log10(out=)` → `np.multiply(out=)`) — see §3.3. |
| `call_soon_threadsafe` allocating a Handle ~187 ×/s       | Tolerated for v1; if profiling shows asyncio-loop jitter, switch the sender to short-interval `asyncio.sleep` polling of the FeatureStore (§16). |
| DSP worker hangs on `dsp_event.wait()` during device hot-switch | `wait(timeout=0.1)` so the worker re-checks `stop_flag` and lifecycle state at least every 100 ms (§3.2). |
| FFT worker pegging CPU draining backlog instead of dropping | Backlog check *before* the per-iteration work; snap `read_block_idx` forward and bump `fft_drops` if behind by more than `n_blocks_per_window + MAX_BACKLOG_HOPS · hop_blocks` (§3.3). |
| Filter retune click                                       | Validation guards + zi reset; v1.1 can crossfade old/new outputs.         |
| Crossover near Nyquist instability                        | Reject `high_hz > 0.45·sr` at validation time.                            |
| `low_hz ≥ high_hz` on slider drag                         | Reject with `{"type":"error"}`; UI snaps slider back.                     |
| Device switch click                                       | Brief silence acceptable for v1; v1.1 add 50 ms gain ramp.                |
| FFT worker starvation under CPU load                      | Drop frames (latest-wins via FFTStore); surface `fft_drops`.              |
| Auto-scaler amplifying sensor noise during silence        | Hard floor on the divisor (`max(peak, noise_floor)`) **and** soft gate on the numerator (`max(value − noise_floor, 0)`); together they produce exactly 0 when the signal is at or below the floor (§6.3.2). Default `noise_floor = 1e-3` (≈ −60 dBFS) is a safe room-tone level; the user can raise it if their input has audible HVAC/hiss. |
| Auto-scaler reacting too slowly when audio gets much louder | Fast attack (τ_attack = 50 ms, fixed) means `peak` catches up to a new loud envelope within one or two blocks. The user-visible "slow" behavior is purely the *release* — slow release is the desired property (it's what defines the rolling window). |
| Auto-scaler reacting too slowly when audio gets much *quieter* | Release τ ≈ 60 s by default, so a sudden drop from loud music to soft music will see the auto-scaler "compress" the soft music for ~30–60 s before fully expanding. This is the trade-off; the user can shorten `tau_release_s` (down to 5 s) for fast bars/intermissions or leave it long for sets with sustained dynamic range. |
| Auto-scaler clipping legitimate transients                | tanh asymptotes to 1 but never reaches it. Brief outliers above the running envelope compress smoothly (no hard knee) — they reach 0.95+ but never exceed 1, which is the contract VJ consumers want. |
| Auto-scaler param change during playback                  | `tau_release_s` and `noise_floor` updates are single-float stores from the asyncio thread; the DSP worker may read a one-block-stale α — the follower's output is continuous in α so this is musically inaudible. No lock needed (§6.3.2). |
| FFT queue full exceptions in event loop                   | Removed by design: store-and-poll instead of `put_nowait`.                |
| WS client connects mid-stream                             | On connect, send fresh `meta`+`devices`+`server_status`.                  |
| `python-osc` latency spike                                | Pre-resolve `(host, port)`; reuse `SimpleUDPClient`; benchmark in CI.     |
| Multiple browser tabs as WS clients                       | Each gets its own bounded queue; broadcaster fans out without blocking.   |
| User picks an output-only device                          | Filter device list to `max_input_channels ≥ 1`.                           |
| Sample-rate mismatch device-claim vs reality              | Always read `stream.samplerate` after open; design filters from that.     |
| config.yaml corruption                                    | Atomic write (tmp + `os.replace`); validation on load with safe defaults. |
| "Active source" probe misleading                          | Renamed to `signal_active_probe`; UI labels it as a one-time observation. |

---

## 14. Implementation milestones

**M0 — Skeleton (½ day)**
- Repo layout (incl. `tests/{unit,integration,perf}/`), `pyproject.toml`, deps:
  - Runtime:
    - `python-osc`
    - `sounddevice`
    - `numpy>=2.0` — we use `np.fft.rfft(out=)` introduced in 2.0
    - `scipy`
    - `websockets>=14,<17` — we use `websockets.asyncio.server.serve` (the modern asyncio impl that became the default in 14.0); `websockets.legacy.server` was deprecated in 14.0 and we deliberately avoid it
    - `pyyaml`
  - Dev (under a `[project.optional-dependencies]` `dev` extra, installed via `uv sync --extra dev` or `pip install -e .[dev]`):
    - `pytest` — runs all three test directories
    - `pytest-asyncio` — for `tests/integration/test_ws_protocol.py` and `test_dispatcher.py` (asyncio loop fixtures)
- `audio/devices.py` with `list_input_devices()` printable from CLI.
- `audio/stream.py` opening a stream and printing per-block RMS to stdout.
- `tests/conftest.py` skeleton + one trivial `tests/unit/test_filters.py::test_sanity` so the harness is wired and CI-runnable from M0 onward.

**M1 — L/M/H over OSC, off-callback (1 day)**
- `audio/ringbuffer.py` SPSC, with overrun counters.
- `audio/callback.py` minimal: mono-mix `out=`, ring write, signal.
- `dsp/filters.py` FilterBank.
- `dsp/features.py` exp-smoother + AutoScaler (§6.3.2) with hardcoded defaults.
- `dsp/worker.py` DSP worker thread (smoother → auto-scaler → publish).
- `io/stores.py` FeatureStore (raw + scaled).
- `io/osc_sender.py` running in asyncio task, sending `/audio/lmh` (post-autoscale) and `/audio/meta`.
- Smoke test with `python-osc`'s built-in receiver — verify scaled values stay in `~[0, 1]` across a soft→loud→silent test recording.

**M2 — WS data + minimal UI (1 day)**
- `io/ws_server.py` broadcasting JSON snapshots, multi-client.
- `ui/index.html` + `ws.js` + `lmh_bars.js` + `lmh_lines.js`.
- Server FPS + UI FPS badges.

**M3 — FFT pipeline (1 day)**
- `dsp/fft.py` worker, log-bin map, dB scaling, FFTStore.
- Binary FFT frame on WS; `fft_2d.js` renderer.
- FFT toggle from UI.

**M4 — Controls, validation, persistence (1 day)**
- `control/validate.py` + `control/dispatcher.py` for all message types (incl. `set_autoscale`, `set_ws_snapshot_hz`, `save_preset`, `load_preset`, `list_presets`).
- Slider-driven retune (`set_band_cutoffs`) with server debounce + validation + error replies.
- Auto-scaler sliders in the UI (`tau_release_s`, `noise_floor`) + raw-vs-scaled monitor strip.
- WS snapshot-rate slider in the UI (`set_ws_snapshot_hz`); verify OSC rate is unaffected.
- Live device list + hot-switch.
- `config.py`: load on boot, atomic save on every successful control message + on shutdown.
- Preset save/load buttons in the UI; `preset-*.yaml` IO via the same atomic-write path as `config.yaml`; `presets` broadcast on connect and after every save.
- `--open` CLI flag: launches the bundled UI in the default browser when WS is enabled, silently ignored otherwise.

**M5 — Polish (1 day)**
- L/M/H scene visualizer (color rectangle + noise).
- Peak-hold on bars and FFT.
- Performance instrumentation (§6.8): per-stage timing rings on the server, `perf` block on `server_status` at 2 Hz, browser-side `performance.now()` rings, Performance panel + drop counters in the UI.
- README with run instructions and OSC schema.

Total: ~5 days of focused work. Each milestone is independently runnable.

---

## 15. Testing strategy

The `tests/` tree from §5 is organised by category; each bullet below maps to a file there.

- **Unit** (`tests/unit/`): filter design produces stable SOS at edge frequencies incl. near Nyquist (`test_filters.py`); every validator accepts in-range and rejects out-of-range/non-finite/wrong-type inputs (`test_validate.py`); SPSC slot ring publishes in order and the seqlock detects a mid-read lap (`test_ringbuffer.py`); log-bin map is monotonic and covers `[f_min, sr/2]` (`test_fft.py`); AutoScaler warm-up snap bounds the first transient at `tanh(1)` and `reset()` re-arms the snap (`test_features.py`); YAML round-trips with unknown keys warned-not-rejected and invalid values falling back to defaults (`test_config.py`).
- **Integration** (`tests/integration/`): synthesized sine sweep 100 Hz → 1 kHz → 8 kHz through the DSP path with envelope-tolerance assertions (`test_dsp_pipeline.py`); each inbound control message drives the right mutation and persists, with `error` replies on bad input (`test_dispatcher.py`); in-process WS client validates JSON framing for `meta`/`snapshot`/`server_status` and the binary FFT layout (`test_ws_protocol.py`); in-process OSC receiver validates the `/audio/{meta,lmh,fft}` schemas (`test_osc_protocol.py`); device hot-switch rebuilds filters, calls `auto_scaler.reset()`, and zeros the ring (`test_device_hot_switch.py`); kill -9 mid-save leaves `config.yaml` either old-valid or new-valid, never partial (`test_persistence.py`).
- **Perf / allocation audit** (`tests/perf/`): `tracemalloc` harness around a 10 s capture asserts net callback-frame allocations are constant after warm (`test_callback_alloc.py`); DSP worker against a recorded buffer asserts avg time < block_period and p95 < 1.5× block_period (`test_dsp_load.py`); FFT worker stays within its hop budget at steady state with zero `fft_drops` (`test_fft_load.py`).
- **Manual realtime:** play music; eyeball UI; on macOS use BlackHole as an input device. Not a CI gate — listed here so it doesn't get forgotten before release.
- **Latency:** clap test (mic → callback → OSC echo on speaker) — should sound "tight" (<20 ms perceived). Manual; not in `tests/`.

---

## 16. Open questions for v1.1+

- 3D FFT landscape — Three.js, scrolling history texture.
- **Onset detection + tempo estimation** as new feature streams alongside L/M/H. Plumbed off the FFT path — magnitudes are already computed there — so the per-hop cost is a single subtract / clip / sum over the existing `mag_buf`. Pipeline:
  - Per-hop **spectral flux** = `sum(max(0, |X[n,k]| − |X[n−1,k]|))`. Half-wave-rectified L1 difference between successive magnitude spectra is the gold-standard onset feature; far more reliable than a low-band envelope, because it triggers on energy *increases* anywhere in the spectrum (kicks, snares, claps, vocal entries).
  - **Per-band variants**: restrict the flux sum to bin ranges `[0, low_hz]`, `[low_hz, high_hz]`, `[high_hz, sr/2]` for kick / mid / hat triggers. Three independent onset streams for the price of three slices over the same magnitude buffer.
  - **Adaptive threshold**: rolling **median** (not mean — single transients shouldn't shift the bar) of recent flux values over ~1 s, multiplied by a user-configurable factor (default 1.5). The median is a cheap O(W log W) sort over a small ring (~94 samples at 1 s / hop) every hop, or O(1) amortized via a streaming median-of-medians if profiling demands.
  - **Peak picker**: a hop is an onset if (a) flux > adaptive threshold, (b) it's a local max in a ±1-hop neighborhood, (c) no onset fired in the last `refractory_ms` (default 80 ms; floor at 50 ms to avoid double-triggering one hit).
  - **Outputs**: continuous `onset_strength = tanh(flux / threshold)` in `~[0, 1]` (same family as the auto-scaler's compression), plus discrete `onset` events. Both surface over OSC (`/audio/onset_strength` per hop, `/audio/onset` per event with `band:s strength:f`) and WS (snapshot fields plus a discrete `{"type":"onset","band":"low|mid|high|full","strength":f,"t":server_ms}` event channel — the discrete channel is push-on-fire, *not* polled by the broadcaster, so consumers see every onset rather than the most-recent one per WS tick).
  - **Tempo estimate** as a separate, low-rate path: autocorrelation of the `onset_strength` envelope buffered over ~6 s, peak-picked in the 60–200 BPM lag range, smoothed with a one-pole IIR. Updated at **1 Hz**, not per hop — the autocorrelation is a few thousand multiplies and isn't worth doing every frame, and BPM doesn't change that fast in real music. Output: `bpm:f, bpm_confidence:f` on `/audio/bpm` (OSC) and in the WS snapshot. Confidence = peak height / median of the autocorr surface, clipped to [0, 1].
  - All thresholds, refractories, multipliers, and the per-band routing live under an `onset:` block in `config.yaml` and are reachable from the UI controls panel; per-band onset routing produces three independent OSC trigger streams that VJ consumers can wire to kick/snare/hat visuals directly without re-deriving onsets downstream.
- Per-band noise floor in the auto-scaler. v1 uses a single `noise_floor` shared across L/M/H; in practice rooms often have HVAC noise concentrated in lows and hiss in highs, so per-band floors would let the user gate each band tightly. Cheap to add: 3 floats instead of 1, identical UI shape (three thin sliders, or "advanced" disclosure).
- `/audio/lmh_raw` OSC channel for consumers that want pre-autoscale RMS alongside the scaled values (e.g., for their own normalization pipeline or for logging). v1 only exposes raw via the WS snapshot.
- Compression-strength knob on the auto-scaler (currently fixed at tanh's natural shape). If users find the average-around-0.7 mapping too "compressed" or too "linear", swap tanh for a parameterized soft-knee (e.g., `tanh(k·x)/tanh(k)` with `k` exposed).
- Multiple OSC destinations editable from UI (currently config.yaml-only, hot-reloadable).
- True lock-free seqlock for the FeatureStore if profiling shows lock contention.
- Pop-free filter retune via short crossfade between old/new filter outputs.
- 50 ms gain ramp around device hot-switch.
- Numba- or C-accelerated in-place biquad cascade for the DSP worker, if SciPy `sosfilt` becomes a bottleneck.
- Replace `loop.call_soon_threadsafe(sender_event.set)` on the **OSC** path (~187 calls/s, each allocates a `Handle` and takes the loop's internal lock) with a sender coroutine that does `await asyncio.sleep(0.005)` and polls `FeatureStore.read()` by `seq`. This trades a tiny extra latency floor for zero asyncio-side allocation per audio block. (The WS path already polls on its own clock — see §3.4.)
- Eliminate the remaining FFT-worker per-frame allocation (`np.bincount`) with an in-place log-bin accumulator (`bins_out[:] = 0; np.add.at(bins_out, bin_idx_valid, db_buf[bin_valid_mask])`). At ~512 B per frame / ≈ 94 frames/s this is well within CPython's small-object pools — `np.add.at` is slower per-call than `bincount` (no C fast path for the unbuffered scatter), so this is only worth doing if profiling shows actual GC pressure. (`np.fft.rfft` is no longer in this list — `out=` was added in NumPy 2.0 and we use it in v1.)
- Make the audio callback truly alloc-free by replacing `time.perf_counter_ns()` (which allocates two PyLongs per call) with a `ctypes.CDLL(None).clock_gettime(CLOCK_MONOTONIC, byref(ts))` into a preallocated `ctypes.c_uint64`. ~10 KB/s of small-int churn vanishes. Only worth doing if §15's allocation audit flags it; the current cost is well below CPython's gen0 GC threshold. See §6.8.
