"""Beat onset detector + rolling BPM estimator on the low-band envelope.

Runs once per audio block inside DSPWorker, fed the FULLY POST-PROCESSED
`low` value (smoother → auto-scaler → strength blend → master gain — i.e.
the same `low` signal sent on `/audio/lmh` and rendered on the bars/lines
viz). This means every UI knob that shapes the visual `low` track also
shapes what the tracker sees: bandpass edges, smoothing tau, auto-scaler
strength, noise gate. The intended workflow is "dial the `low` signal in
until it pulses cleanly on each kick" — the tracker rides that exact
signal and there is no second pipeline to tune.

Pure Python scalar math on the hot path (allocation only inside
`_on_fire`, bounded at < 4 Hz by the refractory period).

Algorithm (industry-standard real-time onset detection):

1. Two one-pole envelopes of the input:
     fast (~12 ms) — removes per-block jitter
     slow (~300 ms) — local DC reference / "what 'normal' sounds like"
2. Novelty function: half-wave rectified flux, `nov = max(0, fast - slow)`.
   Equivalent to a band-pass on the envelope: positive only when the signal
   is rising faster than the slow envelope can track. This is the classic
   kick-detector shape that powers most real-time beat trackers.
3. Adaptive threshold: a slow EMA (~1 s) of `nov` itself, multiplied by
   factors gives Schmitt high/low. Plus an absolute floor so silence-noise
   can never trigger.
4. Schmitt trigger with refractory period:
     - `idle` state: fire (and arm) when `nov > thr_high` AND time-since-
       last-fire > `MIN_IOI_S` (refractory, default 250 ms = 240 BPM ceiling).
     - `armed` state: drop back to idle when `nov < thr_low` (hysteresis
       prevents a single beat re-triggering on noise during decay).
   This gives exactly one `1` per onset, surrounded by `0`s.
5. BPM: median of recent inter-onset intervals (IOIs), with outlier
   rejection (drop IOIs outside 0.5×–2× the running median to discard
   missed/extra beats), octave-folded into [BPM_MIN, BPM_MAX], smoothed
   via a per-onset EMA so it converges in ~3 beats.
6. Silence handling: if no onset for > BPM_DECAY_AFTER_S, the BPM estimate
   resets to 0 and the IOI history is cleared. Avoids reporting a stale
   tempo from the last song into the gap before the next one.
"""
from __future__ import annotations

import math


class BeatTracker:
    # ---------- Tunable params (mutable; see set_params) ----------

    # Schmitt high threshold = K_HIGH × slow novelty EMA. Higher → stricter
    # (fewer triggers, only the strongest onsets fire). Lower → more sensitive.
    DEFAULT_K_HIGH = 1.8

    # Schmitt release multiplier; armed → idle when nov < K_LOW × thr_avg.
    # Held constant — exposing both endpoints is over-tuning.
    K_LOW = 0.6

    # Refractory: shortest allowed inter-onset interval. 0.25 s ≈ 240 BPM
    # upper detection limit (well above any real musical tempo).
    DEFAULT_MIN_IOI_S = 0.25

    # Slow envelope time constant. Determines what counts as a "transient":
    # smaller → only very fast rises trigger; larger → slower swells qualify.
    DEFAULT_SLOW_TAU_S = 0.300

    # Fast envelope time constant — fixed; just removes per-block jitter.
    FAST_TAU_S = 0.012

    # Adaptive-threshold EMA tau — how quickly the trigger floor follows
    # changing dynamic content. Held constant.
    THR_TAU_S = 1.000

    # Absolute silence floor in input units. The post-processed `low` is
    # nominally [0, 1] (master_gain may push above 1), so 0.02 means "any
    # novelty smaller than 2% of full scale is below the noise we trust" —
    # safety net against the autoscaler amplifying near-silence into
    # spurious onsets when the user has set strength<1 and the band is
    # genuinely quiet. Tune the UI noise gate to push true silence to 0
    # before this floor matters.
    ABS_FLOOR = 0.02

    # BPM clamp range (octave-fold any out-of-range estimate into this).
    BPM_MIN = 60.0
    BPM_MAX = 180.0

    # Sliding-window size for IOI median.
    BPM_RING = 12

    # If no beat for this long, BPM is no longer trusted — zero it out.
    BPM_DECAY_AFTER_S = 5.0

    # Per-beat EMA factor for the smoothed BPM. 0.3 → ~3-beat settling.
    BPM_SMOOTH_ALPHA = 0.3

    def __init__(self, sr: float, blocksize: int,
                 sensitivity: float | None = None,
                 refractory_s: float | None = None,
                 slow_tau_s: float | None = None):
        # Tunables — accept overrides at construction and persist them
        # through reconfigure (alpha must be re-derived from dt + tau).
        self.k_high = float(sensitivity) if sensitivity is not None else self.DEFAULT_K_HIGH
        self.min_ioi_s = float(refractory_s) if refractory_s is not None else self.DEFAULT_MIN_IOI_S
        self.slow_tau_s = float(slow_tau_s) if slow_tau_s is not None else self.DEFAULT_SLOW_TAU_S
        self._configure(sr, blocksize)
        self.reset()

    def _configure(self, sr: float, blocksize: int) -> None:
        self.dt = float(blocksize) / max(float(sr), 1.0)
        self.alpha_fast = 1.0 - math.exp(-self.dt / max(self.FAST_TAU_S, 1e-3))
        self.alpha_slow = 1.0 - math.exp(-self.dt / max(self.slow_tau_s, 1e-3))
        self.alpha_thr = 1.0 - math.exp(-self.dt / max(self.THR_TAU_S, 1e-3))

    def reconfigure(self, sr: float, blocksize: int) -> None:
        """Called from the asyncio loop on device hot-switch (sr changes).
        Worker is briefly stalled by the orchestrator before this runs."""
        self._configure(sr, blocksize)

    def set_params(self, sensitivity: float | None = None,
                   refractory_s: float | None = None,
                   slow_tau_s: float | None = None) -> None:
        """Atomic single-attribute swaps from the asyncio loop. The DSP
        worker re-reads these on the next iteration; no lock needed."""
        if sensitivity is not None:
            self.k_high = float(sensitivity)
        if refractory_s is not None:
            self.min_ioi_s = float(refractory_s)
        if slow_tau_s is not None:
            self.slow_tau_s = float(slow_tau_s)
            # Tau drives alpha_slow — recompute. dt is already known.
            self.alpha_slow = 1.0 - math.exp(-self.dt / max(self.slow_tau_s, 1e-3))

    def reset(self) -> None:
        self.fast = 0.0
        self.slow = 0.0
        self.thr_avg = 0.0
        self.armed = False
        self.t_now = 0.0
        self.last_fire_t = -1e9
        self.onset_times: list[float] = []
        self.bpm_smoothed = 0.0
        self.beat_count = 0  # monotonic — used by WS broadcaster to
                             # detect missed onsets between snapshots

    def update(self, rms_lo: float) -> tuple[int, float]:
        """One block. Returns (beat ∈ {0,1}, bpm float).

        `beat == 1` for exactly one block per onset; rest are 0. `bpm` is
        the slowly-smoothed estimate (0 during silence or before enough
        onsets have accumulated).
        """
        self.t_now += self.dt

        # Envelopes
        self.fast += self.alpha_fast * (rms_lo - self.fast)
        self.slow += self.alpha_slow * (rms_lo - self.slow)
        nov = self.fast - self.slow
        if nov < 0.0:
            nov = 0.0

        # Adaptive threshold from slow EMA of novelty.
        self.thr_avg += self.alpha_thr * (nov - self.thr_avg)
        thr_high = self.thr_avg * self.k_high
        if thr_high < self.ABS_FLOOR:
            thr_high = self.ABS_FLOOR
        thr_low = self.thr_avg * self.K_LOW
        floor_lo = self.ABS_FLOOR * 0.5
        if thr_low < floor_lo:
            thr_low = floor_lo

        beat = 0
        if not self.armed:
            if nov > thr_high and (self.t_now - self.last_fire_t) > self.min_ioi_s:
                beat = 1
                self.armed = True
                self._on_fire()
        else:
            if nov < thr_low:
                self.armed = False

        # Silence handling — wipe stale BPM after long quiet.
        if (self.t_now - self.last_fire_t) > self.BPM_DECAY_AFTER_S:
            if self.bpm_smoothed != 0.0 or self.onset_times:
                self.bpm_smoothed = 0.0
                self.onset_times.clear()

        return beat, self.bpm_smoothed

    def _on_fire(self) -> None:
        t = self.t_now
        self.last_fire_t = t
        self.beat_count += 1

        ot = self.onset_times
        ot.append(t)
        if len(ot) > self.BPM_RING:
            ot.pop(0)
        if len(ot) < 4:
            return  # need a few onsets before estimating

        # IOIs from the buffer.
        n = len(ot)
        iois = sorted(ot[i + 1] - ot[i] for i in range(n - 1))
        if not iois:
            return
        m = iois[len(iois) // 2]
        if m <= 0.0:
            return

        # Outlier rejection: keep IOIs within 0.5×–2× the running median.
        # Discards missed/extra beats and double-triggered onsets that
        # slipped through the refractory.
        kept = [v for v in iois if 0.5 * m < v < 2.0 * m]
        if len(kept) < 2:
            return
        med = kept[len(kept) // 2]  # already sorted (subset of sorted list)
        if med <= 0.0:
            return

        bpm = 60.0 / med
        # Octave-fold into the musical range.
        while bpm < self.BPM_MIN:
            bpm *= 2.0
        while bpm > self.BPM_MAX:
            bpm *= 0.5

        if self.bpm_smoothed <= 0.0:
            self.bpm_smoothed = bpm
        else:
            self.bpm_smoothed += self.BPM_SMOOTH_ALPHA * (bpm - self.bpm_smoothed)
