"""
P5.3 -- circuit breaker state machine for high-batch degradation of
speculative decoding.

Scope: this module is PURE control logic -- it decides, given a stream of
observed "current batch size" readings, whether speculative decoding
should be active or disabled this round. It does not run any model and
has no real cost of its own to measure (that's
scripts/p5_circuit_breaker_switch_cost.py's job -- see its docstring for
the real, measured cost of the transition this module decides to take).

Why hysteresis + a minimum dwell time, not just a single threshold:

  A single threshold (disable when batch >= T, re-enable when batch < T)
  flaps every time the batch size hovers near T -- and every re-enable
  pays the real rebuild cost measured in
  scripts/p5_circuit_breaker_switch_cost.py (14.98ms-31.55ms in our
  measurements). Flapping near the boundary would mean paying that cost
  repeatedly for no benefit.

  Two independent guards against this, matching classic circuit-breaker
  design (and directly addressing project_plan_v9.md's "avoid DSD-style
  reactivation difficulty" requirement, which is really two different
  failure modes):
    1. Hysteresis gap (enable_threshold < disable_threshold): once
       disabled, batch must drop meaningfully below the disable point,
       not just barely under it, before re-enabling is even considered.
    2. Minimum dwell time (min_dwell_steps): once disabled, don't even
       LOOK at re-enabling until this many steps have passed, regardless
       of what the batch signal does in the meantime. This is what
       prevents "disabled for one step by a momentary spike, immediately
       re-enabled next step" -- a real DSD-style failure mode where the
       system doesn't have a clear answer for how long to stay off.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Mode(Enum):
    SPECULATIVE = "speculative"
    DISABLED = "disabled"


@dataclass
class CircuitBreakerConfig:
    disable_threshold: int
    """Batch size at or above which speculative decoding is disabled."""

    enable_threshold: int
    """Batch size at or below which re-enabling becomes possible. Must be
    <= disable_threshold; the gap between them is the hysteresis band."""

    min_dwell_steps: int
    """Minimum number of steps to remain DISABLED before the breaker will
    even consider re-enabling, regardless of the batch signal."""

    probe_interval_steps: int = 1
    """Once past min_dwell_steps, how often (in steps) to re-check the
    signal. 1 means check every step after the dwell period."""

    def __post_init__(self):
        if self.enable_threshold > self.disable_threshold:
            raise ValueError(
                f"enable_threshold ({self.enable_threshold}) must be <= "
                f"disable_threshold ({self.disable_threshold}) -- this is "
                f"the hysteresis gap that prevents flapping"
            )
        if self.min_dwell_steps < 0:
            raise ValueError("min_dwell_steps must be >= 0")
        if self.probe_interval_steps < 1:
            raise ValueError("probe_interval_steps must be >= 1")


@dataclass
class StepResult:
    step_index: int
    batch_size: int
    mode: Mode
    event: Optional[str]  # None, "disabled", or "reactivated"


class SpeculativeCircuitBreaker:
    """Stateful controller: call step(batch_size) once per generation
    round with the currently observed (or, under the simplified-proxy
    design this project chose for P5.3, externally simulated) batch size.

    is_speculative tells the caller whether to run this round through
    speculative decoding or plain target-only generation. When step()
    returns event="reactivated", the caller is responsible for actually
    performing the real cache catch-up (see
    scripts/p5_circuit_breaker_switch_cost.py) before resuming draft
    proposals -- this class only decides WHEN, never performs it.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.mode = Mode.SPECULATIVE
        self._steps_since_disabled = 0
        self._step_index = 0
        self.history: list[StepResult] = []

    def step(self, batch_size: int) -> StepResult:
        event = None

        if self.mode is Mode.SPECULATIVE:
            if batch_size >= self.config.disable_threshold:
                self.mode = Mode.DISABLED
                self._steps_since_disabled = 0
                event = "disabled"
        else:
            self._steps_since_disabled += 1
            past_dwell = self._steps_since_disabled >= self.config.min_dwell_steps
            steps_past_dwell = self._steps_since_disabled - self.config.min_dwell_steps
            on_probe_tick = past_dwell and (steps_past_dwell % self.config.probe_interval_steps == 0)

            if on_probe_tick and batch_size <= self.config.enable_threshold:
                self.mode = Mode.SPECULATIVE
                event = "reactivated"

        result = StepResult(
            step_index=self._step_index,
            batch_size=batch_size,
            mode=self.mode,
            event=event,
        )
        self.history.append(result)
        self._step_index += 1
        return result

    @property
    def is_speculative(self) -> bool:
        return self.mode is Mode.SPECULATIVE

    def event_counts(self) -> dict:
        """How many times each event fired across this instance's whole
        history -- used by tests to bound flapping (total disable+
        reactivate events should be small even under an adversarial
        oscillating signal)."""
        counts = {"disabled": 0, "reactivated": 0}
        for r in self.history:
            if r.event is not None:
                counts[r.event] += 1
        return counts
