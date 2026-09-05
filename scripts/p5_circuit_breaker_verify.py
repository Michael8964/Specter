"""
P5.3 -- deterministic verification of the circuit breaker state machine.

This is pure control-flow logic (src/circuit_breaker.py) -- no real model
calls, no randomness, so every scenario here has a hand-computed expected
event sequence. A mismatch is a real logic bug, the same "no noise source,
so no excuse" standard used for
scripts/p5_kv_cache_fake_model_verify.py's zero-noise fake model.

Scenarios covered:
  1. Signal never reaches the disable threshold -> always speculative.
  2. Clean ramp up and down -> disables at the threshold crossing, stays
     disabled until dwell time passes AND batch drops to the enable
     threshold, reactivates at exactly that point.
  3. Spike then immediate drop -> the minimum dwell time must be honored
     even though the batch signal would otherwise allow an immediate
     reactivation the very next step (this is the DSD-style failure mode
     project_plan_v9.md calls out: reactivating too eagerly right after
     disabling, with no real evidence load has actually settled).
  4. Hysteresis band -> batch drops below the disable threshold but not
     down to the enable threshold; must stay disabled indefinitely (never
     reactivates) as long as it hovers in that band.
  5. Boundary hovering / flapping resistance -> batch repeatedly touches
     or exceeds the disable threshold after already being disabled; must
     not fire a second spurious "disabled" event, and must not reactivate
     unless it actually drops far enough.
  6. probe_interval_steps > 1 -> a step where the batch signal already
     satisfies the enable condition, but isn't a probe tick, must NOT
     reactivate; only the next actual probe tick does.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.circuit_breaker import CircuitBreakerConfig, SpeculativeCircuitBreaker, Mode


def run_scenario(name, config, signal, expected_events):
    """expected_events: dict {step_index: "disabled"|"reactivated"} for
    every step that should fire an event; steps not present must have
    event=None."""
    breaker = SpeculativeCircuitBreaker(config)
    mismatches = []

    for i, batch in enumerate(signal):
        result = breaker.step(batch)
        expected = expected_events.get(i)
        if result.event != expected:
            mismatches.append((i, batch, result.event, expected))

    passed = len(mismatches) == 0
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if mismatches:
        for i, batch, got, exp in mismatches:
            print(f"    step={i} batch={batch}: got event={got!r}, expected={exp!r}")
    return passed


def main():
    print("=" * 70)
    print("Specter -- P5.3 circuit breaker state machine verification")
    print("(pure control logic, no model calls -- deterministic, hand-computed")
    print(" expected event sequences; any mismatch is a real logic bug)")
    print("=" * 70)

    results = []

    # --- Scenario 1: never crosses the threshold ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=3)
    signal = [1, 2, 3, 4, 5, 6, 7, 7, 6, 5]
    results.append(run_scenario("never crosses threshold -> always speculative", cfg, signal, {}))

    # --- Scenario 2: clean ramp up and down ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=3)
    signal = [1, 2, 3, 8, 9, 10, 9, 7, 5, 3, 2, 1]
    #          0  1  2  3  4   5  6  7  8  9 10 11
    # step3: batch=8 >= 8 -> disabled
    # steps4-8: since_disabled = 1,2,3,4,5 ; dwell(3) reached at step6 (since=3)
    #   step6: since=3, probe tick, batch=9 -> not <=4, stays disabled
    #   step7: since=4, probe tick, batch=7 -> not <=4, stays disabled
    #   step8: since=5, probe tick, batch=5 -> not <=4, stays disabled
    #   step9: since=6, probe tick, batch=3 -> <=4 -> reactivated
    expected = {3: "disabled", 9: "reactivated"}
    results.append(run_scenario("clean ramp up/down -> disable then reactivate at the right step", cfg, signal, expected))

    # --- Scenario 3: spike then immediate drop, dwell must still be honored ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=3)
    signal = [1, 8, 1, 1, 1, 1]
    #          0  1  2  3  4  5
    # step1: batch=8 -> disabled, since reset to 0
    # step2: since=1, dwell(3) not reached -> stays disabled despite batch=1<=4
    # step3: since=2, dwell not reached -> stays disabled
    # step4: since=3, dwell reached, probe tick, batch=1<=4 -> reactivated
    expected = {1: "disabled", 4: "reactivated"}
    results.append(run_scenario("spike then immediate drop -> dwell time still enforced", cfg, signal, expected))

    # --- Scenario 4: hysteresis band, never drops far enough ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=3)
    signal = [1, 8, 6, 6, 6, 6, 6, 6]
    #          0  1  2  3  4  5  6  7
    # step1: disabled. batch stays at 6 forever after -- 6 is below disable_threshold(8)
    # but ABOVE enable_threshold(4), so it must never reactivate.
    expected = {1: "disabled"}
    results.append(run_scenario("hysteresis band -> never reactivates while hovering between thresholds", cfg, signal, expected))

    # --- Scenario 5: boundary hovering / flapping resistance ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=3)
    signal = [8, 8, 7, 6, 7, 8, 6, 7, 6, 8, 7]
    #          0  1  2  3  4  5  6  7  8  9 10
    # step0: disabled. Every subsequent value is >=6, never <=4, so no
    # reactivation, and touching 8 again (steps 1, 5, 9) while already
    # DISABLED must not fire a second "disabled" event.
    expected = {0: "disabled"}
    results.append(run_scenario("boundary hovering -> exactly one disable event, no flapping", cfg, signal, expected))

    # --- Scenario 6: probe_interval_steps > 1 skips non-probe-tick steps ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=2, probe_interval_steps=3)
    signal = [8, 9, 9, 2, 2, 2, 2, 2, 2, 2]
    #          0  1  2  3  4  5  6  7  8  9
    # step0: disabled, since reset 0
    # step1: since=1, dwell(2) not reached -> disabled
    # step2: since=2, dwell reached, steps_past_dwell=0, probe tick(0%3==0), batch=9 not<=4 -> disabled
    # step3: since=3, steps_past_dwell=1, NOT a probe tick (1%3!=0) -> disabled, EVEN THOUGH batch=2<=4
    # step4: since=4, steps_past_dwell=2, NOT a probe tick (2%3!=0) -> disabled, even though batch=2<=4
    # step5: since=5, steps_past_dwell=3, probe tick(3%3==0), batch=2<=4 -> reactivated
    expected = {0: "disabled", 5: "reactivated"}
    results.append(run_scenario("probe_interval_steps=3 -> only checks every 3rd step past dwell", cfg, signal, expected))

    # --- Scenario 7: multiple disable/reactivate cycles in one run ---
    cfg = CircuitBreakerConfig(disable_threshold=8, enable_threshold=4, min_dwell_steps=2)
    signal = [1, 8, 1, 1, 1, 9, 1, 1, 1]
    #          0  1  2  3  4  6  6  7  8
    # step1: disabled, since=0
    # step2: since=1, dwell(2) not reached -> disabled
    # step3: since=2, dwell reached, probe, batch=1<=4 -> reactivated
    # step4: batch=1, speculative, stays speculative
    # step5: batch=9 -> disabled again, since=0
    # step6: since=1, dwell not reached -> disabled
    # step7: since=2, dwell reached, probe, batch=1<=4 -> reactivated
    # step8: batch=1, speculative
    expected = {1: "disabled", 3: "reactivated", 5: "disabled", 7: "reactivated"}
    results.append(run_scenario("multiple disable/reactivate cycles -> each handled independently", cfg, signal, expected))

    # --- Scenario 8: config validation rejects an inverted hysteresis gap ---
    invalid_config_rejected = False
    try:
        CircuitBreakerConfig(disable_threshold=4, enable_threshold=8, min_dwell_steps=1)
    except ValueError:
        invalid_config_rejected = True
    status = "PASS" if invalid_config_rejected else "FAIL"
    print(f"[{status}] invalid config (enable_threshold > disable_threshold) is rejected at construction")
    results.append(invalid_config_rejected)

    total = len(results)
    passed = sum(1 for r in results if r)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} scenarios passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
