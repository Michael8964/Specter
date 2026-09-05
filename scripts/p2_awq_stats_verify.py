"""
P2.0 -- deterministic verification of the AWQ activation-statistics collector.

This tests src/awq_calibration.py's ActivationStats + register_activation_hooks
using a hand-computed, tiny synthetic example -- same "no randomness, no excuse
for a mismatch" standard as scripts/p5_kv_cache_fake_model_verify.py and
scripts/p5_circuit_breaker_verify.py.

Test setup: a single nn.Linear(4, 2, bias=False) layer. Two calibration
"batches" are fed through it:

  batch1 = [[ 1, -2,  3, -4],
            [ 2, -2,  0,  4]]
  batch2 = [[-5,  1,  1,  1]]

Combined, that's 3 rows total, so per-input-channel (4 channels) statistics
are hand-computable:

  channel 0: |1|, |2|, |-5|  -> sum=8,  max=5
  channel 1: |-2|,|-2|, |1|  -> sum=5,  max=2
  channel 2: |3|, |0|, |1|   -> sum=4,  max=3
  channel 3: |-4|,|4|, |1|   -> sum=9,  max=4

  n_samples = 3
  mean_abs = [8/3, 5/3, 4/3, 9/3] = [2.6667, 1.6667, 1.3333, 3.0]
  max_abs  = [5, 2, 3, 4]

CRITICAL check (the "input vs output" detail the module docstring warns
about): the Linear layer's weight is deliberately set to something that
would produce WILDLY different output statistics than the input stats
above, so that if the hook accidentally captured `output` instead of
`inputs[0]`, this test would fail loudly rather than accidentally passing.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.awq_calibration import ActivationStats, register_activation_hooks, remove_hooks


def approx_equal(a, b, tol=1e-4):
    return abs(a - b) < tol


def main():
    print("=" * 70)
    print("Specter -- P2.0 AWQ activation-statistics collector verification")
    print("(hand-computed expected values, deterministic weights -- any")
    print(" mismatch is a real bug, not noise)")
    print("=" * 70)

    model = nn.Sequential(nn.Linear(4, 2, bias=False))
    # Deliberately distinctive weight, so input-stats != output-stats --
    # if the hook accidentally captured `output` instead of `inputs[0]`
    # this test would fail loudly instead of silently passing.
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [100.0, 100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0, -100.0],
        ]))

    handles, stats = register_activation_hooks(model)

    batch1 = torch.tensor([[1.0, -2.0, 3.0, -4.0],
                            [2.0, -2.0, 0.0, 4.0]])
    batch2 = torch.tensor([[-5.0, 1.0, 1.0, 1.0]])

    with torch.no_grad():
        model(batch1)
        model(batch2)

    remove_hooks(handles)

    results = []

    # --- Scenario 1: exactly one Linear layer was hooked ---
    ok = len(stats) == 1
    print(f"[{'PASS' if ok else 'FAIL'}] exactly one Linear submodule hooked (got {len(stats)})")
    results.append(ok)

    name, s = next(iter(stats.items()))

    # --- Scenario 2: n_samples == 3 (2 rows from batch1 + 1 from batch2) ---
    ok = s.n_samples == 3
    print(f"[{'PASS' if ok else 'FAIL'}] n_samples == 3 (got {s.n_samples})")
    results.append(ok)

    # --- Scenario 3: mean_abs matches hand computation ---
    expected_mean = [8 / 3, 5 / 3, 4 / 3, 9 / 3]
    got_mean = s.mean_abs().tolist()
    ok = all(approx_equal(a, b) for a, b in zip(got_mean, expected_mean))
    print(f"[{'PASS' if ok else 'FAIL'}] mean_abs == {[round(x, 4) for x in expected_mean]} "
          f"(got {[round(x, 4) for x in got_mean]})")
    results.append(ok)

    # --- Scenario 4: max_abs matches hand computation ---
    expected_max = [5.0, 2.0, 3.0, 4.0]
    got_max = s.max_abs.tolist()
    ok = all(approx_equal(a, b) for a, b in zip(got_max, expected_max))
    print(f"[{'PASS' if ok else 'FAIL'}] max_abs == {expected_max} (got {got_max})")
    results.append(ok)

    # --- Scenario 5: in_features recorded correctly ---
    ok = s.in_features == 4
    print(f"[{'PASS' if ok else 'FAIL'}] in_features == 4 (got {s.in_features})")
    results.append(ok)

    # --- Scenario 6: mean_abs() on an empty accumulator raises ---
    raised = False
    try:
        empty = ActivationStats(in_features=4, sum_abs=torch.zeros(4), max_abs=torch.zeros(4))
        empty.mean_abs()
    except ValueError:
        raised = True
    print(f"[{'PASS' if raised else 'FAIL'}] mean_abs() on zero samples raises ValueError")
    results.append(raised)

    # --- Scenario 7: remove_hooks() actually removes them (no further accumulation) ---
    n_samples_before = s.n_samples
    with torch.no_grad():
        model(batch1)  # hooks already removed -- this must NOT update stats
    ok = s.n_samples == n_samples_before
    print(f"[{'PASS' if ok else 'FAIL'}] remove_hooks() stops further accumulation "
          f"(n_samples still {s.n_samples})")
    results.append(ok)

    total = len(results)
    passed = sum(results)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{total} scenarios passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
