"""
P5.1 -- Fluctuation scenario test: does GammaTune's advantage over a fixed
gamma actually show up more clearly when difficulty changes sharply
mid-generation?

The previous script (p5_gamma_tune_speedup.py) compared adaptive vs. fixed
gamma over 3 fairly "steady" natural prompts and found only a modest ~10%
edge for the adaptive controller -- and reasoned that fixed gamma=5 was
already a strong, well-chosen baseline for that kind of steady difficulty.
This script builds a scenario specifically designed to NOT be steady:

  Phase 1 (easy): continue an explicit counting sequence. Both draft and
    target models should agree on "the next number" almost every time --
    this should behave like a high-alpha context (similar to the highest
    theoretical alpha values we saw in p1_statistical_validation.py, e.g.
    0.99+).
  Phase 2 (hard): immediately after phase 1, splice in an instruction that
    abruptly switches to open-ended creative writing. Draft and target are
    expected to diverge far more often here (lower alpha).

Both phases run back to back in a SINGLE continuous generation (no restart),
so the controller has to react to the transition in real time, exactly like
it would have to for any real prompt that shifts in difficulty partway
through.

Both the fixed-gamma and adaptive configurations share one generation loop
(two_phase_generate) parameterized by a "gamma provider" object -- either a
FixedGammaProvider (always returns the same gamma) or a GammaTuneController.
This avoids maintaining two separate, easy-to-desync copies of the loop.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import (
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_model_and_tokenizer,
)
from src.speculative_decode import speculative_decoding_step
from src.gamma_tune import GammaTuneController


EASY_PROMPT = (
    "Count from 1 to 50, writing each number followed by a comma and a space: "
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "
)
TRANSITION_TEXT = (
    " Actually, stop counting right now. Instead, write a surreal short story "
    "about a whale who dreams in colors that don't exist:"
)

PHASE1_TOKENS = 30
PHASE2_TOKENS = 30
FIXED_GAMMA = 5
N_RUNS = 3
BASE_SEED = 200  # distinct from earlier scripts' seeds

GAMMA_INIT, GAMMA_MIN, GAMMA_MAX, ETA, DELTA = 3, 1, 10, 0.3, 2


class FixedGammaProvider:
    """Same .gamma / .update() interface as GammaTuneController, so both can
    drive the exact same generation loop below without any branching."""

    def __init__(self, gamma):
        self.gamma = gamma

    def update(self, n_accepted, gamma_used):
        return self.gamma  # never changes


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def run_phase(draft_model, target_model, input_ids, eos_token_id, n_tokens, gamma_provider, generator, phase_label, trace):
    device = input_ids.device
    generated = 0
    sync()
    start = time.perf_counter()

    while generated < n_tokens:
        remaining = n_tokens - generated
        gamma_used = min(gamma_provider.gamma, remaining)

        result = speculative_decoding_step(
            draft_model, target_model, input_ids, gamma_used, temperature=1.0, generator=generator
        )

        tokens_to_add = result.new_tokens
        cutoff = len(tokens_to_add)
        if eos_token_id is not None and eos_token_id in tokens_to_add:
            cutoff = min(cutoff, tokens_to_add.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        tokens_to_add = tokens_to_add[:cutoff]

        input_ids = torch.cat([input_ids, torch.tensor([tokens_to_add], device=device)], dim=-1)
        generated += len(tokens_to_add)
        gamma_provider.update(result.n_accepted, gamma_used)
        trace.append((phase_label, gamma_used, result.n_accepted))

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    sync()
    elapsed = time.perf_counter() - start
    return input_ids, elapsed, generated


def two_phase_generate(draft_model, target_model, tokenizer, gamma_provider, generator):
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(EASY_PROMPT, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    trace = []
    input_ids, phase1_time, phase1_tokens = run_phase(
        draft_model, target_model, input_ids, eos_token_id, PHASE1_TOKENS, gamma_provider, generator, "phase1_easy", trace
    )

    transition_ids = tokenizer(TRANSITION_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_ids = torch.cat([input_ids, transition_ids], dim=-1)

    input_ids, phase2_time, phase2_tokens = run_phase(
        draft_model, target_model, input_ids, eos_token_id, PHASE2_TOKENS, gamma_provider, generator, "phase2_hard", trace
    )

    return trace, phase1_time, phase1_tokens, phase2_time, phase2_tokens, input_ids


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def main():
    print("=" * 60)
    print("Specter P5.1 - Fluctuation Scenario Test")
    print("=" * 60)
    print(f"Phase 1 (easy, counting):  {PHASE1_TOKENS} tokens")
    print(f"Phase 2 (hard, creative):  {PHASE2_TOKENS} tokens")
    print(f"fixed_gamma={FIXED_GAMMA}, n_runs={N_RUNS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    print("\nWarming up (untimed)...")
    warmup_gen = torch.manual_seed(999)
    two_phase_generate(draft_model, target_model, draft_tokenizer, FixedGammaProvider(FIXED_GAMMA), warmup_gen)

    fixed_p1_tps, fixed_p2_tps, fixed_total_tps = [], [], []
    adaptive_p1_tps, adaptive_p2_tps, adaptive_total_tps = [], [], []
    example_adaptive_trace = None

    for run in range(1, N_RUNS + 1):
        print("\n" + "=" * 60)
        print(f"Run {run}/{N_RUNS}")
        print("=" * 60)

        run_seed = BASE_SEED + run

        fixed_gen = torch.manual_seed(run_seed)
        f_trace, f_p1_time, f_p1_tok, f_p2_time, f_p2_tok, _ = two_phase_generate(
            draft_model, target_model, draft_tokenizer, FixedGammaProvider(FIXED_GAMMA), fixed_gen
        )
        f_p1_tps = f_p1_tok / f_p1_time
        f_p2_tps = f_p2_tok / f_p2_time
        f_total_tps = (f_p1_tok + f_p2_tok) / (f_p1_time + f_p2_time)
        fixed_p1_tps.append(f_p1_tps)
        fixed_p2_tps.append(f_p2_tps)
        fixed_total_tps.append(f_total_tps)

        adaptive_gen = torch.manual_seed(run_seed)
        controller = GammaTuneController(
            gamma_init=GAMMA_INIT, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, eta=ETA, delta=DELTA
        )
        a_trace, a_p1_time, a_p1_tok, a_p2_time, a_p2_tok, _ = two_phase_generate(
            draft_model, target_model, draft_tokenizer, controller, adaptive_gen
        )
        a_p1_tps = a_p1_tok / a_p1_time
        a_p2_tps = a_p2_tok / a_p2_time
        a_total_tps = (a_p1_tok + a_p2_tok) / (a_p1_time + a_p2_time)
        adaptive_p1_tps.append(a_p1_tps)
        adaptive_p2_tps.append(a_p2_tps)
        adaptive_total_tps.append(a_total_tps)

        if example_adaptive_trace is None:
            example_adaptive_trace = a_trace

        print(f"  Fixed:    phase1={f_p1_tps:.2f} tok/s | phase2={f_p2_tps:.2f} tok/s | total={f_total_tps:.2f} tok/s")
        print(f"  Adaptive: phase1={a_p1_tps:.2f} tok/s | phase2={a_p2_tps:.2f} tok/s | total={a_total_tps:.2f} tok/s")

    print("\n" + "=" * 60)
    print("Example gamma trace (Run 1, adaptive) across the phase1 -> phase2 transition")
    print("=" * 60)
    for i, (phase, gamma_used, n_accepted) in enumerate(example_adaptive_trace, start=1):
        print(f"  round {i:02d} | {phase:12s} | gamma_used={gamma_used:2d} | accepted={n_accepted:2d}")

    f_p1_mean, f_p1_std = mean_std(fixed_p1_tps)
    f_p2_mean, f_p2_std = mean_std(fixed_p2_tps)
    f_total_mean, f_total_std = mean_std(fixed_total_tps)
    a_p1_mean, a_p1_std = mean_std(adaptive_p1_tps)
    a_p2_mean, a_p2_std = mean_std(adaptive_p2_tps)
    a_total_mean, a_total_std = mean_std(adaptive_total_tps)

    print("\n" + "=" * 60)
    print("FINAL RESULT (3 runs, mean +/- std, tokens/second)")
    print("=" * 60)
    print(f"{'':12s} {'Fixed gamma=5':>22s} {'GammaTune adaptive':>22s}")
    print(f"{'Phase 1':12s} {f_p1_mean:9.2f} +/- {f_p1_std:5.2f}    {a_p1_mean:9.2f} +/- {a_p1_std:5.2f}")
    print(f"{'Phase 2':12s} {f_p2_mean:9.2f} +/- {f_p2_std:5.2f}    {a_p2_mean:9.2f} +/- {a_p2_std:5.2f}")
    print(f"{'Total':12s} {f_total_mean:9.2f} +/- {f_total_std:5.2f}    {a_total_mean:9.2f} +/- {a_total_std:5.2f}")

    print(f"\nPhase2/Phase1 throughput ratio (how much each config slows down in the hard phase):")
    print(f"  Fixed:    {f_p2_mean / f_p1_mean:.3f}")
    print(f"  Adaptive: {a_p2_mean / a_p1_mean:.3f}")
    print(f"\nOverall speedup (adaptive / fixed), total throughput: {a_total_mean / f_total_mean:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
