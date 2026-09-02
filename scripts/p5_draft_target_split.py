"""
P5.1 -- Splitting round time into draft-side vs. target-side.

The fluctuation scenario test (p5_fluctuation_scenario.py) showed that TOTAL
round time barely differed between fixed gamma=5 and the adaptive controller
during the hard phase, even though the adaptive controller was correctly
shrinking gamma down to 1 most of the time there. The follow-up question
(raised while reviewing that result): is the adaptive controller actually
saving real time on the DRAFT model's computation, and that saving is just
being masked in the TOTAL time by the target model's verification cost
staying roughly constant regardless of gamma?

This script answers that directly by timing draft_propose() and
target_verify() SEPARATELY, instead of only timing the whole round. It
reuses draft_propose / target_verify unmodified from src/speculative_decode.py
(same pattern already used in scripts/p1_fault_injection.py) and locally
reimplements the sampling-mode (temperature=1.0) accept/reject/resample/bonus
bookkeeping so we get this timing granularity without touching the
already-verified core algorithm.

Same two-phase (easy counting -> hard creative writing) construction as
p5_fluctuation_scenario.py, so results are directly comparable to that run.
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
from src.speculative_decode import draft_propose, target_verify, _sample
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
BASE_SEED = 300  # distinct from earlier scripts' seeds

GAMMA_INIT, GAMMA_MIN, GAMMA_MAX, ETA, DELTA = 3, 1, 10, 0.3, 2


class FixedGammaProvider:
    def __init__(self, gamma):
        self.gamma = gamma

    def update(self, n_accepted, gamma_used):
        return self.gamma


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_step(draft_model, target_model, context_ids, gamma, generator):
    """Sampling-mode round (temperature=1.0), mirroring
    speculative_decoding_step's sampling branch exactly, but with
    draft_propose and target_verify timed as two separate segments."""
    sync()
    t0 = time.perf_counter()
    draft_tokens, draft_dists = draft_propose(draft_model, context_ids, gamma, temperature=1.0, generator=generator)
    sync()
    t1 = time.perf_counter()
    target_dists = target_verify(target_model, context_ids, draft_tokens)
    sync()
    t2 = time.perf_counter()

    draft_time = t1 - t0
    target_time = t2 - t1

    n_accepted = 0
    for i in range(gamma):
        token = draft_tokens[i]
        p_draft = draft_dists[i][token].item()
        p_target = target_dists[i][token].item()
        accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0
        r = torch.rand(1, generator=generator).item()
        if r < accept_prob:
            n_accepted += 1
        else:
            break

    accepted_tokens = draft_tokens[:n_accepted]
    if n_accepted < gamma:
        p_target_reject = target_dists[n_accepted]
        p_draft_reject = draft_dists[n_accepted]
        adjusted = torch.clamp(p_target_reject - p_draft_reject, min=0.0)
        total = adjusted.sum()
        if total <= 0:
            adjusted = p_target_reject
            total = adjusted.sum()
        adjusted = adjusted / total
        tail_token = _sample(adjusted.unsqueeze(0), generator)
    else:
        tail_token = _sample(target_dists[gamma].unsqueeze(0), generator)

    new_tokens = accepted_tokens + [tail_token]
    return new_tokens, n_accepted, draft_time, target_time


def run_phase(draft_model, target_model, input_ids, eos_token_id, n_tokens, gamma_provider, generator):
    device = input_ids.device
    generated = 0
    total_draft_time = 0.0
    total_target_time = 0.0

    while generated < n_tokens:
        remaining = n_tokens - generated
        gamma_used = min(gamma_provider.gamma, remaining)

        new_tokens, n_accepted, draft_time, target_time = timed_step(
            draft_model, target_model, input_ids, gamma_used, generator
        )

        cutoff = len(new_tokens)
        if eos_token_id is not None and eos_token_id in new_tokens:
            cutoff = min(cutoff, new_tokens.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        new_tokens = new_tokens[:cutoff]

        input_ids = torch.cat([input_ids, torch.tensor([new_tokens], device=device)], dim=-1)
        generated += len(new_tokens)
        gamma_provider.update(n_accepted, gamma_used)
        total_draft_time += draft_time
        total_target_time += target_time

        if eos_token_id is not None and eos_token_id in new_tokens:
            break

    return input_ids, generated, total_draft_time, total_target_time


def two_phase_generate(draft_model, target_model, tokenizer, gamma_provider, generator):
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(EASY_PROMPT, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    input_ids, p1_tokens, p1_draft_time, p1_target_time = run_phase(
        draft_model, target_model, input_ids, eos_token_id, PHASE1_TOKENS, gamma_provider, generator
    )

    transition_ids = tokenizer(TRANSITION_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_ids = torch.cat([input_ids, transition_ids], dim=-1)

    input_ids, p2_tokens, p2_draft_time, p2_target_time = run_phase(
        draft_model, target_model, input_ids, eos_token_id, PHASE2_TOKENS, gamma_provider, generator
    )

    return {
        "phase1": {"tokens": p1_tokens, "draft_time": p1_draft_time, "target_time": p1_target_time},
        "phase2": {"tokens": p2_tokens, "draft_time": p2_draft_time, "target_time": p2_target_time},
    }


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def main():
    print("=" * 60)
    print("Specter P5.1 - Draft-side vs Target-side Time Split")
    print("=" * 60)
    print(f"Same fluctuation scenario as p5_fluctuation_scenario.py")
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

    # Also measure a single isolated forward pass for each model, to get a
    # clean baseline estimate of c = T_target / T_draft, decoupled from any
    # round bookkeeping.
    single_prompt_ids = draft_tokenizer(EASY_PROMPT, return_tensors="pt")["input_ids"].to(
        next(draft_model.parameters()).device
    )
    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        draft_model(input_ids=single_prompt_ids)
    sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        target_model(input_ids=single_prompt_ids)
    sync()
    t2 = time.perf_counter()
    single_draft_time = t1 - t0
    single_target_time = t2 - t1
    print(f"\nSingle isolated forward pass: draft={single_draft_time*1000:.1f}ms, target={single_target_time*1000:.1f}ms")
    print(f"c = T_target / T_draft (single pass estimate) = {single_target_time / single_draft_time:.2f}")

    fixed_p2_draft, fixed_p2_target = [], []
    adaptive_p2_draft, adaptive_p2_target = [], []

    for run in range(1, N_RUNS + 1):
        print("\n" + "=" * 60)
        print(f"Run {run}/{N_RUNS}")
        print("=" * 60)

        run_seed = BASE_SEED + run

        fixed_gen = torch.manual_seed(run_seed)
        f_result = two_phase_generate(draft_model, target_model, draft_tokenizer, FixedGammaProvider(FIXED_GAMMA), fixed_gen)

        adaptive_gen = torch.manual_seed(run_seed)
        controller = GammaTuneController(gamma_init=GAMMA_INIT, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, eta=ETA, delta=DELTA)
        a_result = two_phase_generate(draft_model, target_model, draft_tokenizer, controller, adaptive_gen)

        fixed_p2_draft.append(f_result["phase2"]["draft_time"])
        fixed_p2_target.append(f_result["phase2"]["target_time"])
        adaptive_p2_draft.append(a_result["phase2"]["draft_time"])
        adaptive_p2_target.append(a_result["phase2"]["target_time"])

        print(
            f"  Phase 2 (hard) -- Fixed:    draft={f_result['phase2']['draft_time']:.3f}s, "
            f"target={f_result['phase2']['target_time']:.3f}s"
        )
        print(
            f"  Phase 2 (hard) -- Adaptive: draft={a_result['phase2']['draft_time']:.3f}s, "
            f"target={a_result['phase2']['target_time']:.3f}s"
        )

    f_draft_mean, f_draft_std = mean_std(fixed_p2_draft)
    f_target_mean, f_target_std = mean_std(fixed_p2_target)
    a_draft_mean, a_draft_std = mean_std(adaptive_p2_draft)
    a_target_mean, a_target_std = mean_std(adaptive_p2_target)

    print("\n" + "=" * 60)
    print("FINAL RESULT -- Phase 2 (hard) time breakdown, 3 runs mean +/- std")
    print("=" * 60)
    print(f"{'':20s} {'Fixed gamma=5':>20s} {'GammaTune adaptive':>20s}")
    print(f"{'Draft-side time':20s} {f_draft_mean:7.3f}s +/- {f_draft_std:.3f}   {a_draft_mean:7.3f}s +/- {a_draft_std:.3f}")
    print(f"{'Target-side time':20s} {f_target_mean:7.3f}s +/- {f_target_std:.3f}   {a_target_mean:7.3f}s +/- {a_target_std:.3f}")
    print(f"\nDraft-side time saved by adaptive: {f_draft_mean - a_draft_mean:.3f}s ({100*(1 - a_draft_mean/f_draft_mean):.1f}% reduction)")
    print(f"Target-side time difference: {f_target_mean - a_target_mean:.3f}s ({100*(1 - a_target_mean/f_target_mean):.1f}% reduction)")
    print(
        "\n(If the hypothesis is right, draft-side time should drop a lot while "
        "target-side time stays roughly flat -- confirming the target model's "
        "cost dominates the round regardless of gamma, which is why the TOTAL "
        "time in p5_fluctuation_scenario.py barely moved.)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
