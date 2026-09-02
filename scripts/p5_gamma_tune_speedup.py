"""
P5.1 -- Wall-clock speedup: GammaTune (adaptive gamma) vs. a fixed gamma.

Everything measured in this project so far (alpha, mean round length) has
been a TOKEN-COUNT metric, not a TIME metric -- and we've flagged repeatedly
(P1.4, and the discussion of the friend's repo reporting 0.93-1.0x "speedup")
that a good token-count curve does not automatically mean real wall-clock
speedup. This script is the first one in the project that actually times
generations with a clock, to see whether GammaTune's adaptive gamma beats a
sensibly-chosen fixed gamma (gamma=5, the sweet spot found empirically in
scripts/p1_gamma_scan.py) in real seconds, not just in tokens produced.

Two things need care here that none of the earlier scripts had to deal with:

  1. Warmup: the first MPS forward pass of a process often pays a one-time
     setup cost (kernel compilation, memory allocation) unrelated to the
     algorithm itself. We run one untimed throwaway generation first so that
     cost doesn't leak into the real measurements.

  2. Synchronization: MPS (like CUDA) queues work asynchronously -- a Python
     call can return before the GPU has actually finished the work. Wrapping
     code in time.perf_counter() without synchronizing risks measuring "how
     long it took to submit the work" instead of "how long the work actually
     took". torch.mps.synchronize() is called immediately before starting and
     immediately before stopping every timer, forcing Python to wait for all
     pending MPS work to really finish.

Each configuration (fixed gamma=5, adaptive GammaTune) is run 3 times (with a
different seed each time) over the same 3 prompts, and we report throughput
(tokens/second) as mean +/- standard deviation, per the project's own "don't
trust a single run" standard.
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
from src.speculative_decode import speculative_decoding_step, speculative_generate
from src.gamma_tune import GammaTuneController


PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "The capital of France is",
    "Explain why the sky is blue in simple terms.",
]

MAX_NEW_TOKENS = 40
FIXED_GAMMA = 5  # the empirical sweet spot from scripts/p1_gamma_scan.py
N_RUNS = 3
BASE_SEED = 100  # distinct from earlier scripts' seed=0, to avoid accidental reuse

GAMMA_INIT, GAMMA_MIN, GAMMA_MAX, ETA, DELTA = 3, 1, 10, 0.3, 2


def sync():
    """Block until all pending MPS work has actually finished, so our timer
    measures real compute time instead of just 'time to submit work'."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_fixed_generate(draft_model, target_model, tokenizer, prompt, generator):
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    sync()
    start = time.perf_counter()
    result = speculative_generate(
        draft_model, target_model, tokenizer, prompt,
        gamma=FIXED_GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, generator=generator,
    )
    sync()
    elapsed = time.perf_counter() - start

    n_generated = result["input_ids"].shape[1] - prompt_len
    return elapsed, n_generated


def timed_adaptive_generate(draft_model, target_model, tokenizer, prompt, generator):
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    controller = GammaTuneController(
        gamma_init=GAMMA_INIT, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, eta=ETA, delta=DELTA
    )

    generated = 0

    sync()
    start = time.perf_counter()

    while generated < MAX_NEW_TOKENS:
        remaining = MAX_NEW_TOKENS - generated
        gamma_used = min(controller.gamma, remaining)

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
        controller.update(result.n_accepted, gamma_used)

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    sync()
    elapsed = time.perf_counter() - start

    return elapsed, generated


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def main():
    print("=" * 60)
    print("Specter P5.1 - Wall-Clock Speedup: GammaTune vs Fixed Gamma")
    print("=" * 60)
    print(f"Draft:  {DRAFT_MODEL_NAME}")
    print(f"Target: {TARGET_MODEL_NAME}")
    print(f"fixed_gamma={FIXED_GAMMA}, max_new_tokens={MAX_NEW_TOKENS}, n_runs={N_RUNS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    print("\nWarming up (untimed)...")
    warmup_generator = torch.manual_seed(999)
    timed_fixed_generate(draft_model, target_model, draft_tokenizer, PROMPTS[0], warmup_generator)

    fixed_throughputs = []
    adaptive_throughputs = []

    for run in range(1, N_RUNS + 1):
        print("\n" + "=" * 60)
        print(f"Run {run}/{N_RUNS}")
        print("=" * 60)

        run_seed = BASE_SEED + run

        fixed_total_tokens = 0
        fixed_total_time = 0.0
        adaptive_total_tokens = 0
        adaptive_total_time = 0.0

        for idx, prompt in enumerate(PROMPTS, start=1):
            fixed_gen = torch.manual_seed(run_seed)
            f_elapsed, f_tokens = timed_fixed_generate(
                draft_model, target_model, draft_tokenizer, prompt, fixed_gen
            )
            fixed_total_tokens += f_tokens
            fixed_total_time += f_elapsed

            adaptive_gen = torch.manual_seed(run_seed)
            a_elapsed, a_tokens = timed_adaptive_generate(
                draft_model, target_model, draft_tokenizer, prompt, adaptive_gen
            )
            adaptive_total_tokens += a_tokens
            adaptive_total_time += a_elapsed

            print(
                f"  Prompt {idx}: fixed={f_tokens} tok / {f_elapsed:.3f}s "
                f"({f_tokens / f_elapsed:.2f} tok/s)  |  "
                f"adaptive={a_tokens} tok / {a_elapsed:.3f}s "
                f"({a_tokens / a_elapsed:.2f} tok/s)"
            )

        fixed_tps = fixed_total_tokens / fixed_total_time
        adaptive_tps = adaptive_total_tokens / adaptive_total_time
        fixed_throughputs.append(fixed_tps)
        adaptive_throughputs.append(adaptive_tps)

        print(f"\n  Run {run} totals: fixed={fixed_tps:.2f} tok/s, adaptive={adaptive_tps:.2f} tok/s")

    fixed_mean, fixed_std = mean_std(fixed_throughputs)
    adaptive_mean, adaptive_std = mean_std(adaptive_throughputs)
    speedup = adaptive_mean / fixed_mean

    print("\n" + "=" * 60)
    print("FINAL RESULT (3 runs, mean +/- std, tokens/second)")
    print("=" * 60)
    print(f"Fixed gamma={FIXED_GAMMA}:  {fixed_mean:.2f} +/- {fixed_std:.2f} tok/s   (per-run: {[f'{v:.2f}' for v in fixed_throughputs]})")
    print(f"GammaTune adaptive:  {adaptive_mean:.2f} +/- {adaptive_std:.2f} tok/s   (per-run: {[f'{v:.2f}' for v in adaptive_throughputs]})")
    print(f"\nSpeedup (adaptive / fixed): {speedup:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
