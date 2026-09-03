"""
P5.3 prerequisite -- KV-cache wall-clock benefit.

speculative_decode.py's draft_propose/target_verify recompute attention over
the ENTIRE sequence so far on every call. speculative_decode_kv.py's cached
versions only process new tokens. The gap between the two should GROW as the
generated sequence gets longer (each round in the non-cached version pays
for re-deriving an ever-longer prefix; the cached version's round cost
should stay roughly flat). This script measures exactly that on a longer
generation (150 tokens, well past the 30-40 token budgets used in
p5_gamma_tune_speedup.py / p5_fluctuation_scenario.py, specifically to make
this growth visible), fixed gamma=5, non-cached vs cached, 3 runs.

Do not trust these numbers unless scripts/p5_kv_cache_verify.py passes
100% first -- a caching bug that silently produces wrong-but-plausible
output would also silently produce a wrong-but-plausible speedup number.

This is also directly relevant to P5.1's own finding: p5_draft_target_split.py
measured target-side time as ~flat per round on the NON-cached
implementation. Once caching exists, that per-round target cost should drop
as context grows -- by how much is worth knowing before P5.3 tries to build
a circuit breaker around it.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from src.speculative_decode import speculative_generate
from src.speculative_decode_kv import speculative_generate_cached


PROMPT = (
    "Write a detailed, step-by-step explanation of how a hash table works, "
    "including how collisions are handled."
)
GAMMA = 5
MAX_NEW_TOKENS = 400
N_RUNS = 3
BASE_SEED = 900


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_run(fn, *args, **kwargs):
    sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    sync()
    elapsed = time.perf_counter() - start
    return result, elapsed


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance ** 0.5


def main():
    print("=" * 60)
    print("Specter -- KV-cache wall-clock speedup")
    print("=" * 60)
    print(f"gamma={GAMMA}, max_new_tokens={MAX_NEW_TOKENS}, n_runs={N_RUNS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()

    print("\nWarming up (untimed)...")
    warm_a = torch.manual_seed(999)
    speculative_generate(
        draft_model, target_model, draft_tokenizer, PROMPT,
        gamma=GAMMA, max_new_tokens=20, temperature=1.0, generator=warm_a,
    )
    warm_b = torch.manual_seed(999)
    speculative_generate_cached(
        draft_model, target_model, draft_tokenizer, PROMPT,
        gamma=GAMMA, max_new_tokens=20, temperature=1.0, generator=warm_b,
    )

    nocache_times, cached_times = [], []

    for run in range(1, N_RUNS + 1):
        seed = BASE_SEED + run

        gen_a = torch.manual_seed(seed)
        result_a, t_a = timed_run(
            speculative_generate, draft_model, target_model, draft_tokenizer, PROMPT,
            gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, generator=gen_a,
        )
        gen_b = torch.manual_seed(seed)
        result_b, t_b = timed_run(
            speculative_generate_cached, draft_model, target_model, draft_tokenizer, PROMPT,
            gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, generator=gen_b,
        )

        match = result_a["input_ids"][0].tolist() == result_b["input_ids"][0].tolist()
        nocache_times.append(t_a)
        cached_times.append(t_b)
        print(f"  Run {run}: no-cache={t_a:.3f}s  cached={t_b:.3f}s  (outputs match: {match})")

    nc_mean, nc_std = mean_std(nocache_times)
    c_mean, c_std = mean_std(cached_times)

    print("\n" + "=" * 60)
    print(f"FINAL RESULT (3 runs, mean +/- std, seconds for one {MAX_NEW_TOKENS}-token generation)")
    print("=" * 60)
    print(f"No cache: {nc_mean:.3f}s +/- {nc_std:.3f}")
    print(f"Cached:   {c_mean:.3f}s +/- {c_std:.3f}")
    print(f"Speedup from caching alone (no-cache / cached): {nc_mean / c_mean:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
