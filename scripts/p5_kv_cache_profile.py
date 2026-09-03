"""
P5.3 prerequisite -- Why is the cached implementation SLOWER, not faster?

p5_kv_cache_speedup.py found the opposite of the expected result: caching
made a 150-token generation ~20-35% SLOWER (0.74-0.80x), not faster, even
after fixing _crop_cache to use the non-deprecated negative-count form.
Before trying more fixes blindly, this script decomposes where time
actually goes in each implementation -- same "don't trust an aggregate
number" standard as p5_draft_target_split.py, but comparing cached against
non-cached instead of adaptive against fixed gamma.

Each implementation's single generation is timed and split into buckets:
  - draft forward time:  total time inside all draft-model forward calls
  - target forward time: total time inside all target-model forward calls
  - crop time (cached only): total time inside _crop_cache calls
The remainder (total wall time minus the sum of the above) is reported as
"unaccounted" -- Python-level loop/bookkeeping overhead not inside any
single instrumented call. If the cached version's draft+target forward
time is much lower (as expected from avoiding redundant recompute) but its
"unaccounted" bucket is much higher, that points to per-call Python/cache-
management overhead -- not the model math itself -- as the real cause.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from src.speculative_decode import _sample, _next_token_probs
from src.speculative_decode_kv import _forward_step, _crop_cache, _cache_len


PROMPT = (
    "Write a detailed, step-by-step explanation of how a hash table works, "
    "including how collisions are handled."
)
GAMMA = 5
MAX_NEW_TOKENS = 400
N_RUNS = 2
BASE_SEED = 500


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(fn, *args, **kwargs):
    sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    sync()
    return result, time.perf_counter() - t0


def run_nocache(draft_model, target_model, tokenizer, generator):
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    draft_time = target_time = 0.0
    generated = 0

    sync()
    wall_start = time.perf_counter()

    while generated < MAX_NEW_TOKENS:
        remaining = MAX_NEW_TOKENS - generated
        gamma = min(GAMMA, remaining)

        cur = input_ids
        draft_tokens, draft_dists = [], []
        for _ in range(gamma):
            probs, dt = timed(_next_token_probs, draft_model, cur)
            draft_time += dt
            probs = probs.squeeze(0)
            token = _sample(probs.unsqueeze(0), generator)
            draft_tokens.append(token)
            draft_dists.append(probs)
            cur = torch.cat([cur, torch.tensor([[token]], device=device)], dim=-1)

        full_ids = torch.cat([input_ids, torch.tensor([draft_tokens], device=device)], dim=-1)
        ctx_len = input_ids.shape[1]

        def _tv():
            with torch.inference_mode():
                logits = target_model(input_ids=full_ids).logits
            return torch.softmax(logits[:, ctx_len - 1: ctx_len + gamma, :].float(), dim=-1).squeeze(0)

        target_dists_all, tt = timed(_tv)
        target_time += tt
        target_dists = [target_dists_all[i] for i in range(gamma + 1)]

        n_accepted = 0
        for i in range(gamma):
            token = draft_tokens[i]
            p_draft = draft_dists[i][token].item()
            p_target = target_dists[i][token].item()
            r = torch.rand(1, generator=generator).item()
            accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0
            if r < accept_prob:
                n_accepted += 1
            else:
                break

        if n_accepted < gamma:
            p_t, p_d = target_dists[n_accepted], draft_dists[n_accepted]
            adjusted = torch.clamp(p_t - p_d, min=0.0)
            total = adjusted.sum()
            if total <= 0:
                adjusted, total = p_t, p_t.sum()
            tail = _sample((adjusted / total).unsqueeze(0), generator)
        else:
            tail = _sample(target_dists[gamma].unsqueeze(0), generator)

        new_tokens = draft_tokens[:n_accepted] + [tail]
        cutoff = min(len(new_tokens), remaining)
        new_tokens = new_tokens[:cutoff]
        input_ids = torch.cat([input_ids, torch.tensor([new_tokens], device=device)], dim=-1)
        generated += len(new_tokens)

    sync()
    wall_time = time.perf_counter() - wall_start
    return wall_time, draft_time, target_time, 0.0


def run_cached(draft_model, target_model, tokenizer, generator):
    device = next(draft_model.parameters()).device
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    draft_time = target_time = crop_time = 0.0
    generated = 0
    draft_seed = prompt_ids
    target_seed = prompt_ids
    draft_cache = None
    target_cache = None

    sync()
    wall_start = time.perf_counter()

    while generated < MAX_NEW_TOKENS:
        remaining = MAX_NEW_TOKENS - generated
        gamma = min(GAMMA, remaining)

        draft_start_len = _cache_len(draft_cache)
        target_start_len = _cache_len(target_cache)

        cur_new = draft_seed
        draft_tokens, draft_dists = [], []
        for _ in range(gamma):
            (step_dists, draft_cache), dt = timed(_forward_step, draft_model, cur_new, draft_cache)
            draft_time += dt
            probs = step_dists[-1]
            token = _sample(probs.unsqueeze(0), generator)
            draft_tokens.append(token)
            draft_dists.append(probs)
            cur_new = torch.tensor([[token]], device=device)

        draft_tensor = torch.tensor([draft_tokens], device=device)
        new_ids = torch.cat([target_seed, draft_tensor], dim=-1)
        (dists, target_cache), tt = timed(_forward_step, target_model, new_ids, target_cache)
        target_time += tt
        target_dists = [dists[i] for i in range(dists.shape[0] - gamma - 1, dists.shape[0])]

        n_accepted = 0
        for i in range(gamma):
            token = draft_tokens[i]
            p_draft = draft_dists[i][token].item()
            p_target = target_dists[i][token].item()
            r = torch.rand(1, generator=generator).item()
            accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0
            if r < accept_prob:
                n_accepted += 1
            else:
                break

        if n_accepted < gamma:
            p_t, p_d = target_dists[n_accepted], draft_dists[n_accepted]
            adjusted = torch.clamp(p_t - p_d, min=0.0)
            total = adjusted.sum()
            if total <= 0:
                adjusted, total = p_t, p_t.sum()
            tail = _sample((adjusted / total).unsqueeze(0), generator)
        else:
            tail = _sample(target_dists[gamma].unsqueeze(0), generator)

        new_draft_cache, ct1 = timed(_crop_cache, draft_cache, draft_start_len + n_accepted)
        crop_time += ct1
        draft_cache = new_draft_cache

        new_target_cache, ct2 = timed(_crop_cache, target_cache, target_start_len + target_seed.shape[1] + n_accepted)
        crop_time += ct2
        target_cache = new_target_cache

        next_seed = torch.tensor([[tail]], device=device)
        draft_seed = next_seed
        target_seed = next_seed

        new_tokens = draft_tokens[:n_accepted] + [tail]
        cutoff = min(len(new_tokens), remaining)
        generated += cutoff

    sync()
    wall_time = time.perf_counter() - wall_start
    return wall_time, draft_time, target_time, crop_time


def mean(values):
    return sum(values) / len(values)


def main():
    print("=" * 70)
    print("Specter -- KV-cache time-breakdown diagnosis (why is it slower?)")
    print("=" * 70)
    print(f"gamma={GAMMA}, max_new_tokens={MAX_NEW_TOKENS}, n_runs={N_RUNS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    print("\nWarming up (untimed)...")
    warm = torch.manual_seed(999)
    run_nocache(draft_model, target_model, draft_tokenizer, warm)
    warm2 = torch.manual_seed(999)
    run_cached(draft_model, target_model, draft_tokenizer, warm2)

    nocache_results, cached_results = [], []

    for run in range(1, N_RUNS + 1):
        seed = BASE_SEED + run
        gen_a = torch.manual_seed(seed)
        wall, dr, tg, cr = run_nocache(draft_model, target_model, draft_tokenizer, gen_a)
        nocache_results.append((wall, dr, tg, cr))
        print(f"\nRun {run} NO CACHE:  wall={wall:.3f}s  draft={dr:.3f}s  target={tg:.3f}s  unaccounted={wall-dr-tg:.3f}s")

        gen_b = torch.manual_seed(seed)
        wall, dr, tg, cr = run_cached(draft_model, target_model, draft_tokenizer, gen_b)
        cached_results.append((wall, dr, tg, cr))
        print(f"Run {run} CACHED:    wall={wall:.3f}s  draft={dr:.3f}s  target={tg:.3f}s  crop={cr:.3f}s  unaccounted={wall-dr-tg-cr:.3f}s")

    nc_wall = mean([r[0] for r in nocache_results])
    nc_draft = mean([r[1] for r in nocache_results])
    nc_target = mean([r[2] for r in nocache_results])
    nc_unacc = nc_wall - nc_draft - nc_target

    c_wall = mean([r[0] for r in cached_results])
    c_draft = mean([r[1] for r in cached_results])
    c_target = mean([r[2] for r in cached_results])
    c_crop = mean([r[3] for r in cached_results])
    c_unacc = c_wall - c_draft - c_target - c_crop

    print("\n" + "=" * 70)
    print(f"AVERAGE OVER {N_RUNS} RUNS")
    print("=" * 70)
    print(f"{'':16s} {'No cache':>12s} {'Cached':>12s}")
    print(f"{'Wall time':16s} {nc_wall:11.3f}s {c_wall:11.3f}s")
    print(f"{'Draft time':16s} {nc_draft:11.3f}s {c_draft:11.3f}s")
    print(f"{'Target time':16s} {nc_target:11.3f}s {c_target:11.3f}s")
    print(f"{'Crop time':16s} {'n/a':>12s} {c_crop:11.3f}s")
    print(f"{'Unaccounted':16s} {nc_unacc:11.3f}s {c_unacc:11.3f}s")
    print("\n(\"Unaccounted\" = wall time minus every instrumented bucket -- Python-")
    print(" level loop/bookkeeping overhead outside any single model or crop call.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
