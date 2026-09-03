"""
P5.3 prerequisite -- target-only cache ablation (no speculative decoding at all).

Purpose: the user's friend measured a 2.7x speedup from KV-caching a PLAIN
autoregressive target-model loop (no draft model, no speculative decoding),
at only 64 new tokens -- much bigger and much faster to show up than the
1.65x we measured for the full *speculative* pipeline at 400 tokens.

This isolates the same ablation on our own model pair / code: does our own
_forward_step / cache handling, in the simplest possible loop (target model
only, one new token per call), also show a big win at a short length?

  - If YES (~similar to the friend's 2.7x): our low-level caching primitive
    is fine. The reason the *speculative* pipeline needed 400 tokens to pay
    off is specific to how speculative decoding uses the cache (many small
    draft calls, crop-and-rollback, bookkeeping per round) -- not a defect
    in _forward_step itself.
  - If NO: there is likely a real inefficiency in _forward_step / cache
    handling that also hurts the speculative pipeline, worth auditing before
    writing this off as "just needs longer sequences".
"""
import sys
import time
import statistics

import torch

sys.path.insert(0, ".")
from src.model_loader import TARGET_MODEL_NAME, load_model_and_tokenizer
from src.speculative_decode_kv import _forward_step

MAX_NEW_TOKENS = 64
N_RUNS = 3
BASE_SEED = 700


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


@torch.no_grad()
def run_nocache(model, tokenizer, prompt, device, max_new_tokens, generator):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    sync()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model(input_ids=ids, use_cache=False)
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        tok = torch.multinomial(probs.cpu(), 1, generator=generator).to(device)
        ids = torch.cat([ids, tok.unsqueeze(0)], dim=1)
    sync()
    return time.perf_counter() - t0


@torch.no_grad()
def run_cached(model, tokenizer, prompt, device, max_new_tokens, generator):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past = None
    feed = ids
    sync()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        dists, past = _forward_step(model, feed, past)
        probs = dists[-1]
        tok = torch.multinomial(probs.cpu(), 1, generator=generator).to(device)
        feed = tok.unsqueeze(0)
    sync()
    return time.perf_counter() - t0


def main():
    print("=" * 70)
    print("Specter -- target-only KV-cache ablation (mirrors friend's benchmark)")
    print("=" * 70)
    print(f"max_new_tokens={MAX_NEW_TOKENS}, n_runs={N_RUNS}")

    print("Loading Target...")
    model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)
    device = next(model.parameters()).device

    prompt = "Explain how a hash table resolves collisions."
    gen = torch.Generator()
    gen.manual_seed(BASE_SEED)
    run_nocache(model, tokenizer, prompt, device, 10, gen)
    gen.manual_seed(BASE_SEED)
    run_cached(model, tokenizer, prompt, device, 10, gen)
    print("Warmed up (untimed).")

    nocache_times, cached_times = [], []
    for i in range(N_RUNS):
        gen.manual_seed(BASE_SEED + i)
        t_nc = run_nocache(model, tokenizer, prompt, device, MAX_NEW_TOKENS, gen)
        gen.manual_seed(BASE_SEED + i)
        t_c = run_cached(model, tokenizer, prompt, device, MAX_NEW_TOKENS, gen)
        nocache_times.append(t_nc)
        cached_times.append(t_c)
        print(f"  Run {i+1}: no-cache={t_nc:.3f}s  cached={t_c:.3f}s  "
              f"(no-cache {MAX_NEW_TOKENS/t_nc:.2f} tok/s, cached {MAX_NEW_TOKENS/t_c:.2f} tok/s)")

    nc_mean = statistics.mean(nocache_times)
    nc_std = statistics.stdev(nocache_times) if len(nocache_times) > 1 else 0.0
    c_mean = statistics.mean(cached_times)
    c_std = statistics.stdev(cached_times) if len(cached_times) > 1 else 0.0
    print("=" * 70)
    print(f"No cache: {nc_mean:.3f}s +/- {nc_std:.3f}   ({MAX_NEW_TOKENS/nc_mean:.2f} tok/s)")
    print(f"Cached:   {c_mean:.3f}s +/- {c_std:.3f}   ({MAX_NEW_TOKENS/c_mean:.2f} tok/s)")
    print(f"Speedup from caching alone (no-cache / cached): {nc_mean/c_mean:.3f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
