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

REWRITE (2026-09-05, after the P5.3 crop-target bugfix in commit e4ec874):
this script used to have its own hand-rolled, standalone copy of both the
cached and non-cached generation loops, instrumented inline -- and that
copy was never updated when the real crop-target bug (missing the
`+ seed.shape[1]` term) was found and fixed in
src/speculative_decode_kv.py. It kept measuring the OLD, buggy behavior
indefinitely, silently, because nothing forced the two copies to stay in
sync. This is now fixed the same way the crop-target formula itself was
fixed (commit a0e0d10): instead of a second hand-written copy of the
algorithm, this script calls the real, verified
speculative_generate/speculative_generate_cached directly, and gets its
draft/target/crop time buckets by temporarily monkey-patching the
relevant module-level functions with timing wrappers, then restoring the
originals. There is now exactly one implementation of the algorithm in
this repo -- this script cannot drift from it again.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
import src.speculative_decode as sd
import src.speculative_decode_kv as sdkv


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


class Bucket:
    """Just an accumulator: total wall time (device-synced) spent inside
    whatever function this bucket is attached to, across an entire run."""

    def __init__(self):
        self.total = 0.0


def _timed_wrapper(fn, bucket):
    """Wrap `fn` so every call adds its synced wall time to `bucket.total`
    and otherwise behaves EXACTLY like `fn` -- same args, same return
    value. Used to instrument the real implementation without changing
    a single line of its logic."""

    def wrapped(*args, **kwargs):
        sync()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        sync()
        bucket.total += time.perf_counter() - t0
        return result

    return wrapped


class patched_timers:
    """Context manager: temporarily replace module-level function names
    with timing-instrumented wrappers, then restore the originals on
    exit. `replacements` maps function name -> Bucket to accumulate into.

    This is what lets this script measure the REAL implementation's
    internals (draft calls / target calls / crop calls) without keeping a
    second, hand-written copy of the algorithm around to drift out of
    sync with it -- which is exactly what happened to the version of this
    script this replaces (see module docstring)."""

    def __init__(self, module, **replacements):
        self.module = module
        self.replacements = replacements
        self._originals = {}

    def __enter__(self):
        for name in self.replacements:
            self._originals[name] = getattr(self.module, name)
        for name, bucket in self.replacements.items():
            setattr(self.module, name, _timed_wrapper(self._originals[name], bucket))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, fn in self._originals.items():
            setattr(self.module, name, fn)
        return False


def run_nocache(draft_model, target_model, tokenizer, generator):
    draft_bucket, target_bucket = Bucket(), Bucket()
    with patched_timers(sd, draft_propose=draft_bucket, target_verify=target_bucket):
        sync()
        t0 = time.perf_counter()
        sd.speculative_generate(
            draft_model, target_model, tokenizer, PROMPT,
            gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, generator=generator,
        )
        sync()
        wall_time = time.perf_counter() - t0
    return wall_time, draft_bucket.total, target_bucket.total, 0.0


def run_cached(draft_model, target_model, tokenizer, generator):
    draft_bucket, target_bucket, crop_bucket = Bucket(), Bucket(), Bucket()
    with patched_timers(
        sdkv,
        draft_propose_cached=draft_bucket,
        target_verify_cached=target_bucket,
        _crop_cache=crop_bucket,
    ):
        sync()
        t0 = time.perf_counter()
        sdkv.speculative_generate_cached(
            draft_model, target_model, tokenizer, PROMPT,
            gamma=GAMMA, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, generator=generator,
        )
        sync()
        wall_time = time.perf_counter() - t0
    return wall_time, draft_bucket.total, target_bucket.total, crop_bucket.total


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
    print(f"\nSpeedup (wall, no-cache / cached): {nc_wall/c_wall:.3f}x")
    print("\n(\"Unaccounted\" = wall time minus every instrumented bucket -- Python-")
    print(" level loop/bookkeeping overhead outside any single model or crop call.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
