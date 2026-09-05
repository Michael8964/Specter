"""
P5.3 prerequisite -- Diagnose the sampling-mode divergence found by
p5_kv_cache_verify.py (greedy: 9/9 PASS; sampling: 0/9 FAIL).

Hypothesis: this is NOT a logic bug in the cache bookkeeping (the greedy
pass, requiring token-for-token agreement across dozens of rounds, is
strong evidence the cropping/position_ids/threading logic is correct), but
ordinary floating-point non-associativity between "recompute the whole
sequence in one forward pass" and "reuse cached K/V, only compute the new
tokens" -- two different code paths through the same math that are not
guaranteed to produce bit-identical results on GPU/accelerator hardware.
This is invisible in greedy mode (argmax is robust to tiny numerical noise)
but can flip which token gets drawn in sampling mode whenever the random
draw lands near a probability boundary -- and once one token differs,
every later token differs too, since the two runs are now conditioned on
different text.

This script isolates exactly that, with two checks per test context:

  Check 1 (sanity check): feed the FULL context through the cached code
    path with an EMPTY cache (so it's one single forward pass, same as the
    non-cached path) -- this should be numerically IDENTICAL (or extremely
    close to it) to the non-cached reference, since it's essentially the
    same computation. If this check fails badly, the bug is in
    _forward_step itself (e.g. the extra position_ids / use_cache=True
    arguments), not in cache reuse.

  Check 2 (the real test): split the context into a prefix + last token.
    Build a cache from the prefix ALONE, then feed just the last token
    using that cache (this is genuine cache reuse across two separate
    forward calls) -- and compare the resulting distribution against the
    same full-context-in-one-shot computation. If the hypothesis is right,
    this should be CLOSE (e.g. within ~1e-4) but not bit-identical; if it's
    wildly different, that points to a real bug in how the cache is being
    used, not just numerical noise.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from src.speculative_decode import _next_token_probs
from src.speculative_decode_kv import _forward_step


PROMPTS = [
    "The capital of France is",
    "Write a Python function that checks whether a number is prime.",
    "Explain why the sky is blue in simple terms.",
]


def diagnose_one(model, tokenizer, prompt, label):
    device = next(model.parameters()).device
    context = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Reference: exactly what the non-cached implementation does.
    ref_probs = _next_token_probs(model, context).squeeze(0)  # [vocab]

    # Check 1: cached code path, but with an EMPTY cache, fed the WHOLE
    # context in one shot -- essentially the same computation as ref_probs,
    # just going through _forward_step instead of _next_token_probs.
    fresh_probs, _ = _forward_step(model, context, None)
    fresh_probs = fresh_probs[-1]  # distribution at the last position
    check1_diff = (ref_probs - fresh_probs).abs().max().item()

    # Check 2: the REAL cache-reuse case -- build a cache from everything
    # except the last token, then feed just the last token using that
    # cache. This is two separate forward calls, exactly like round 2+ of
    # real generation.
    prefix, last_token = context[:, :-1], context[:, -1:]
    _, prefix_cache = _forward_step(model, prefix, None)
    incremental_probs, _ = _forward_step(model, last_token, prefix_cache)
    incremental_probs = incremental_probs[-1]
    check2_diff = (ref_probs - incremental_probs).abs().max().item()

    # Does the actual argmax (what greedy mode reads) change under either check?
    argmax_ref = torch.argmax(ref_probs).item()
    argmax_check1 = torch.argmax(fresh_probs).item()
    argmax_check2 = torch.argmax(incremental_probs).item()

    print(f"[{label}] prompt={prompt[:40]!r}")
    print(f"  check1 (fresh full-context via cached path): max|diff|={check1_diff:.2e}  argmax_same={argmax_ref == argmax_check1}")
    print(f"  check2 (prefix cached + 1 new token):         max|diff|={check2_diff:.2e}  argmax_same={argmax_ref == argmax_check2}")
    return check1_diff, check2_diff


def main():
    print("=" * 60)
    print("Specter -- KV-cache numerical divergence diagnosis")
    print("=" * 60)

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    all_diffs = []
    for prompt in PROMPTS:
        all_diffs.append(diagnose_one(draft_model, draft_tokenizer, prompt, "draft "))
        all_diffs.append(diagnose_one(target_model, draft_tokenizer, prompt, "target"))

    max_check1 = max(d[0] for d in all_diffs)
    max_check2 = max(d[1] for d in all_diffs)

    print("\n" + "=" * 60)
    print(f"Worst-case check1 (sanity, should be ~0): {max_check1:.2e}")
    print(f"Worst-case check2 (real cache reuse):      {max_check2:.2e}")
    print()
    if max_check1 > 1e-3:
        print("Check 1 is NOT near-zero -- something in _forward_step itself")
        print("(the extra position_ids / use_cache=True args) is changing the")
        print("computation even with an empty cache. This points to a real bug,")
        print("not numerical noise from cache reuse.")
    elif max_check2 > 1e-2:
        print("Check 2 is large -- cache reuse is producing meaningfully")
        print("different distributions, not just tiny float noise. Worth")
        print("investigating further before trusting the cached implementation.")
    else:
        print("Check 1 ~0 and check 2 small: this is consistent with the")
        print("hypothesis that sampling-mode divergence is ordinary")
        print("floating-point noise from cache reuse (small enough to almost")
        print("never flip an argmax, but occasionally enough to flip which")
        print("token a random draw lands on).")
    print("=" * 60)


if __name__ == "__main__":
    main()
