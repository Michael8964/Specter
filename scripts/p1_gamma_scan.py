"""
P1.4 -- Gamma scan: how does the number of tokens produced per round change
as gamma (how many candidates the draft model proposes before each target
verification pass) varies?

Why this matters: every round of speculative decoding costs exactly ONE
target-model forward pass, no matter what gamma is (that single pass verifies
all gamma candidates at once). So "average tokens produced per round" is
directly "average tokens produced per expensive target forward pass" -- this
number, not gamma itself, is the real measure of how much speedup we're
getting. A round produces (n_accepted + 1) tokens: the accepted candidates,
plus either a resampled token (on rejection) or a bonus token (if every
candidate was accepted) -- see StepResult.new_tokens in src/speculative_decode.py.

The tension this script is meant to make visible: larger gamma means each
round *can* produce more tokens (the ceiling is gamma + 1), but it also means
more candidates have to survive the accept/reject test in a row, and a single
rejection anywhere in the chain throws away every candidate proposed after it.
So the mean round length should grow with gamma, but with diminishing (and
possibly negative, in a bad case) returns -- this script measures where that
curve actually sits for our draft/target pair, rather than assuming it.

Scope note: this is an exploratory scan (5 prompts x 1 seed per gamma), not a
statistical hypothesis test like P1.3 -- the goal here is just to characterize
the shape of the curve, not to prove a formula. Treat single-run numbers with
the same "don't over-trust a small sample" caution documented in P1.1/P1.3.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.model_loader import (
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_model_and_tokenizer,
)
from src.speculative_decode import speculative_generate


PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "Implement binary search in Java.",
    "The capital of France is",
    "Explain why the sky is blue in simple terms.",
    "Return a JSON object with fields name, age, and city.",
]

GAMMAS = [1, 3, 5, 7, 10]
MAX_NEW_TOKENS = 40
SEED = 0


def main():
    print("=" * 60)
    print("Specter P1.4 - Gamma Scan")
    print("=" * 60)
    print(f"Draft:  {DRAFT_MODEL_NAME}")
    print(f"Target: {TARGET_MODEL_NAME}")
    print(f"gammas={GAMMAS}, max_new_tokens={MAX_NEW_TOKENS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    generator = torch.manual_seed(SEED)

    gamma_summaries = []

    for gamma in GAMMAS:
        print("\n" + "=" * 60)
        print(f"gamma = {gamma}")
        print("=" * 60)

        round_lengths = []

        for idx, prompt in enumerate(PROMPTS, start=1):
            result = speculative_generate(
                draft_model,
                target_model,
                draft_tokenizer,
                prompt,
                gamma=gamma,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                generator=generator,
            )

            # Use the untruncated length of each round's own output (before
            # speculative_generate's EOS/budget cutoff), since that cutoff is
            # a bookkeeping artifact of stopping generation, not a property
            # of gamma's effect on the accept/reject chain itself.
            lengths_this_prompt = [len(step.new_tokens) for step in result["steps"]]
            round_lengths.extend(lengths_this_prompt)

            print(
                f"  Prompt {idx}: {len(result['steps'])} rounds, "
                f"round lengths = {lengths_this_prompt}"
            )

        mean_length = sum(round_lengths) / len(round_lengths)
        min_length = min(round_lengths)
        max_length = max(round_lengths)

        # Histogram: how many rounds produced each possible length (1..gamma+1).
        histogram = {
            length: round_lengths.count(length) for length in range(1, gamma + 2)
        }

        gamma_summaries.append(
            {
                "gamma": gamma,
                "total_rounds": len(round_lengths),
                "mean_round_length": mean_length,
                "min_round_length": min_length,
                "max_round_length": max_length,
                "histogram": histogram,
            }
        )

        print(f"\n  Total rounds: {len(round_lengths)}")
        print(f"  Mean tokens/round (== mean tokens per target forward pass): {mean_length:.3f}")
        print(f"  Min: {min_length}, Max: {max_length}")
        print(f"  Histogram (length -> count): {histogram}")

    print("\n" + "=" * 60)
    print("SUMMARY -- mean tokens produced per target forward pass, by gamma")
    print("=" * 60)
    for s in gamma_summaries:
        print(f"  gamma={s['gamma']:2d} -> mean round length = {s['mean_round_length']:.3f}")
    print(
        "\n(Vanilla, non-speculative decoding is the gamma=0 baseline: exactly "
        "1 token per target forward pass, always. Any number above 1.0 here "
        "represents a theoretical speedup on the target-forward-pass count, "
        "before accounting for the extra draft-model forward passes gamma costs.)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
