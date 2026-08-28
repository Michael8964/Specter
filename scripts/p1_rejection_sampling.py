"""
P1.1 -- Rejection sampling core algorithm, smoke test + basic diagnostics.

Runs speculative_generate (sampling mode) for several prompts, printing the
number of accepted candidates per step, and reports an aggregate "empirical
acceptance ratio" (sum(n_accepted) / sum(gamma), counting bonus tokens as an
extra acceptance) as a rough cross-check against the theoretical alpha upper
bound already computed by p1_acceptance.py.

Note: this is NOT the P1.3 sampling-mode verifier itself -- P1.3 requires a
much stricter statistical test (empirical alpha vs. the theoretical formula
E[min(p,q)], plus downstream task metric parity). This script is only meant
as a "does P1.1 run and look sane" smoke test; rigorous statistical
verification is left to P1.3.
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

GAMMA = 5
MAX_NEW_TOKENS = 40


def main():
    print("=" * 60)
    print("Specter P1.1 - Rejection Sampling Core Algorithm")
    print("=" * 60)
    print(f"Draft:  {DRAFT_MODEL_NAME}")
    print(f"Target: {TARGET_MODEL_NAME}")
    print(f"gamma={GAMMA}, max_new_tokens={MAX_NEW_TOKENS}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)

    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    generator = torch.manual_seed(0)

    total_accepted = 0
    total_gamma = 0

    for idx, prompt in enumerate(PROMPTS, start=1):
        print("\n" + "-" * 60)
        print(f"Prompt {idx}: {prompt}")
        print("-" * 60)

        result = speculative_generate(
            draft_model,
            target_model,
            draft_tokenizer,
            prompt,
            gamma=GAMMA,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            generator=generator,
        )

        for step_idx, step in enumerate(result["steps"], start=1):
            tail = "bonus" if step.bonus_token is not None else "resample"
            print(
                f"  step {step_idx:02d} | "
                f"accepted {step.n_accepted}/{step.gamma} | "
                f"tail={tail}"
            )
            total_accepted += step.n_accepted + (1 if step.bonus_token is not None else 0)
            total_gamma += step.gamma + (1 if step.bonus_token is not None else 0)

        print(f"\n  Output: {result['output_text']!r}")

    empirical_alpha = total_accepted / total_gamma

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total accepted (incl. bonus): {total_accepted}")
    print(f"Total proposed (incl. bonus slot): {total_gamma}")
    print(f"Empirical accept ratio: {empirical_alpha:.4f}")
    print(
        "\n(For reference: p1_acceptance.py's theoretical alpha upper bound is "
        "results/p1_0_gate_result.json -> pytorch_acceptance.overall_alpha. "
        "The ratio above is only a rough sanity check, not the rigorous "
        "statistical test required by P1.3.)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
