"""
P1.2 -- Greedy mode verifier

Requirement (project_plan_v9.md Section 12, validation/test plan):
  In greedy mode, speculative decoding (temperature=0) must produce, token
  for token, exactly the same output as the target model's independent
  direct inference (i.e. no draft model involved) -- allowing at most
  negligible floating-point error.

Approach:
  For the same prompt, run both:
    (a) target_model decoding N tokens on its own, greedily (argmax each step)
        -- this is the reference answer.
    (b) speculative_generate(..., temperature=0) for the same length.
  Then strictly compare the token id sequences from (a) and (b) position by
  position.

This is the one check so far that should pass 100% or not at all -- unlike
sampling mode, which only needs to be statistically close, here any single
mismatched token counts as a verification failure.
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
from src.speculative_decode import speculative_generate, _next_token_probs, _greedy


PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "Implement binary search in Java.",
    "The capital of France is",
    "Explain why the sky is blue in simple terms.",
    "Return a JSON object with fields name, age, and city.",
]

GAMMA = 5
MAX_NEW_TOKENS = 30


def target_only_greedy(target_model, tokenizer, prompt, max_new_tokens):
    """Target model decoding on its own, greedily, token by token -- the reference answer."""
    device = next(target_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        probs = _next_token_probs(target_model, input_ids)
        token = _greedy(probs)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[token]], device=device)], dim=-1
        )
        if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
            break

    return input_ids


def main():
    print("=" * 60)
    print("Specter P1.2 - Greedy Mode Verifier")
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

    all_pass = True

    for idx, prompt in enumerate(PROMPTS, start=1):
        print("\n" + "-" * 60)
        print(f"Prompt {idx}: {prompt}")
        print("-" * 60)

        reference_ids = target_only_greedy(
            target_model, target_tokenizer, prompt, MAX_NEW_TOKENS
        )

        result = speculative_generate(
            draft_model,
            target_model,
            draft_tokenizer,
            prompt,
            gamma=GAMMA,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )
        spec_ids = result["input_ids"]

        # Only compare up to the shorter of the two lengths (EOS can make them
        # differ in length -- that mismatch itself needs to be reported, not
        # silently ignored).
        ref_list = reference_ids[0].tolist()
        spec_list = spec_ids[0].tolist()
        compare_len = min(len(ref_list), len(spec_list))

        mismatches = [
            (i, ref_list[i], spec_list[i])
            for i in range(compare_len)
            if ref_list[i] != spec_list[i]
        ]

        length_mismatch = len(ref_list) != len(spec_list)
        prompt_pass = (len(mismatches) == 0) and (not length_mismatch)
        all_pass = all_pass and prompt_pass

        print(f"Reference length: {len(ref_list)}, Spec-decode length: {len(spec_list)}")
        print(f"Mismatched positions: {len(mismatches)}")

        if mismatches:
            for i, r, s in mismatches[:5]:
                r_text = target_tokenizer.decode([r])
                s_text = target_tokenizer.decode([s])
                print(f"  pos {i}: reference={r!r}({r_text!r}) vs spec={s!r}({s_text!r})")

        if length_mismatch:
            print(
                "  [length mismatch -- most likely one side hit EOS first; "
                "needs manual review, conservatively marked FAIL here]"
            )

        print("PASS" if prompt_pass else "FAIL")

    print("\n" + "=" * 60)
    print("OVERALL:", "PASS (100% match on all prompts)" if all_pass else "FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()
