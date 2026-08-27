import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_loader import (
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_model_and_tokenizer,
)


PROMPTS = [
    # Coding
    "Write a Python function that checks whether a number is prime.",
    "Implement binary search in Java.",
    "Write a SQL query to find duplicate emails.",

    # Factual QA
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water freezes at a temperature of",

    # Explanation
    "Explain why the sky is blue in simple terms.",
    "Explain how a hash table works.",
    "Explain what DNS does in simple terms.",

    # Structured / instruction
     "Return a JSON object with fields name, age, and city.",
    "List three advantages of using Linux servers.",
    "Summarize the benefits of unit testing in three sentences.",
]

STEPS_PER_PROMPT = 10


def get_next_token_distribution(model, input_ids):
    """
    Run one forward pass and return the probability
    distribution for the next token.
    """

    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    # Only care about the logits for the NEXT token
    logits = outputs.logits[:, -1, :]

    # Use float32 for numerical stability before softmax
    probabilities = torch.softmax(
        logits.float(),
        dim=-1,
    )

    return probabilities


def compute_expected_alpha(draft_probs, target_probs):
    """
    Expected speculative acceptance probability:

        alpha = sum_x min(p_draft(x), p_target(x))
    """

    overlap = torch.minimum(
        draft_probs,
        target_probs,
    )

    alpha = overlap.sum().item()

    return alpha


def main():
    print("=" * 60)
    print("Specter P1.0 - Acceptance Alpha Gate")
    print("=" * 60)

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(
        DRAFT_MODEL_NAME
    )

    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(
        TARGET_MODEL_NAME
    )

    # Safety check
    assert (
        draft_tokenizer.get_vocab()
        == target_tokenizer.get_vocab()
    ), "Draft and target vocabularies do not match."

    device = next(draft_model.parameters()).device

    all_alphas = []

    for prompt_index, prompt in enumerate(PROMPTS, start=1):

        print("\n" + "-" * 60)
        print(f"Prompt {prompt_index}: {prompt}")
        print("-" * 60)

        inputs = draft_tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(device)

        prompt_alphas = []

        for step in range(STEPS_PER_PROMPT):

            draft_probs = get_next_token_distribution(
                draft_model,
                input_ids,
            )

            target_probs = get_next_token_distribution(
                target_model,
                input_ids,
            )

            alpha = compute_expected_alpha(
                draft_probs,
                target_probs,
            )

            prompt_alphas.append(alpha)
            all_alphas.append(alpha)

            # For now, continue along the TARGET greedy path.
            # This keeps our evaluation context anchored to
            # what the target model itself would generate.
            target_token = torch.argmax(
                target_probs,
                dim=-1,
                keepdim=True,
            )

            token_text = target_tokenizer.decode(
                [target_token.item()]
            )

            print(
                f"Step {step + 1:02d} | "
                f"alpha = {alpha:.4f} | "
                f"next = {repr(token_text)}"
            )

            input_ids = torch.cat(
                [input_ids, target_token],
                dim=-1,
            )

        prompt_mean = sum(prompt_alphas) / len(prompt_alphas)

        print(
            f"\nPrompt mean alpha: "
            f"{prompt_mean:.4f}"
        )

    overall_alpha = sum(all_alphas) / len(all_alphas)

    min_alpha = min(all_alphas)
    max_alpha = max(all_alphas)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Draft:          {DRAFT_MODEL_NAME}")
    print(f"Target:         {TARGET_MODEL_NAME}")
    print(f"Contexts tested:{len(all_alphas)}")
    print(f"Overall alpha:  {overall_alpha:.4f}")
    print(f"Minimum alpha:  {min_alpha:.4f}")
    print(f"Maximum alpha:  {max_alpha:.4f}")

    print("\nGate decision:")

    if overall_alpha < 0.40:
        print("❌ FAIL")
        print("alpha < 0.40 — change model pair.")

    elif overall_alpha < 0.65:
        print("⚠️ CONDITIONAL PASS")
        print(
            "0.40 <= alpha < 0.65 — continue, "
            "but lower performance expectations."
        )

    else:
        print("✅ PASS")
        print("alpha >= 0.65 — proceed normally.")

    print("=" * 60)


if __name__ == "__main__":
    main()