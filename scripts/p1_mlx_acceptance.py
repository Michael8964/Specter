import mlx.core as mx
from mlx_lm import load


DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
    # MLX model expects shape: [batch, sequence]
    logits = model(input_ids)

    # Last position = prediction for next token
    next_logits = logits[:, -1, :]

    # Convert logits to probabilities
    probabilities = mx.softmax(next_logits, axis=-1)

    return probabilities


def compute_expected_alpha(draft_probs, target_probs):
    overlap = mx.minimum(draft_probs, target_probs)

    alpha = mx.sum(overlap)

    # Force MLX to actually evaluate the lazy computation
    mx.eval(alpha)

    return alpha.item()


def main():
    print("=" * 60)
    print("Specter P1.0 - MLX Acceptance Cross-check")
    print("=" * 60)

    print("\nLoading Draft with MLX...")
    draft_model, draft_tokenizer = load(DRAFT_MODEL)
    print("Draft loaded ✅")

    print("\nLoading Target with MLX...")
    target_model, target_tokenizer = load(TARGET_MODEL)
    print("Target loaded ✅")

    # Same vocabulary safety check
    assert (
        draft_tokenizer.get_vocab()
        == target_tokenizer.get_vocab()
    ), "Draft and target vocabularies do not match."

    all_alphas = []

    for prompt_index, prompt in enumerate(PROMPTS, start=1):

        print("\n" + "-" * 60)
        print(f"Prompt {prompt_index}: {prompt}")
        print("-" * 60)

        token_ids = draft_tokenizer.encode(prompt)

        input_ids = mx.array([token_ids])

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

            # Follow the target greedy path,
            # same idea as our PyTorch implementation.
            target_token = mx.argmax(
                target_probs,
                axis=-1,
            )

            mx.eval(target_token)

            token_id = target_token.item()

            token_text = target_tokenizer.decode(
                [token_id]
            )

            print(
                f"Step {step + 1:02d} | "
                f"alpha = {alpha:.4f} | "
                f"next = {repr(token_text)}"
            )

            # Append target token to context
            input_ids = mx.concatenate(
                [
                    input_ids,
                    mx.array([[token_id]]),
                ],
                axis=1,
            )

    overall_alpha = sum(all_alphas) / len(all_alphas)

    print("\n" + "=" * 60)
    print("FINAL MLX RESULT")
    print("=" * 60)

    print(f"Draft:          {DRAFT_MODEL}")
    print(f"Target:         {TARGET_MODEL}")
    print(f"Contexts tested:{len(all_alphas)}")
    print(f"Overall alpha:  {overall_alpha:.4f}")
    print(f"Minimum alpha:  {min(all_alphas):.4f}")
    print(f"Maximum alpha:  {max(all_alphas):.4f}")

    print("\nPyTorch reference:")
    print("Overall alpha:  0.7685")

    difference = abs(overall_alpha - 0.7685)

    print(f"\nAbsolute difference: {difference:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()