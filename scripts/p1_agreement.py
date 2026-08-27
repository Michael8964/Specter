import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_loader import (
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_model_and_tokenizer,
)


def predict_next_token(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, -1, :]
    token_id = torch.argmax(logits, dim=-1, keepdim=True)

    return token_id


def main():
    print("=" * 50)
    print("Specter - Greedy Agreement Test")
    print("=" * 50)

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(
        DRAFT_MODEL_NAME
    )

    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(
        TARGET_MODEL_NAME
    )

    device = next(draft_model.parameters()).device

    prompt = "Write a Python function that checks whether a number is prime."

    inputs = draft_tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)

    matches = 0
    total = 50

    print("\nRunning 50-token agreement test...\n")

    for i in range(total):
        draft_token = predict_next_token(
            draft_model,
            input_ids,
        )

        target_token = predict_next_token(
            target_model,
            input_ids,
        )

        draft_text = draft_tokenizer.decode(
            [draft_token.item()]
        )

        target_text = target_tokenizer.decode(
            [target_token.item()]
        )

        same = draft_token.item() == target_token.item()

        if same:
            matches += 1

        print(
            f"{i + 1:02d} | "
            f"Draft: {repr(draft_text):15} | "
            f"Target: {repr(target_text):15} | "
            f"{'✓' if same else '✗'}"
        )

        # 继续使用 draft 的 token 作为上下文
        input_ids = torch.cat(
            [input_ids, draft_token],
            dim=-1,
        )

    agreement = matches / total

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"Tokens tested:     {total}")
    print(f"Matching tokens:   {matches}")
    print(f"Greedy agreement:  {agreement:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()