import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_loader import (
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    load_model_and_tokenizer,
)


def main():
    print("=" * 50)
    print("Specter P1.0 - Vocabulary Gate")
    print("=" * 50)

    print("\nLoading draft model...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(
        DRAFT_MODEL_NAME
    )

    print("\nLoading target model...")
    target_model, target_tokenizer = load_model_and_tokenizer(
        TARGET_MODEL_NAME
    )

    draft_vocab = draft_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    print("\nDraft vocab size:", len(draft_vocab))
    print("Target vocab size:", len(target_vocab))

    if draft_vocab == target_vocab:
        print("\n✅ VOCAB CHECK PASSED")
        print("Draft and target tokenizers are compatible.")
    else:
        print("\n❌ VOCAB CHECK FAILED")
        print("Do not continue with this model pair.")


if __name__ == "__main__":
    main()