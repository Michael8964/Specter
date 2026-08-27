from mlx_lm import load


DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print("=" * 60)
    print("Specter P1.0 - MLX Cross-check")
    print("=" * 60)

    print("\nLoading Draft with MLX...")
    draft_model, draft_tokenizer = load(DRAFT_MODEL)
    print("Draft loaded ✅")

    print("\nLoading Target with MLX...")
    target_model, target_tokenizer = load(TARGET_MODEL)
    print("Target loaded ✅")

    draft_vocab = draft_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()

    print("\nDraft vocab size:", len(draft_vocab))
    print("Target vocab size:", len(target_vocab))

    if draft_vocab == target_vocab:
        print("\n✅ MLX VOCAB CHECK PASSED")
    else:
        print("\n❌ MLX VOCAB CHECK FAILED")


if __name__ == "__main__":
    main()