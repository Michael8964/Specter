import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model_and_tokenizer(model_name):
    device = get_device()

    print(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer

