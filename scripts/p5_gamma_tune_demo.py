"""
P5.1 -- GammaTune demo: watch the adaptive controller move gamma up and down,
round by round, on real generations.

This deliberately reuses speculative_decoding_step() unmodified from
src/speculative_decode.py -- only the *loop* around it is new (it asks the
GammaTuneController for a gamma value each round instead of using one fixed
number for the whole generation). This keeps the already-verified P1.1-P1.3
core untouched.

This script only shows the controller's behavior (the gamma trace). It does
NOT yet measure wall-clock speedup vs. a fixed gamma, or test the
"fluctuation scenario" the plan calls for -- those are separate, larger
pieces of P5.1 to build once this basic behavior looks correct.
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
from src.speculative_decode import speculative_decoding_step
from src.gamma_tune import GammaTuneController


PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "The capital of France is",
    "Explain why the sky is blue in simple terms.",
]

MAX_NEW_TOKENS = 40
SEED = 0

# GammaTune hyperparameters, matching the worked example in
# project_plan_v9.md Appendix A.2 (eta=0.3, delta=2, gamma_min=1, gamma_max=10).
GAMMA_INIT = 3
GAMMA_MIN = 1
GAMMA_MAX = 10
ETA = 0.3
DELTA = 2


def adaptive_generate(draft_model, target_model, tokenizer, prompt, max_new_tokens, generator):
    """Mirrors speculative_generate()'s round/EOS/budget bookkeeping, but asks
    a GammaTuneController for gamma each round instead of using a fixed value."""
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    controller = GammaTuneController(
        gamma_init=GAMMA_INIT, gamma_min=GAMMA_MIN, gamma_max=GAMMA_MAX, eta=ETA, delta=DELTA
    )

    generated = 0
    trace = []  # one entry per round: (gamma_used, n_accepted, next_gamma)

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        gamma_used = min(controller.gamma, remaining)

        result = speculative_decoding_step(
            draft_model, target_model, input_ids, gamma_used, temperature=1.0, generator=generator
        )

        tokens_to_add = result.new_tokens
        cutoff = len(tokens_to_add)
        if eos_token_id is not None and eos_token_id in tokens_to_add:
            cutoff = min(cutoff, tokens_to_add.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        tokens_to_add = tokens_to_add[:cutoff]

        input_ids = torch.cat([input_ids, torch.tensor([tokens_to_add], device=device)], dim=-1)
        generated += len(tokens_to_add)

        next_gamma = controller.update(result.n_accepted, gamma_used)
        trace.append((gamma_used, result.n_accepted, next_gamma))

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return trace, output_text


def main():
    print("=" * 60)
    print("Specter P5.1 - GammaTune Demo")
    print("=" * 60)
    print(f"Draft:  {DRAFT_MODEL_NAME}")
    print(f"Target: {TARGET_MODEL_NAME}")
    print(f"gamma_init={GAMMA_INIT}, gamma_min={GAMMA_MIN}, gamma_max={GAMMA_MAX}, eta={ETA}, delta={DELTA}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    generator = torch.manual_seed(SEED)

    for idx, prompt in enumerate(PROMPTS, start=1):
        print("\n" + "-" * 60)
        print(f"Prompt {idx}: {prompt}")
        print("-" * 60)

        trace, output_text = adaptive_generate(
            draft_model, target_model, draft_tokenizer, prompt, MAX_NEW_TOKENS, generator
        )

        for round_idx, (gamma_used, n_accepted, next_gamma) in enumerate(trace, start=1):
            full = "FULL ACCEPT -> expand" if n_accepted == gamma_used else "partial -> EMA fallback"
            print(
                f"  round {round_idx:02d} | gamma_used={gamma_used:2d} | "
                f"accepted={n_accepted:2d} | next_gamma={next_gamma:2d} | {full}"
            )

        gammas_used = [g for g, _, _ in trace]
        print(f"\n  gamma trace: {gammas_used}")
        print(f"  min={min(gammas_used)}, max={max(gammas_used)}, rounds={len(trace)}")

    print("\n" + "=" * 60)
    print(
        "(This only demonstrates the controller's expand/contract behavior. "
        "Wall-clock speedup vs. a fixed gamma, and the fluctuation-scenario "
        "test, are separate follow-up steps.)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
