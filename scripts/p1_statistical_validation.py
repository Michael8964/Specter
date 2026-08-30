"""
P1.3 (remaining half) -- Statistical validation of the accept/reject core.

project_plan_v9.md's validation plan for sampling mode is:
  "over a large sample, the empirical acceptance rate should converge to the
   theoretical value alpha = E[min(p, q)]"
(the same closed-form quantity p1_acceptance.py already computes as the
"theoretical alpha upper bound", overall_alpha=0.7685 in results/p1_0_gate_result.json).

Why this is a different, stronger test than P1.1's smoke test (p1_rejection_sampling.py):
  P1.1 ran full multi-step generation and only looked at 5 prompts once each --
  small sample, and every step's *context* itself depends on the random choices
  made in earlier steps, so no two runs are even asking the same question. That
  is why its result (0.5724) swung wildly step to step and can't be compared
  directly to the theoretical 0.7685.

  This script instead isolates exactly ONE thing: at a FIXED context, does our
  implementation's accept/reject test (the "r < min(1, p_target/p_draft)" line
  inside speculative_decoding_step, src/speculative_decode.py lines ~205-207)
  actually accept at the rate the closed-form formula predicts? To answer that
  with real statistical confidence we need MANY independent trials at the SAME
  context -- not many different contexts sampled once each.

Key efficiency trick: for a fixed context, draft_probs and target_probs (the
two full vocabulary distributions) only need ONE forward pass each, no matter
how many trials we run -- only the sampled draft token and the random accept
draw change between trials, and both of those are cheap tensor ops with no
model call involved. So we get hundreds of independent trials per context for
the price of the 2 forward passes p1_acceptance.py already spends per context.

We reuse the exact same 12 prompts / 10-steps-per-prompt / "walk along the
target's own greedy path" setup as p1_acceptance.py, so the 120 contexts here
are the same 120 contexts the theoretical alpha=0.7685 was computed over --
this keeps the two numbers genuinely comparable instead of two different
experiments that happen to produce similar-looking floats.
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
from src.speculative_decode import _next_token_probs, _sample


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
TRIALS_PER_CONTEXT = 300  # independent accept/reject draws per fixed context
SEED = 0
Z_SCORE = 2.0  # ~95% confidence band width, in units of standard error


def compute_expected_alpha(draft_probs, target_probs):
    """Theoretical acceptance probability: alpha = sum_x min(p_draft(x), p_target(x)).

    Identical formula to p1_acceptance.py's compute_expected_alpha -- duplicated
    here (rather than imported) to keep this script runnable on its own, matching
    the pattern already used across scripts/p1_*.py.
    """
    overlap = torch.minimum(draft_probs, target_probs)
    return overlap.sum().item()


def empirical_accept_rate(draft_probs, target_probs, n_trials, generator):
    """
    Run n_trials independent draws of the real sampling-mode accept/reject test
    at a single fixed context, and return the fraction accepted.

    Each trial mirrors speculative_decoding_step's sampling branch exactly:
      1. draft proposes one token by sampling from draft_probs (via the real
         _sample() helper -- same function draft_propose() calls in production)
      2. accept_prob = min(1, p_target(token) / p_draft(token))
      3. draw r ~ Uniform(0,1); accept iff r < accept_prob
    No model forward pass happens in this loop -- draft_probs/target_probs are
    already fixed, computed once by the caller.
    """
    accepts = 0
    for _ in range(n_trials):
        token = _sample(draft_probs, generator)
        p_draft = draft_probs[0, token].item()
        p_target = target_probs[0, token].item()
        accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0
        r = torch.rand(1, generator=generator).item()
        if r < accept_prob:
            accepts += 1
    return accepts / n_trials


def main():
    print("=" * 60)
    print("Specter P1.3 - Statistical Validation (empirical vs theoretical alpha)")
    print("=" * 60)
    print(f"Draft:  {DRAFT_MODEL_NAME}")
    print(f"Target: {TARGET_MODEL_NAME}")
    print(f"Contexts: {len(PROMPTS)} prompts x {STEPS_PER_PROMPT} steps = {len(PROMPTS) * STEPS_PER_PROMPT}")
    print(f"Trials per context: {TRIALS_PER_CONTEXT}")

    print("\nLoading Draft...")
    draft_model, draft_tokenizer = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    print("\nLoading Target...")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    assert (
        draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()
    ), "vocab mismatch -- run scripts/p1_gate.py first"

    device = next(draft_model.parameters()).device
    generator = torch.manual_seed(SEED)

    context_records = []
    total_accepts = 0
    total_trials = 0

    for prompt_idx, prompt in enumerate(PROMPTS, start=1):
        print("\n" + "-" * 60)
        print(f"Prompt {prompt_idx}: {prompt}")
        print("-" * 60)

        input_ids = draft_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        for step in range(STEPS_PER_PROMPT):
            draft_probs = _next_token_probs(draft_model, input_ids)
            target_probs = _next_token_probs(target_model, input_ids)

            theoretical = compute_expected_alpha(draft_probs, target_probs)
            empirical = empirical_accept_rate(
                draft_probs, target_probs, TRIALS_PER_CONTEXT, generator
            )

            # Standard error of a proportion estimated from TRIALS_PER_CONTEXT draws.
            se = (empirical * (1 - empirical) / TRIALS_PER_CONTEXT) ** 0.5
            diff = abs(empirical - theoretical)
            within_margin = diff <= Z_SCORE * se

            context_records.append(
                {
                    "prompt": prompt,
                    "step": step + 1,
                    "theoretical_alpha": theoretical,
                    "empirical_alpha": empirical,
                    "standard_error": se,
                    "within_margin": within_margin,
                }
            )
            total_accepts += round(empirical * TRIALS_PER_CONTEXT)
            total_trials += TRIALS_PER_CONTEXT

            print(
                f"Step {step + 1:02d} | theoretical={theoretical:.4f} | "
                f"empirical={empirical:.4f} | se={se:.4f} | "
                f"{'OK' if within_margin else 'OUT OF MARGIN'}"
            )

            # Advance the context along the target's own greedy path, same as
            # p1_acceptance.py -- keeps this validation anchored to what the
            # target model would actually generate.
            target_token = torch.argmax(target_probs, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, target_token], dim=-1)

    n_within = sum(1 for c in context_records if c["within_margin"])
    n_total = len(context_records)
    pooled_empirical = total_accepts / total_trials
    mean_theoretical = sum(c["theoretical_alpha"] for c in context_records) / n_total

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Contexts tested:            {n_total}")
    print(f"Contexts within ~95% margin: {n_within}/{n_total} ({100 * n_within / n_total:.1f}%)")
    print(f"Mean theoretical alpha:      {mean_theoretical:.4f}")
    print(f"Pooled empirical alpha:      {pooled_empirical:.4f}")
    print(f"(For reference: results/p1_0_gate_result.json overall_alpha = 0.7685)")

    print("\nGate decision:")
    if n_within / n_total >= 0.90:
        print("PASS -- empirical acceptance rate matches the theoretical formula within statistical margin.")
    else:
        print("FAIL -- too many contexts fell outside the expected statistical margin; investigate the accept/reject implementation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
