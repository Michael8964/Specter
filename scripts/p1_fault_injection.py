"""
P1.3 (partial) -- Fault injection test for the correctness verifier itself.

Addresses project_plan_v9.md Section 9.6, Risk 3: a correctness verifier that
never fails hasn't actually been proven reliable -- it might simply never
have been exercised against a real bug. This script deliberately reintroduces
three known, specific bugs into the rejection-sampling algorithm and confirms
that the same token-for-token comparison methodology used by P1.2's greedy
verifier (scripts/p1_greedy_verifier.py) actually detects each one, rather
than silently reporting a match regardless of what the implementation does.

Three mutations, each tied to a concrete, named failure mode:

  1. bonus_from_draft
     The bonus token is sampled from the DRAFT model's distribution instead
     of the target's -- exactly the mistake Pitfall 2 (project_plan_v9.md
     Section 9.2) warns against.

  2. greedy_residual_resample
     Greedy-mode rejection uses the sampling-mode residual formula
     (norm(max(0, p_target - p_draft))) instead of the target model's own
     argmax. This is the *exact* bug that was found and fixed in
     src/speculative_decode.py earlier today. Re-injecting it here closes
     the loop: if this fix ever regresses, this test is what would catch it.

  3. swap_accepted_order
     Reverses the order of already-accepted tokens before they're appended
     -- a structural corruption unrelated to the accept/reject math itself
     (the individual tokens are still "correct", just placed in the wrong
     position).

For each mutation, this script re-runs the same comparison p1_greedy_verifier
runs (mutant speculative decode vs. independent target-only greedy decode)
and expects a MISMATCH -- the opposite of what P1.2 expects. A mutation that
still comes back as a perfect match would mean our verification methodology
has a blind spot for that failure mode.

Note: this only covers the "fault injection" half of P1.3. The other half --
rigorous statistical validation of empirical alpha vs. the theoretical
formula, plus downstream task metric parity -- is separate, larger work not
attempted here.
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
from src.speculative_decode import (
    draft_propose,
    target_verify,
    _greedy,
    _next_token_probs,
)


PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "Implement binary search in Java.",
    "The capital of France is",
]

GAMMA = 5
MAX_NEW_TOKENS = 30

MUTATIONS = ["bonus_from_draft", "greedy_residual_resample", "swap_accepted_order"]


def target_only_greedy(target_model, tokenizer, prompt, max_new_tokens):
    """Same reference generator as p1_greedy_verifier.py -- the ground truth."""
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


def mutant_step(draft_model, target_model, context_ids, gamma, mutation):
    """
    Greedy-mode rejection-sampling step (mirrors speculative_decoding_step's
    temperature=0 path), with exactly one deliberately injected bug depending
    on `mutation`. draft_propose / target_verify are reused unmodified from
    the real module -- only the accept/reject/resample/bonus decision below
    is mutated, so any detected mismatch is attributable to that one change.
    """
    draft_tokens, draft_dists = draft_propose(
        draft_model, context_ids, gamma, temperature=0
    )
    target_dists = target_verify(target_model, context_ids, draft_tokens)

    n_accepted = 0
    target_greedy_token = None

    for i in range(gamma):
        target_greedy_token = _greedy(target_dists[i].unsqueeze(0))
        accept = draft_tokens[i] == target_greedy_token
        if accept:
            n_accepted += 1
        else:
            break

    accepted_tokens = draft_tokens[:n_accepted]

    if mutation == "swap_accepted_order":
        accepted_tokens = list(reversed(accepted_tokens))

    resampled_token = None
    bonus_token = None

    if n_accepted < gamma:
        if mutation == "greedy_residual_resample":
            # BUG (reintroduced on purpose): the formula that is only
            # correct for sampling mode, used here in greedy mode.
            p_target_reject = target_dists[n_accepted]
            p_draft_reject = draft_dists[n_accepted]
            adjusted = torch.clamp(p_target_reject - p_draft_reject, min=0.0)
            if adjusted.sum() <= 0:
                adjusted = p_target_reject
            resampled_token = _greedy(adjusted.unsqueeze(0))
        else:
            resampled_token = target_greedy_token
    else:
        if mutation == "bonus_from_draft":
            # BUG (Pitfall 2, reintroduced on purpose): bonus token sampled
            # from the draft model instead of the target model.
            bonus_token = _greedy(draft_dists[gamma - 1].unsqueeze(0))
        else:
            bonus_token = _greedy(target_dists[gamma].unsqueeze(0))

    tail = [resampled_token] if resampled_token is not None else [bonus_token]
    return accepted_tokens + tail


def mutant_generate(draft_model, target_model, tokenizer, prompt, gamma, max_new_tokens, mutation):
    """Same round/EOS/budget bookkeeping as speculative_generate, but calling
    mutant_step instead of the real speculative_decoding_step."""
    device = next(draft_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id
    generated = 0

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        step_gamma = min(gamma, remaining)

        tokens_to_add = mutant_step(
            draft_model, target_model, input_ids, step_gamma, mutation
        )

        cutoff = len(tokens_to_add)
        if eos_token_id is not None and eos_token_id in tokens_to_add:
            cutoff = min(cutoff, tokens_to_add.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        tokens_to_add = tokens_to_add[:cutoff]

        input_ids = torch.cat(
            [input_ids, torch.tensor([tokens_to_add], device=device)], dim=-1
        )
        generated += len(tokens_to_add)

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    return input_ids


def compare(reference_ids, candidate_ids):
    ref_list = reference_ids[0].tolist()
    cand_list = candidate_ids[0].tolist()
    compare_len = min(len(ref_list), len(cand_list))
    mismatches = sum(1 for i in range(compare_len) if ref_list[i] != cand_list[i])
    length_mismatch = len(ref_list) != len(cand_list)
    return mismatches, length_mismatch


def main():
    print("=" * 60)
    print("Specter P1.3 (partial) - Fault Injection Test")
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

    print("\nComputing reference (target-only greedy) sequences once per prompt...")
    references = {
        prompt: target_only_greedy(target_model, target_tokenizer, prompt, MAX_NEW_TOKENS)
        for prompt in PROMPTS
    }

    all_detected = True

    for mutation in MUTATIONS:
        print("\n" + "=" * 60)
        print(f"Mutation: {mutation}")
        print("=" * 60)

        mutation_detected = False

        for idx, prompt in enumerate(PROMPTS, start=1):
            mutant_ids = mutant_generate(
                draft_model, target_model, draft_tokenizer, prompt,
                GAMMA, MAX_NEW_TOKENS, mutation,
            )
            mismatches, length_mismatch = compare(references[prompt], mutant_ids)
            detected = mismatches > 0 or length_mismatch
            mutation_detected = mutation_detected or detected

            print(
                f"  Prompt {idx}: mismatches={mismatches}, "
                f"length_mismatch={length_mismatch} -> "
                f"{'DETECTED' if detected else 'MISSED'}"
            )

        verdict = "DETECTED (verifier caught it)" if mutation_detected else "MISSED (verifier blind spot!)"
        print(f"\n{mutation}: {verdict}")
        all_detected = all_detected and mutation_detected

    print("\n" + "=" * 60)
    if all_detected:
        print("OVERALL: PASS -- every injected fault was detected")
    else:
        print("OVERALL: FAIL -- at least one injected fault went undetected")
    print("=" * 60)


if __name__ == "__main__":
    main()
