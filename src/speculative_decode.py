"""
P1.1 -- Rejection Sampling core algorithm (Leviathan et al. 2023, arXiv:2211.17192)

Corresponds to the standard algorithm in project_plan_v9.md Appendix A.1:

    for i in 1..gamma:
        draft_token[i] ~ p_DM(x | context, draft_token[1..i-1])
    target_probs = TargetModel.forward(context, draft_token[1..gamma])  # single forward pass
    n_accepted = 0
    for i in 1..gamma:
        r ~ Uniform(0, 1)
        if r < min(1, p_TM(draft_token[i]) / p_DM(draft_token[i])):
            accept draft_token[i]; n_accepted += 1
        else:
            break
    if n_accepted < gamma:
        x_new ~ norm(max(0, p_TM(x) - p_DM(x)))   # resample from the adjusted distribution
    else:
        x_new ~ p_TM(x | context, draft_token[1..gamma])  # bonus token

Correctness requirements this module must satisfy (P1.2 / P1.3 will validate these):
  - greedy mode (temperature=0): output must match the target model's independent
    greedy decoding token-for-token, 100%.
  - sampling mode: over a large sample, the empirical acceptance rate should converge
    to the theoretical value alpha = E[min(p, q)]
    (the same upper bound already computed in p1_acceptance.py).

Known-pitfall defenses (see project_plan_v9.md Section 9.2):
  - Pitfall 1 (tokenizer/vocab mismatch): this module assumes the caller has already
    run the vocab_match assertion via model_loader / p1_gate. We don't re-check it here,
    but speculative_generate() still does one lightweight assert at the entry point in
    case this module gets called directly without the upstream gate.
  - Pitfall 2 (bonus token sampled from the wrong distribution): when all candidates are
    accepted, the new token must come from target_probs at the position right after
    "context + all candidates" -- never from the draft model's distribution, and never
    by asking the draft model to generate one more token to fill the slot.
    See the explicit comment on the bonus_token branch inside speculative_decoding_step().
"""

from dataclasses import dataclass, field

import torch


@dataclass
class StepResult:
    """Bundles the outcome of one round of speculative decoding.

    Mirrors (a subset of) the step-level telemetry fields in
    project_plan_v9.md Appendix C.
    """

    draft_tokens: list[int]
    draft_token_probs: list[float]      # p_DM(draft_token[i]), kept for diagnostics / recomputing alpha
    target_token_probs: list[float]     # p_TM(draft_token[i]), same purpose
    n_accepted: int
    gamma: int
    accepted_tokens: list[int]          # first n_accepted entries of draft_tokens
    resampled_token: int | None         # token resampled from the adjusted distribution, if a rejection happened
    bonus_token: int | None             # token sampled from the target distribution, if all candidates were accepted
    new_tokens: list[int] = field(init=False)

    def __post_init__(self):
        tail = (
            [self.resampled_token]
            if self.resampled_token is not None
            else [self.bonus_token]
        )
        self.new_tokens = self.accepted_tokens + tail


def _next_token_probs(model, input_ids):
    """Run one forward pass and return the softmax distribution at the last position."""
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits

    return torch.softmax(logits[:, -1, :].float(), dim=-1)


def _sample(probs, generator=None):
    """
    Sample one token id from a distribution (batch=1).

    Sampling is always done on CPU: backends like MPS require the generator's
    device to match the input tensor's device, and we want a single fixed-seed
    CPU generator for reproducibility across backends. Model forward passes still
    run on mps/cuda; the sampling step itself operates on a small tensor, so moving
    it to CPU is negligible overhead.
    """
    probs_cpu = probs.detach().to("cpu")
    token = torch.multinomial(probs_cpu, num_samples=1, generator=generator)
    return token.item()


def _greedy(probs):
    return int(torch.argmax(probs, dim=-1).item())


def draft_propose(draft_model, input_ids, gamma, temperature, generator=None):
    """
    Draft model autoregressively proposes `gamma` candidate tokens.

    Returns:
      tokens: List[int], length gamma
      dists: List[Tensor], the full distribution at each step (before picking a token)
        -- we keep the full distribution rather than just the probability of the
           chosen token, because the later rejection-resampling step needs the full
           p_DM distribution to compute norm(max(0, p_TM - p_DM)), not just a scalar.
    """
    tokens = []
    dists = []
    cur = input_ids

    for _ in range(gamma):
        probs = _next_token_probs(draft_model, cur)

        if temperature == 0:
            token = _greedy(probs)
        else:
            token = _sample(probs, generator)

        tokens.append(token)
        dists.append(probs.squeeze(0))  # [vocab]

        cur = torch.cat(
            [cur, torch.tensor([[token]], device=cur.device)],
            dim=-1,
        )

    return tokens, dists


def target_verify(target_model, context_ids, draft_tokens):
    """
    Single forward pass of the target model over [context + all candidates],
    yielding gamma+1 distributions in one shot:
      position right after context's last token   -> predicts draft_token[0]
      ...
      position right after draft_token[gamma-2]    -> predicts draft_token[gamma-1]
      position right after draft_token[gamma-1]    -> predicts the bonus token
                                                        (used if every candidate is accepted)

    This single forward pass is the whole point of speculative decoding's speedup:
    gamma candidates only cost the target model one forward pass, not gamma of them.
    """
    draft_tensor = torch.tensor([draft_tokens], device=context_ids.device)
    full_ids = torch.cat([context_ids, draft_tensor], dim=-1)

    with torch.inference_mode():
        logits = target_model(input_ids=full_ids).logits

    context_len = context_ids.shape[1]
    gamma = len(draft_tokens)

    # Take positions context_len-1 .. context_len+gamma-1 (gamma+1 positions total).
    # (the logits at position i predict token i+1)
    relevant_logits = logits[:, context_len - 1 : context_len + gamma, :]
    target_dists = torch.softmax(relevant_logits.float(), dim=-1).squeeze(0)

    return [target_dists[i] for i in range(gamma + 1)]


def speculative_decoding_step(
    draft_model,
    target_model,
    context_ids,
    gamma,
    temperature=1.0,
    generator=None,
):
    """
    Runs one full round of the rejection-sampling algorithm from
    project_plan_v9.md Appendix A.1.

    When temperature=0, this degenerates into greedy mode: the draft model
    proposes greedily, and the acceptance rule becomes "does the candidate equal
    the target model's argmax at the same position" (this is exactly what the
    P1.2 greedy verifier checks: the result must match the target model's
    independent greedy decoding token-for-token, 100%).
    """
    draft_tokens, draft_dists = draft_propose(
        draft_model, context_ids, gamma, temperature, generator
    )
    target_dists = target_verify(target_model, context_ids, draft_tokens)

    draft_token_probs = []
    target_token_probs = []
    n_accepted = 0
    target_greedy_token = None  # in greedy mode, holds the target's argmax at the first rejected position

    for i in range(gamma):
        token = draft_tokens[i]
        p_draft = draft_dists[i][token].item()
        p_target = target_dists[i][token].item()

        draft_token_probs.append(p_draft)
        target_token_probs.append(p_target)

        if temperature == 0:
            # Greedy mode: acceptance reduces to "is the draft's greedy pick also
            # the target's greedy pick".
            target_greedy_token = _greedy(target_dists[i].unsqueeze(0))
            accept = token == target_greedy_token
        else:
            r = torch.rand(1, generator=generator).item()
            accept_prob = min(1.0, p_target / p_draft) if p_draft > 0 else 0.0
            accept = r < accept_prob

        if accept:
            n_accepted += 1
        else:
            break

    accepted_tokens = draft_tokens[:n_accepted]
    resampled_token = None
    bonus_token = None

    if n_accepted < gamma:
        if temperature == 0:
            # The greedy-equivalence proof requires: on rejection, output the
            # target model's own argmax directly (already computed above as
            # target_greedy_token, at the position where the loop broke).
            # We must NOT fall through to the residual-resampling formula below --
            # that formula exists to preserve the correct marginal distribution
            # under SAMPLING mode. argmax(residual) is not guaranteed to equal
            # argmax(p_target):
            # counterexample: p_target={A:0.5,B:0.3,C:0.2}, p_draft={A:0.49,B:0.01,C:0.5}
            # -> the residual is larger at B than at A, so argmax(residual)=B,
            # which is not the target's true argmax (A).
            # Using this formula in greedy mode would break the "token-for-token
            # match with independent greedy decoding" invariant that P1.2 checks.
            resampled_token = target_greedy_token
        else:
            # Resample from the adjusted distribution: p'_TM = norm(max(0, p_TM - p_DM))
            # using the distributions at the first rejected position (index = n_accepted).
            p_target_reject = target_dists[n_accepted]
            p_draft_reject = draft_dists[n_accepted]

            adjusted = torch.clamp(p_target_reject - p_draft_reject, min=0.0)
            total = adjusted.sum()

            if total <= 0:
                # Numerical edge case: p_target <= p_draft everywhere. Shouldn't
                # happen in theory, but we need a fallback so norm() doesn't
                # divide by zero -- fall back to sampling directly from p_target.
                adjusted = p_target_reject
                total = adjusted.sum()

            adjusted = adjusted / total
            resampled_token = _sample(adjusted.unsqueeze(0), generator)
    else:
        # Pitfall 2: when all gamma candidates are accepted, the extra token must
        # come from target_dists[gamma] (the target model's distribution at the
        # position right after the candidate sequence) -- never from the draft
        # model, and never reused from anything in draft_dists.
        p_bonus = target_dists[gamma]
        if temperature == 0:
            bonus_token = _greedy(p_bonus.unsqueeze(0))
        else:
            bonus_token = _sample(p_bonus.unsqueeze(0), generator)

    return StepResult(
        draft_tokens=draft_tokens,
        draft_token_probs=draft_token_probs,
        target_token_probs=target_token_probs,
        n_accepted=n_accepted,
        gamma=gamma,
        accepted_tokens=accepted_tokens,
        resampled_token=resampled_token,
        bonus_token=bonus_token,
    )


def speculative_generate(
    draft_model,
    target_model,
    tokenizer,
    prompt,
    gamma,
    max_new_tokens,
    temperature=1.0,
    generator=None,
    eos_token_id=None,
):
    """
    Repeatedly calls speculative_decoding_step until max_new_tokens is reached
    or EOS is produced.

    Callers are responsible for having verified that draft_model / target_model
    share the same tokenizer vocab (the P1.0 gate's job); this function only does
    one lightweight assert at the entry point in case the upstream gate got skipped.
    """
    assert (
        draft_model.config.vocab_size == target_model.config.vocab_size
    ), "draft/target vocab_size mismatch -- run the P1.0 gate first (Pitfall 1)"

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    device = next(draft_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    steps = []
    generated = 0

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        step_gamma = min(gamma, remaining)

        result = speculative_decoding_step(
            draft_model,
            target_model,
            input_ids,
            step_gamma,
            temperature=temperature,
            generator=generator,
        )
        steps.append(result)

        tokens_to_add = result.new_tokens

        # A round can yield up to step_gamma + 1 tokens (the extra +1 being
        # the bonus token awarded when every candidate is accepted). Two
        # separate things can make that batch too long to append as-is, and
        # we take whichever cutoff is smaller:
        #
        #   1. EOS appears in the middle of the batch -- everything after it
        #      must be discarded, or we'd silently generate content past the
        #      point where an independent, one-token-at-a-time decode would
        #      already have stopped.
        #   2. The bonus token can push this round's output past `remaining`,
        #      even though step_gamma itself was already capped at
        #      `remaining`. Example: remaining=1 -> step_gamma=1 -> the one
        #      candidate gets accepted -> bonus branch fires -> this round
        #      actually emits 2 tokens instead of the 1 we budgeted for.
        #      (This -- not EOS -- turned out to be the real cause of
        #      spec-decode output being consistently 1 token longer than the
        #      greedy reference in P1.2, even though 0 tokens ever mismatched.)
        cutoff = len(tokens_to_add)
        if eos_token_id is not None and eos_token_id in tokens_to_add:
            cutoff = min(cutoff, tokens_to_add.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        tokens_to_add = tokens_to_add[:cutoff]

        new_tokens = torch.tensor(
            [tokens_to_add], device=device
        )
        input_ids = torch.cat([input_ids, new_tokens], dim=-1)
        generated += len(tokens_to_add)

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    output_text = tokenizer.decode(
        input_ids[0], skip_special_tokens=True
    )

    return {
        "input_ids": input_ids,
        "output_text": output_text,
        "steps": steps,
    }
