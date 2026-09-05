"""
P5.3 prerequisite -- Incremental KV-cache for draft_propose / target_verify.

speculative_decode.py's draft_propose() and target_verify() are correct
(v1.0 report) but inefficient: every call re-runs the model over the ENTIRE
sequence so far, recomputing attention for tokens that were already
processed in a previous call. This module reimplements the same algorithm
using HuggingFace's `past_key_values` incremental cache, so each forward
pass only processes tokens that haven't been fed to the model yet.

This file deliberately does NOT modify speculative_decode.py -- that core is
already verified and this module is cross-checked directly against it
(scripts/p5_kv_cache_verify.py): with the same seed, this implementation
must produce token-for-token identical output.

Cache bookkeeping, one round at a time:

  draft cache: draft_propose_cached() is called with `seed_ids` = whatever
    hasn't been fed to the draft model yet (the WHOLE prompt on round 1,
    otherwise just last round's tail token). It walks gamma steps, feeding
    one new token at a time, extending the cache by exactly gamma
    positions. If some of those gamma proposed tokens are later rejected,
    the caller crops the draft cache back down before the next round.

  target cache: target_verify_cached() is called with `seed_ids` (same idea)
    PLUS this round's gamma draft tokens, all in ONE forward pass -- this is
    still the single-pass batch verification that makes speculative
    decoding fast; caching doesn't change that. It returns gamma+1
    distributions exactly like the original target_verify(), by taking the
    LAST gamma+1 positions of the output regardless of how long `seed_ids`
    was (length context_len on round 1, length 1 on every later round).

  Whichever tail token (resample or bonus) is chosen at the end of a round
  has NOT yet been fed to either model -- it becomes next round's seed_ids
  for both caches, so the next call folds it in for free.
"""

import torch

from src.speculative_decode import _sample, _greedy, StepResult


def _cache_len(past_key_values):
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[-2]  # legacy tuple-of-(key, value) per layer


def _crop_cache(past_key_values, new_len):
    """Roll a cache back to its first `new_len` positions, discarding
    anything after -- used when some proposed tokens turn out to be
    rejected and must not influence future rounds. This crop is the actual
    'KV cache rebuild' cost P5.3 needs to measure.

    Newer transformers Cache.crop() wants a NEGATIVE count ("remove this
    many tokens from the end") rather than a positive absolute target
    length -- passing a positive value still works but goes through a
    deprecated compatibility path. We compute the negative delta ourselves
    so we always call the current, non-deprecated form."""
    if past_key_values is None:
        return None
    current_len = _cache_len(past_key_values)
    if current_len == new_len:
        return past_key_values
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(new_len - current_len)  # negative: tokens to drop from the end
        return past_key_values
    return tuple((k[..., :new_len, :], v[..., :new_len, :]) for k, v in past_key_values)


def _forward_step(model, new_ids, past_key_values, keep_last_n=None):
    """Feed ONLY the tokens not yet in past_key_values; return the softmax
    distribution at the last `keep_last_n` new positions (default: all of
    them) plus the updated cache.

    `keep_last_n` is a pure performance knob, not a correctness one: on
    round 1, `new_ids` can be the whole prompt (draft only ever needs its
    LAST position's distribution to propose the next token; target only
    ever needs the last gamma+1), so computing a softmax over the full
    ~150k-token vocab for every discarded prompt position is wasted work.
    Slicing logits BEFORE softmax does not change the kept positions'
    values -- softmax is computed independently per sequence position --
    re-verified unchanged by scripts/p5_kv_cache_verify.py after this
    change (perf-only, added while diagnosing P5.3's KV-cache overhead)."""
    past_len = _cache_len(past_key_values)
    position_ids = torch.arange(
        past_len, past_len + new_ids.shape[1], device=new_ids.device
    ).unsqueeze(0)
    with torch.inference_mode():
        out = model(
            input_ids=new_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )
    logits = out.logits
    if keep_last_n is not None and keep_last_n < logits.shape[1]:
        logits = logits[:, -keep_last_n:, :]
    dists = torch.softmax(logits.float(), dim=-1).squeeze(0)  # [kept_n, vocab]
    return dists, out.past_key_values


def draft_propose_cached(draft_model, seed_ids, past_key_values, gamma, temperature, generator=None):
    """Same contract as draft_propose(), but incremental."""
    tokens, dists = [], []
    cur_new = seed_ids

    for _ in range(gamma):
        step_dists, past_key_values = _forward_step(draft_model, cur_new, past_key_values, keep_last_n=1)
        probs = step_dists[-1]

        if temperature == 0:
            token = _greedy(probs.unsqueeze(0))
        else:
            token = _sample(probs.unsqueeze(0), generator)

        tokens.append(token)
        dists.append(probs)
        cur_new = torch.tensor([[token]], device=seed_ids.device)

    return tokens, dists, past_key_values


def target_verify_cached(target_model, seed_ids, draft_tokens, past_key_values):
    """Same contract as target_verify() (returns gamma+1 distributions), but
    only feeds `seed_ids` + this round's draft candidates, reusing the cache
    for everything before that."""
    gamma = len(draft_tokens)
    draft_tensor = torch.tensor([draft_tokens], device=seed_ids.device)
    new_ids = torch.cat([seed_ids, draft_tensor], dim=-1)

    dists, past_key_values = _forward_step(target_model, new_ids, past_key_values, keep_last_n=gamma + 1)
    target_dists = [dists[i] for i in range(dists.shape[0] - gamma - 1, dists.shape[0])]
    return target_dists, past_key_values


def speculative_decoding_step_cached(
    draft_model, target_model, draft_seed_ids, target_seed_ids,
    draft_cache, target_cache, gamma, temperature=1.0, generator=None,
):
    """One round, cached. Mirrors speculative_decoding_step()'s accept/
    reject/resample/bonus logic exactly (copied rather than imported so the
    verified original is never touched), except state is threaded through
    past_key_values instead of a growing input_ids tensor.

    Returns (StepResult, new_draft_cache, new_target_cache, next_draft_seed,
    next_target_seed) -- both "next_*_seed" are always just the tail token.
    """
    draft_start_len = _cache_len(draft_cache)
    target_start_len = _cache_len(target_cache)

    draft_tokens, draft_dists, draft_cache = draft_propose_cached(
        draft_model, draft_seed_ids, draft_cache, gamma, temperature, generator
    )
    target_dists, target_cache = target_verify_cached(
        target_model, target_seed_ids, draft_tokens, target_cache
    )

    draft_token_probs, target_token_probs = [], []
    n_accepted = 0
    target_greedy_token = None

    for i in range(gamma):
        token = draft_tokens[i]
        p_draft = draft_dists[i][token].item()
        p_target = target_dists[i][token].item()
        draft_token_probs.append(p_draft)
        target_token_probs.append(p_target)

        if temperature == 0:
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
            resampled_token = target_greedy_token
        else:
            p_target_reject = target_dists[n_accepted]
            p_draft_reject = draft_dists[n_accepted]
            adjusted = torch.clamp(p_target_reject - p_draft_reject, min=0.0)
            total = adjusted.sum()
            if total <= 0:
                adjusted = p_target_reject
                total = adjusted.sum()
            adjusted = adjusted / total
            resampled_token = _sample(adjusted.unsqueeze(0), generator)
        tail_token = resampled_token
    else:
        # BUGFIX (found via scripts/debug/p5_kv_cache_fake_model_trace.py, same
        # investigation as the crop-target fix above): draft_propose_cached's
        # loop samples draft_tokens[gamma-1] but never feeds it back into the
        # draft model (by design -- there is no gamma+1-th draft step), so on
        # full acceptance the draft cache is one position SHORT of prefix +
        # gamma. Feed it now, purely to keep the draft cache in sync with the
        # crop target below -- this is a real forward pass (not a generator
        # draw), so it cannot perturb sampling parity, only cache bookkeeping.
        last_draft_token = torch.tensor([[draft_tokens[-1]]], device=draft_seed_ids.device)
        _, draft_cache = _forward_step(draft_model, last_draft_token, draft_cache, keep_last_n=1)

        p_bonus = target_dists[gamma]
        if temperature == 0:
            bonus_token = _greedy(p_bonus.unsqueeze(0))
        else:
            bonus_token = _sample(p_bonus.unsqueeze(0), generator)
        tail_token = bonus_token

    # Roll both caches back to "confirmed length + n_accepted", discarding
    # anything from rejected candidates.
    # BUGFIX (found via scripts/debug/p5_kv_cache_fake_model_trace.py): this used to
    # read `_crop_cache(draft_cache, draft_start_len + n_accepted)`, missing the
    # `+ draft_seed_ids.shape[1]` term that target_cache's crop (one line below)
    # already had. On round 0 that cropped the draft cache to 0, wiping the
    # just-cached prompt; every later round then computed the draft model's
    # position_ids from a wrong (too-small) cache length, so the draft's own
    # proposals were computed from the wrong absolute position from round 1
    # onward. Silent in greedy mode (a rejection always falls back to the
    # correctly-tracked target's own greedy token, regardless of what the
    # mispositioned draft proposed) -- only visible as sampling-mode divergence
    # (p5_kv_cache_fake_model_verify.py) and, almost certainly, as a real
    # acceptance-rate/throughput hit in every real run so far.
    draft_cache = _crop_cache(draft_cache, draft_start_len + draft_seed_ids.shape[1] + n_accepted)
    target_cache = _crop_cache(target_cache, target_start_len + target_seed_ids.shape[1] + n_accepted)

    device = draft_seed_ids.device
    next_seed = torch.tensor([[tail_token]], device=device)

    result = StepResult(
        draft_tokens=draft_tokens, draft_token_probs=draft_token_probs,
        target_token_probs=target_token_probs, n_accepted=n_accepted, gamma=gamma,
        accepted_tokens=accepted_tokens, resampled_token=resampled_token, bonus_token=bonus_token,
    )
    return result, draft_cache, target_cache, next_seed, next_seed


def speculative_generate_cached(
    draft_model, target_model, tokenizer, prompt, gamma, max_new_tokens,
    temperature=1.0, generator=None, eos_token_id=None,
):
    assert (
        draft_model.config.vocab_size == target_model.config.vocab_size
    ), "draft/target vocab_size mismatch -- run the P1.0 gate first"

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    device = next(draft_model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    input_ids = prompt_ids
    draft_seed = prompt_ids
    target_seed = prompt_ids
    draft_cache = None
    target_cache = None

    steps = []
    generated = 0

    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        step_gamma = min(gamma, remaining)

        result, draft_cache, target_cache, draft_seed, target_seed = speculative_decoding_step_cached(
            draft_model, target_model, draft_seed, target_seed,
            draft_cache, target_cache, step_gamma, temperature, generator,
        )
        steps.append(result)

        tokens_to_add = result.new_tokens
        cutoff = len(tokens_to_add)
        if eos_token_id is not None and eos_token_id in tokens_to_add:
            cutoff = min(cutoff, tokens_to_add.index(eos_token_id) + 1)
        cutoff = min(cutoff, remaining)
        tokens_to_add = tokens_to_add[:cutoff]

        input_ids = torch.cat([input_ids, torch.tensor([tokens_to_add], device=device)], dim=-1)
        generated += len(tokens_to_add)

        if eos_token_id is not None and eos_token_id in tokens_to_add:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return {"input_ids": input_ids, "output_text": output_text, "steps": steps}
