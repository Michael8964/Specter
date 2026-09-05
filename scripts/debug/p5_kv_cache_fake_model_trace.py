"""
P5.3 prerequisite -- round-by-round trace to find WHERE cached vs non-cached
diverge in sampling mode, using the noise-free PositionOnlyFakeModel.

p5_kv_cache_fake_model_diagnose.py proved the fake model's logits are
bit-identical regardless of batching -- so p5_kv_cache_fake_model_verify.py's
sampling-mode failures (greedy 72/72 pass, sampling 72/72 fail) are a REAL
logic difference, not floating-point noise. Since greedy mode never touches
the random generator (argmax needs no draw) while sampling mode does
(torch.multinomial / torch.rand at several points per round), the leading
hypothesis is that the two implementations consume the SAME shared generator
a DIFFERENT number of times or in a different order -- not that any position
or probability is computed wrong. This script drives both implementations one
round at a time (gamma=1, simplest case) and prints every generator-relevant
quantity so the first divergence is visible directly, not inferred.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.speculative_decode import (
    _greedy,
    _sample,
    draft_propose,
    speculative_decoding_step,
    target_verify,
)
from src.speculative_decode_kv import (
    speculative_decoding_step_cached,
)


class FakeCache:
    def __init__(self):
        self._len = 0

    def get_seq_length(self):
        return self._len

    def crop(self, delta):
        assert delta <= 0
        self._len += delta
        assert self._len >= 0

    def _grow(self, n):
        self._len += n


class PositionOnlyFakeModel:
    def __init__(self, vocab_size=32, phase=0.0):
        self.vocab_size = vocab_size
        self.phase = phase
        self.config = type("cfg", (), {"vocab_size": vocab_size})()

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, past_key_values=None, position_ids=None, use_cache=None):
        L = input_ids.shape[1]
        positions = torch.arange(0, L) if position_ids is None else position_ids[0]
        v = torch.arange(self.vocab_size).float()
        logits = torch.zeros(1, L, self.vocab_size)
        for i in range(L):
            p = float(positions[i])
            logits[0, i] = torch.sin(v * 0.9 + p * 0.4 + self.phase) * 2.0 + torch.cos(v * 0.3 - p * 0.2)
        new_past = past_key_values
        if use_cache:
            if new_past is None:
                new_past = FakeCache()
            new_past._grow(L)
        return type("Out", (), {"logits": logits, "past_key_values": new_past})()


def main():
    gamma = 1
    prompt_ids = torch.arange(1, 4).unsqueeze(0)  # 3-token "prompt", positions 0,1,2
    max_rounds = 8

    draft = PositionOnlyFakeModel(phase=0.0)
    target = PositionOnlyFakeModel(phase=0.6)

    # -------- non-cached reference, one round at a time -----------------
    gen_ref = torch.Generator()
    gen_ref.manual_seed(0)
    ctx = prompt_ids.clone()
    ref_tokens = []

    # -------- cached, one round at a time --------------------------------
    gen_kv = torch.Generator()
    gen_kv.manual_seed(0)
    draft_cache = None
    target_cache = None
    draft_seed = prompt_ids.clone()
    target_seed = prompt_ids.clone()
    kv_tokens = []

    print("=" * 100)
    print(f"{'round':>5} | {'ref new_tokens':<20} {'ref draft_probs':<12} {'ref accept':<20} "
          f"| {'kv new_tokens':<20} {'kv draft_probs':<12} {'match'}")
    print("=" * 100)

    for round_i in range(max_rounds):
        result_ref = speculative_decoding_step(draft, target, ctx, gamma, temperature=1.0, generator=gen_ref)
        ctx = torch.cat([ctx, torch.tensor([result_ref.new_tokens])], dim=-1)
        ref_tokens.extend(result_ref.new_tokens)

        result_kv, draft_cache, target_cache, draft_seed, target_seed = speculative_decoding_step_cached(
            draft, target, draft_seed, target_seed, draft_cache, target_cache, gamma, temperature=1.0, generator=gen_kv,
        )
        kv_tokens.extend(result_kv.new_tokens)

        match = "OK" if result_ref.new_tokens == result_kv.new_tokens else "DIVERGE <---"
        print(f"{round_i:>5} | {str(result_ref.new_tokens):<20} "
              f"{str([round(x,4) for x in result_ref.draft_token_probs]):<12} "
              f"n_acc={result_ref.n_accepted} bonus={result_ref.bonus_token} resamp={result_ref.resampled_token}"
              f" | {str(result_kv.new_tokens):<20} "
              f"{str([round(x,4) for x in result_kv.draft_token_probs]):<12} {match}")

        if match == "DIVERGE <---":
            print("\n--- first divergence details ---")
            print(f"ref:  draft_tokens={result_ref.draft_tokens} draft_probs={result_ref.draft_token_probs} "
                  f"target_probs={result_ref.target_token_probs}")
            print(f"kv :  draft_tokens={result_kv.draft_tokens} draft_probs={result_kv.draft_token_probs} "
                  f"target_probs={result_kv.target_token_probs}")
            break

    print("=" * 100)
    print(f"ref full sequence: {ref_tokens}")
    print(f"kv  full sequence: {kv_tokens}")


if __name__ == "__main__":
    main()
