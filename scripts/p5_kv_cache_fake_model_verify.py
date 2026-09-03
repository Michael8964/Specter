"""
P5.3 prerequisite -- deterministic (position-only) fake-model cache verification.

Why this exists: scripts/p5_kv_cache_verify.py uses REAL Qwen models, so its
sampling-mode mismatches could only be explained by *inference*, not proof
("we diagnosed it as floating-point noise" -- p5_kv_cache_diagnose.py -- but a
real logic bug affecting position bookkeeping could in principle hide behind
that same excuse). It's also blind to one specific class of bug: Qwen uses
RoPE, a RELATIVE positional encoding -- if our cache's internal notion of
"how many tokens are cached" ever drifted from the true absolute position by
a CONSTANT amount, RoPE's attention pattern (which only depends on the
difference between two positions) could still come out correct, silently
absorbing a real position-tracking bug that a real-model test could never see.

This script's FakeModel logits depend on ABSOLUTE position directly (no RoPE,
no learned structure) -- torch.sin/cos of (vocab_index, position). Cached and
non-cached forward passes over the SAME true position are then bit-identical
by construction (down to the last float), so:
  - ANY mismatch, even in greedy mode, is proof of a real cache/position bug
    -- not floating-point noise, because there IS no floating-point noise
    source here (the fake model's arithmetic is trivial and stable).
  - A position-tracking bug that RoPE might mask on a real model CANNOT hide
    here, because absolute position is exactly what determines the output.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.speculative_decode import speculative_generate
from src.speculative_decode_kv import speculative_generate_cached


class FakeCache:
    """Length-only stand-in for a real Cache: tracks how many positions have
    been fed, supports the two operations _cache_len/_crop_cache actually use."""

    def __init__(self):
        self._len = 0

    def get_seq_length(self):
        return self._len

    def crop(self, delta):
        assert delta <= 0, f"speculative_decode_kv.py should only ever shrink a cache (got delta={delta})"
        self._len += delta
        assert self._len >= 0, "cache length went negative -- cropped past what was ever cached"

    def _grow(self, n):
        self._len += n


class PositionOnlyFakeModel:
    """logits at absolute position p depend ONLY on p (and this instance's
    `phase`, to make draft/target disagree sometimes) -- never on token
    identity, never on whether a cache was used. A cached incremental forward
    and a full-recompute forward over the same true position are therefore
    bit-identical, by construction, not by luck."""

    def __init__(self, vocab_size=32, phase=0.0):
        self.vocab_size = vocab_size
        self.phase = phase
        self.config = type("cfg", (), {"vocab_size": vocab_size})()

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, past_key_values=None, position_ids=None, use_cache=None):
        L = input_ids.shape[1]
        if position_ids is None:
            # non-cached calling convention (speculative_decode.py): input_ids
            # is always the FULL sequence from position 0.
            positions = torch.arange(0, L)
        else:
            positions = position_ids[0]
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


class FakeTokenizer:
    eos_token_id = None

    def __call__(self, prompt, return_tensors="pt"):
        n = 3 + (len(prompt) % 5)  # short, deterministic, prompt-length-dependent
        return {"input_ids": torch.arange(1, 1 + n).unsqueeze(0)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


PROMPTS = ["hello", "a somewhat longer prompt to stress round 1", "kv cache test", "x"]
GAMMAS = [1, 3, 5]
SEEDS = [0, 1, 2]


def run_pair(draft, target, tok, prompt, gamma, temperature, seed):
    g1 = torch.manual_seed(seed)
    ref = speculative_generate(
        draft, target, tok, prompt, gamma=gamma, max_new_tokens=24, temperature=temperature, generator=g1,
    )
    g2 = torch.manual_seed(seed)
    cached = speculative_generate_cached(
        draft, target, tok, prompt, gamma=gamma, max_new_tokens=24, temperature=temperature, generator=g2,
    )
    return ref["input_ids"][0].tolist(), cached["input_ids"][0].tolist()


def main():
    print("=" * 70)
    print("Specter -- deterministic fake-model KV-cache verification")
    print("(no real neural net, no floating-point noise -- any mismatch here")
    print(" is a real logic bug, not something we can wave away as 'expected')")
    print("=" * 70)

    total = 0
    failures = []

    for temperature, label in [(0.0, "greedy"), (1.0, "sampling")]:
        # different phase per (temperature, seed) draft/target pair so we cover
        # a range of agreement levels, including low-acceptance stress
        for phase_gap, gap_label in [(0.6, "normal"), (2.4, "high-rejection-stress")]:
            draft = PositionOnlyFakeModel(phase=0.0)
            target = PositionOnlyFakeModel(phase=phase_gap)
            for seed in SEEDS:
                for gamma in GAMMAS:
                    for prompt in PROMPTS:
                        total += 1
                        tok = FakeTokenizer()
                        ref_ids, cached_ids = run_pair(draft, target, tok, prompt, gamma, temperature, seed)
                        match = ref_ids == cached_ids
                        if not match:
                            first_diff = next(
                                (i for i, (a, b) in enumerate(zip(ref_ids, cached_ids)) if a != b),
                                min(len(ref_ids), len(cached_ids)),
                            )
                            failures.append((label, gap_label, seed, gamma, prompt, first_diff, ref_ids, cached_ids))

    print(f"\nRan {total} cases (greedy + sampling, 2 phase gaps, {len(SEEDS)} seeds, "
          f"{len(GAMMAS)} gammas, {len(PROMPTS)} prompts)")
    print(f"RESULT: {total - len(failures)}/{total} passed")

    if failures:
        print("\nFAILURES (showing up to 5):")
        for label, gap_label, seed, gamma, prompt, first_diff, ref_ids, cached_ids in failures[:5]:
            print(f"  mode={label} gap={gap_label} seed={seed} gamma={gamma} prompt={prompt!r}")
            print(f"    first diff at index {first_diff}")
            print(f"    ref:    {ref_ids}")
            print(f"    cached: {cached_ids}")
        greedy_fail = [f for f in failures if f[0] == "greedy"]
        if greedy_fail:
            print("\nGREEDY MISMATCH WITH A DETERMINISTIC MODEL: this is a real bug in")
            print("speculative_decode_kv.py's cache/position bookkeeping (most likely the")
            print("crop-target math), NOT floating-point noise -- there is no noise source")
            print("in this fake model. Root-cause and fix before trusting any timing number.")
        else:
            print("\nOnly sampling-mode cases failed (greedy all passed) -- consistent with")
            print("the same benign floating-point... wait, this fake model has none. A")
            print("sampling-only failure here would still be suspicious and worth a look.")
    else:
        print("\nAll cases matched exactly, including sampling mode. Since this fake model")
        print("has no floating-point noise source, this is a STRONGER correctness signal")
        print("than the real-model test: the cache/position bookkeeping in")
        print("speculative_decode_kv.py is verified correct by construction, not by")
        print("statistical argument.")
    print("=" * 70)


if __name__ == "__main__":
    main()
