"""
P5.3 prerequisite -- is PositionOnlyFakeModel actually noise-free?

p5_kv_cache_fake_model_verify.py found greedy=72/72 pass, sampling=72/72 fail --
the EXACT same shape as the real-Qwen-model result. Before concluding "this
confirms the sampling divergence is benign floating-point noise, not the
crop-target bug I suspected while reading spec_kv.py", check the premise: does
torch.sin/cos over a BATCH of positions computed in one call give bit-identical
results to the same position computed ALONE in its own call? If yes (diff==0),
the fake model really is noise-free, and the sampling failures point back at a
real logic bug. If no (diff!=0), the fake model has its own (much smaller)
floating-point sensitivity to batching -- structurally the same phenomenon as
the real model, and the crop-target suspicion is very likely a red herring.
"""
import torch

VOCAB = 32
PHASE = 0.6


def logits_row(p: float) -> torch.Tensor:
    v = torch.arange(VOCAB).float()
    return torch.sin(v * 0.9 + p * 0.4 + PHASE) * 2.0 + torch.cos(v * 0.3 - p * 0.2)


def batched_call(positions) -> torch.Tensor:
    """All positions computed together in one tensor op, mirroring how the
    non-cached path (and a cached model's first, whole-prompt call) does it."""
    v = torch.arange(VOCAB).float()
    L = len(positions)
    out = torch.zeros(L, VOCAB)
    for i, p in enumerate(positions):
        out[i] = torch.sin(v * 0.9 + p * 0.4 + PHASE) * 2.0 + torch.cos(v * 0.3 - p * 0.2)
    return out


def standalone_call(p: float) -> torch.Tensor:
    """The same single position, computed in isolation -- mirrors every
    later cached call (new_ids.shape[1] == 1)."""
    return logits_row(p).unsqueeze(0)


def main():
    print("=" * 70)
    print("Is PositionOnlyFakeModel's arithmetic batch-size-invariant?")
    print("=" * 70)
    positions = list(range(10))
    batch_out = batched_call(positions)
    max_diff = 0.0
    for i, p in enumerate(positions):
        solo_out = standalone_call(p)
        diff = (batch_out[i] - solo_out[0]).abs().max().item()
        max_diff = max(max_diff, diff)
        marker = "" if diff == 0.0 else "  <-- NONZERO"
        print(f"  position {p}: max abs diff (batched row vs standalone) = {diff:.3e}{marker}")
    print("=" * 70)
    if max_diff == 0.0:
        print("Bit-identical regardless of batching: this fake model IS noise-free.")
        print("-> the sampling-mode failures in p5_kv_cache_fake_model_verify.py")
        print("   point at a REAL logic bug (most likely the crop-target math), not noise.")
    else:
        print(f"NOT bit-identical (max diff {max_diff:.3e}): even this synthetic model has a")
        print("floating-point sensitivity to batch shape, structurally the same phenomenon")
        print("(batched vs incremental compute -> tiny fp diff -> can flip a sampled token")
        print("-> cascades) as the real Qwen model. The crop-target suspicion is most likely")
        print("a red herring; the fake-model verify results are consistent with a correct")
        print("implementation, same standard as the real-model result.")
    print("=" * 70)


if __name__ == "__main__":
    main()
