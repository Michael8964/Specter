"""
Hand-written AWQ-style activation-aware weight quantization.

See notes/project_plan_v9.md §7 "支柱2：AWQ 量化" (P2.0/P2.1) and appendix A.3
for the algorithm this implements, and §8's decision table ("量化自研对象"
row: AWQ over GPTQ) for why AWQ (not GPTQ) is the hand-implemented method.
This module implements the
per-channel scale search and fake-quantize routines from scratch; it does not
call into AutoAWQ / llm-awq. AutoAWQ's quantizer.py was read only to confirm
the grid-search structure (uniform grid over a scaling exponent alpha) before
writing this, per the project's non-reuse rule for this specific piece.

Two design points worth stating explicitly because the plan calls out both as
the actual implementation difficulty (not the underlying math):

1. Activation-stat hooks are attached to each target nn.Linear's *input*
   (forward pre-hook), not its output. AWQ's channel saliency is defined over
   the layer's input feature dimension -- that's the axis the per-channel
   scale `s` is indexed by, and the axis the group-wise fake-quantize groups
   are cut along.
2. Fake-quantize follows round -> clamp -> dequantize, in that order, on a
   per-group (contiguous chunk of in_features, size `group_size`) basis with
   an asymmetric (min/max + zero-point) quantization grid. Clamping before
   rounding, or dequantizing with the wrong zero-point sign, silently produces
   a plausible-looking but wrong weight tensor -- it doesn't crash, it just
   makes perplexity worse in a way that's hard to distinguish from "AWQ isn't
   working" without a step-by-step check against a hand-computed toy example
   (see notes/project_plan_v9.md 附录A.3).

Only nn.Linear modules inside the decoder's transformer blocks
(model.model.layers[*].self_attn.{q,k,v,o}_proj and
model.model.layers[*].mlp.{gate,up,down}_proj) are quantized. Embeddings and
lm_head are left at full precision, matching common AWQ/GPTQ practice --
the embedding table is a lookup (no matmul quantization benefit) and lm_head
errors compound directly into every output logit.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn

TARGET_SUBMODULE_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def iter_target_linears(model):
    """Yield (layer_idx, submodule_name, nn.Linear) for every quantization target."""
    for layer_idx, layer in enumerate(model.model.layers):
        for name in TARGET_SUBMODULE_NAMES:
            module = layer
            for part in name.split("."):
                module = getattr(module, part)
            assert isinstance(module, nn.Linear), f"{name} is not nn.Linear"
            yield layer_idx, name, module


@dataclass
class ActivationStats:
    """Running per-input-channel |activation| stats for one nn.Linear, plus a
    reservoir sample of raw activation rows used later for the grid-search
    loss (the loss must be measured against real activations, not just their
    mean -- the mean alone can't tell two candidate scales apart)."""

    abs_sum: torch.Tensor  # [in_features], running sum of |x|
    n_seen: int = 0
    sample_rows: list = field(default_factory=list)
    n_sample_rows: int = 0  # total rows actually stored in sample_rows (not len(sample_rows)!)
    max_sample_rows: int = 256

    def update(self, x: torch.Tensor):
        # x: [n_tokens, in_features], already flattened over batch/seq dims.
        x = x.detach().to(torch.float32)
        if self.abs_sum.device != x.device:
            self.abs_sum = self.abs_sum.to(x.device)
        self.abs_sum += x.abs().sum(dim=0)
        self.n_seen += x.shape[0]
        # Reservoir-ish sample: cheap approximation (random subset per call is
        # fine here since calibration batches are already i.i.d. draws from
        # the calibration set, not sequential/correlated data). Cap tracked by
        # total ROW count (n_sample_rows), not number of append() calls -- a
        # cap on len(sample_rows) alone lets total stored rows grow far past
        # max_sample_rows (each call can add up to max_sample_rows rows), and
        # is what caused an MPS OOM (~30GB) when this was first tried against
        # the real 3B model with 128 calibration sequences.
        need = self.max_sample_rows - self.n_sample_rows
        if need > 0:
            idx = torch.randperm(x.shape[0])[:need]
            self.sample_rows.append(x[idx].clone())
            self.n_sample_rows += idx.shape[0]

    def mean_abs(self) -> torch.Tensor:
        return self.abs_sum / max(self.n_seen, 1)

    def sample(self) -> torch.Tensor:
        if not self.sample_rows:
            raise RuntimeError("no activation samples collected -- calibration did not run")
        return torch.cat(self.sample_rows, dim=0)


class ActivationCollector:
    """Forward-pre-hook based collector for every target nn.Linear's input."""

    def __init__(self, model):
        self.model = model
        self.stats: dict[str, ActivationStats] = {}
        self._handles = []

    def _make_hook(self, key: str, in_features: int):
        def hook(module, args):
            x = args[0]
            x = x.reshape(-1, in_features)
            if key not in self.stats:
                self.stats[key] = ActivationStats(abs_sum=torch.zeros(in_features, dtype=torch.float32))
            self.stats[key].update(x)
        return hook

    def __enter__(self):
        for layer_idx, name, module in iter_target_linears(self.model):
            key = f"{layer_idx}.{name}"
            handle = module.register_forward_pre_hook(self._make_hook(key, module.in_features))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles = []


def fake_quantize_groupwise(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Per-group asymmetric fake-quantize: round -> clamp -> dequantize.

    w: [out_features, in_features], in_features must be divisible by group_size.
    Returns a same-shape, same-dtype tensor of dequantized (still float) values
    -- this is "fake" quantization (simulates the numerical effect of int
    storage using plain float math), not a real packed int4 tensor. Real
    int4 kernels aren't reachable from MPS (see notes/project_plan_v9.md
    §9.1 风险B); this is what §9.1 calls the Mac-stage-appropriate substitute.
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} not divisible by group_size={group_size}"
    )
    orig_dtype = w.dtype
    w32 = w.to(torch.float32).reshape(out_features, in_features // group_size, group_size)

    w_min = w32.amin(dim=-1, keepdim=True)
    w_max = w32.amax(dim=-1, keepdim=True)
    max_int = 2 ** bits - 1

    scale = (w_max - w_min).clamp(min=1e-8) / max_int
    zero_point = torch.round(-w_min / scale)

    w_int = torch.round(w32 / scale + zero_point)
    w_int = w_int.clamp(0, max_int)

    w_dequant = (w_int - zero_point) * scale
    return w_dequant.reshape(out_features, in_features).to(orig_dtype)


def _grid_search_scale(
    linear: nn.Linear,
    stats: ActivationStats,
    bits: int,
    group_size: int,
    n_grid: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Search alpha in a uniform grid over [0, 1], scale s = mean_abs(x)^alpha,
    normalized so W*s and W/s stay numerically well-scaled. Returns
    (best_scale, best_fake_quantized_weight, best_loss)."""
    device = linear.weight.device
    W = linear.weight.detach().to(torch.float32)  # [out, in]
    saliency = stats.mean_abs().to(device).clamp(min=eps)
    X_sample = stats.sample().to(device)  # [n_rows, in]
    Y_ref = X_sample @ W.T  # unquantized reference output, [n_rows, out]

    best_loss = float("inf")
    best_scale: torch.Tensor | None = None
    best_wq: torch.Tensor | None = None

    for i in range(n_grid):
        alpha = i / (n_grid - 1)
        s = saliency.pow(alpha)
        # Normalize by geometric mean so W*s / X/s don't drift in magnitude
        # across alpha values -- pure bookkeeping, doesn't change the ratio
        # W*s dot X/s = W dot X that the scale-invariance relies on.
        s = s / torch.sqrt(s.max() * s.min())

        W_scaled = W * s.unsqueeze(0)  # broadcast over out_features
        W_q_scaled = fake_quantize_groupwise(W_scaled, bits=bits, group_size=group_size)
        W_hat = W_q_scaled / s.unsqueeze(0)  # reconstructed approx of W

        Y_hat = X_sample @ W_hat.T
        loss = torch.mean((Y_hat - Y_ref) ** 2).item()

        if loss < best_loss:
            best_loss = loss
            best_scale = s
            best_wq = W_hat

    assert best_scale is not None and best_wq is not None, "n_grid must be >= 1"
    return best_scale, best_wq.to(linear.weight.dtype), best_loss


def quantize_model_awq(
    model,
    calib_input_ids: list[torch.Tensor],
    bits: int = 4,
    group_size: int = 128,
    n_grid: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run P2.0 (activation stats) + P2.1 (scale search + fake-quantize) end
    to end, mutating `model`'s target Linear weights in place.

    calib_input_ids: list of 1D LongTensors (already tokenized, on the
    model's device), one per calibration sequence.

    Returns a per-layer diagnostic dict (best alpha's implied loss, etc.) so
    callers can sanity-check that the search actually moved (e.g. AWQ loss
    should not be worse than the alpha=0 / plain-RTN loss).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    with ActivationCollector(model) as collector:
        with torch.no_grad():
            for ids in calib_input_ids:
                model(ids.unsqueeze(0))

    diagnostics = {}
    linears = list(iter_target_linears(model))
    for i, (layer_idx, name, module) in enumerate(linears):
        key = f"{layer_idx}.{name}"
        stats = collector.stats.get(key)
        if stats is None or stats.n_seen == 0:
            raise RuntimeError(f"no activations collected for {key} -- hook did not fire")
        scale, w_hat, loss = _grid_search_scale(module, stats, bits, group_size, n_grid)
        with torch.no_grad():
            module.weight.copy_(w_hat)
        diagnostics[key] = {"best_loss": loss}
        if verbose and (i + 1) % 32 == 0:
            print(f"  quantized {i + 1}/{len(linears)} linears...")

    return diagnostics
