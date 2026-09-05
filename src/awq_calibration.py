"""
P2.0 -- AWQ prerequisite: per-input-channel activation statistics.

AWQ (Lin et al. 2023, "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration") does not treat every weight channel as
equally important to quantize precisely. Its core idea (implemented in
P2.1, not here): find one scaling factor PER INPUT CHANNEL of each Linear
layer's weight matrix, so that channels the model relies on heavily keep
more effective precision after quantization, at the cost of channels it
barely uses. "Relies on heavily" is judged by activation magnitude, not
by the weight values themselves -- a channel whose incoming activations
are consistently large is a "salient" channel.

This module collects exactly that signal: for every nn.Linear submodule
in a model, the per-input-channel mean and max absolute activation value,
accumulated across a calibration dataset. P2.1 will consume this output
to compute the actual scaling factors.

CRITICAL DETAIL (flagged explicitly in project_plan_v9.md P2.1's
"workload correction" note): the hook must capture the Linear layer's
INPUT, not its output. A forward hook's `output` argument is what AWQ
scaling is protecting the precision OF (indirectly, via the weights) --
but the signal that decides which channels matter is what flows IN. This
is easy to get backwards, since `register_forward_hook`'s callback
signature is `(module, input, output)` and it is tempting to just use
whichever one is at hand.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ActivationStats:
    """Per-Linear-layer accumulator. `sum_abs` and `max_abs` are running
    (not yet finalized) totals; call `.mean_abs()` to get the actual
    per-channel mean, which needs the total sample count to divide by."""

    in_features: int
    sum_abs: torch.Tensor  # [in_features], running sum of per-sample |activation|
    max_abs: torch.Tensor  # [in_features], running elementwise max
    n_samples: int = 0     # total number of individual activation rows seen

    def update(self, x: torch.Tensor):
        """x: [..., in_features] -- any number of leading dims (batch,
        sequence length, etc.), flattened internally into one row per
        individual activation vector."""
        x = x.detach().float().reshape(-1, self.in_features)
        self.sum_abs += x.abs().sum(dim=0)
        self.max_abs = torch.maximum(self.max_abs, x.abs().max(dim=0).values)
        self.n_samples += x.shape[0]

    def mean_abs(self) -> torch.Tensor:
        if self.n_samples == 0:
            raise ValueError("no samples accumulated yet")
        return self.sum_abs / self.n_samples


def register_activation_hooks(model: nn.Module) -> tuple[list, dict]:
    """Register a forward hook on every nn.Linear submodule that updates
    an ActivationStats accumulator from that layer's INPUT tensor.

    Returns (handles, stats): `handles` is a list of hook handles (call
    .remove() on each when calibration is done -- see remove_hooks());
    `stats` maps the module's fully-qualified name (from
    model.named_modules()) to its ActivationStats accumulator, growing in
    place as more calibration batches are run through the model.
    """
    stats: dict[str, ActivationStats] = {}
    handles = []

    def make_hook(name: str, in_features: int):
        def hook(module, inputs, output):
            # inputs is a tuple (positional args to forward()); a Linear
            # layer's forward(input) takes exactly one, so inputs[0] is
            # the activation tensor flowing INTO this layer -- NOT
            # `output`, which is what this layer produces (see module
            # docstring: AWQ's salient-channel signal is about what flows
            # in, not what comes out).
            x = inputs[0]
            if name not in stats:
                # BUGFIX (found running the real P2.0 collector against
                # Qwen2.5-1.5B-Instruct on MPS, after the CPU-only toy
                # verification in p2_awq_stats_verify.py passed 7/7 without
                # catching it): torch.zeros(in_features) with no device
                # argument defaults to CPU. On CPU that happens to match
                # the incoming activation tensor's device, so the toy test
                # never noticed. On a real model running on mps:0, the
                # accumulator stayed on CPU while `x` arrived on MPS, and
                # `self.sum_abs += x.abs().sum(dim=0)` inside update()
                # raised "Expected all tensors to be on the same device".
                # Fix: allocate the accumulator on whatever device the
                # activation itself is on.
                stats[name] = ActivationStats(
                    in_features=in_features,
                    sum_abs=torch.zeros(in_features, device=x.device),
                    max_abs=torch.zeros(in_features, device=x.device),
                )
            stats[name].update(x)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(make_hook(name, module.in_features))
            handles.append(handle)

    return handles, stats


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
