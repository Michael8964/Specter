"""
P5.0-P5.1 -- GammaTune: an adaptive controller for gamma (Kim et al. 2025).

project_plan_v9.md Appendix A.2 gives the cost model this controller is
implicitly trying to minimize:

    cost = N / (alpha*gamma + 1) * (c + gamma) * T_draft

where N is the number of tokens to generate, (alpha*gamma + 1) approximates
the expected tokens produced per round (see scripts/p1_gamma_scan.py for the
empirical version of this curve), and (c + gamma) is the expected wall-clock
cost of one round in units of T_draft (c = T_target / T_draft, the ratio of
one target forward pass's cost to one draft forward pass's cost). A fixed
gamma can only sit at one point on this cost curve; GammaTune instead adjusts
gamma round by round based on what actually happened last round, trying to
track the curve's minimum as it shifts with how "easy" or "hard" the current
context is for the draft model.

The update rule (project_plan_v9.md Appendix A.2, Algorithm 1):

    if n_accepted == gamma_used:      # every candidate survived -- possibly leaving room to grow
        gamma <- n_accepted + delta
    else:                              # a rejection happened
        gamma_bar <- clip(gamma_min, gamma_max, (1-eta)*gamma_bar + eta*n_accepted)
        gamma <- ceil(gamma_bar)

The asymmetry is deliberate: a full-acceptance round immediately probes a
larger gamma (fast expansion, no smoothing -- gamma_bar is left untouched so
this probe doesn't get baked into the long-run average until it's actually
confirmed by a later round). A round with any rejection instead blends into
an exponential moving average (EMA) of gamma_bar, so a single bad round
doesn't cause gamma to crash to whatever n_accepted happened to be -- it eases
down smoothly. "Expand fast, shrink smoothly" is exactly the behavior
P5.1's fluctuation-scenario test (still to come) is meant to confirm.
"""

import math


class GammaTuneController:
    """
    Stateful controller: call update() once after every round of speculative
    decoding, with that round's actual n_accepted and the gamma that round
    actually used, and read the new value to use for the NEXT round from
    self.gamma (or the update() return value -- same thing).
    """

    def __init__(self, gamma_init=3, gamma_min=1, gamma_max=10, eta=0.3, delta=2):
        self.gamma = gamma_init
        self.gamma_bar = float(gamma_init)  # smoothed/EMA baseline estimate
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.eta = eta
        self.delta = delta

    def _clip(self, value):
        return min(self.gamma_max, max(self.gamma_min, value))

    def update(self, n_accepted, gamma_used):
        """
        n_accepted: how many candidates were actually accepted this round
            (0..gamma_used).
        gamma_used: the gamma value that was actually used for this round
            (may be smaller than self.gamma if the token budget capped it --
            always compare against what really happened, not the controller's
            uncapped internal target).
        """
        if n_accepted == gamma_used:
            # Full acceptance: fast, un-smoothed probe upward.
            self.gamma = self._clip(n_accepted + self.delta)
        else:
            # Partial acceptance: smooth fallback via EMA, not a hard reset.
            self.gamma_bar = self._clip((1 - self.eta) * self.gamma_bar + self.eta * n_accepted)
            self.gamma = math.ceil(self.gamma_bar)

        return self.gamma
