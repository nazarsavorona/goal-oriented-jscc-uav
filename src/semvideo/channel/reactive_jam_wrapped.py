"""Channel wrapper that composes a base channel with post-channel reactive jamming.

Train-time augmentation that mirrors :class:`ChannelWithErasure` but applies
DRFM-style reactive jamming instead of Bernoulli erasure. The two-state
reactive automaton (detect-then-hold) follows the threat model described by
Li et al. 2025 (arXiv:2502.02385, "Achieving Hiding and Smart Anti-Jamming
Communication: A Parallel DRL Approach against Moving Reactive Jammer";
defender-side paper, jammer model in threat-model section), with classical
DRFM grounding from Schleher 1986 and Pace 2009.
Used by the P80/P81 family: a single 8-ep stage-3 continuation of P46 with
reactive-jam in the training distribution that – if the erasure-aug pattern
transfers – should close the reactive-jam hard-negative (cls=0.637
@ JNR=10 on P68) at training-time cost only.

Usage:
    net = SwinJSCC(...)
    net.channel = ChannelWithReactiveJam(
        net.channel,
        jam_prob=0.5,            # fraction of minibatches that get jammed
        jnr_min_db=5.0, jnr_max_db=15.0,  # JNR per jammed minibatch
    )

Acts like ``Dropout``: jamming applied during ``self.training`` only by default
(``always_apply=False``), so eval-time numbers reproduce. Set
``always_apply=True`` for an eval-only jamming sweep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from semvideo.channel.jamming_reactive import ReactiveJammer


class ChannelWithReactiveJam(nn.Module):
    """Wrap a SwinJSCC ``Channel`` with optional post-channel reactive jamming.

    The base channel runs first (AWGN / Rayleigh / etc.), then surviving
    symbols pass through a ``ReactiveJammer`` with JNR sampled uniformly
    from ``[jnr_min_db, jnr_max_db]`` per forward pass – but only with
    probability ``jam_prob``, so a fraction of training minibatches see a
    clean channel and a fraction see jam.

    Args:
        base_channel: The SwinJSCC ``Channel`` instance to wrap.
        jam_prob: Probability per forward of applying jam (0.0 disables).
        jnr_min_db, jnr_max_db: Uniform JNR range when jam is applied.
        detect_threshold: Forwarded to ``ReactiveJammer``. Default 0.5 =
            "above-average symbol energy".
        react_latency: Forwarded to ``ReactiveJammer``. Default 8 symbols.
        hold_duration: Forwarded to ``ReactiveJammer``. Default 64 symbols.
        always_apply: If True, jamming runs at eval too (for eval sweeps).
            Default False (training-only, like Dropout).
    """

    def __init__(
        self,
        base_channel: nn.Module,
        jam_prob: float = 0.5,
        jnr_min_db: float = 5.0,
        jnr_max_db: float = 15.0,
        detect_threshold: float = 0.5,
        react_latency: int = 8,
        hold_duration: int = 64,
        always_apply: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= jam_prob <= 1.0:
            raise ValueError(f"jam_prob must be in [0, 1], got {jam_prob}")
        if jnr_min_db > jnr_max_db:
            raise ValueError(f"jnr_min_db ({jnr_min_db}) > jnr_max_db ({jnr_max_db})")
        self.base_channel = base_channel
        self.jam_prob = float(jam_prob)
        self.jnr_min_db = float(jnr_min_db)
        self.jnr_max_db = float(jnr_max_db)
        self.detect_threshold = float(detect_threshold)
        self.react_latency = int(react_latency)
        self.hold_duration = int(hold_duration)
        self.always_apply = bool(always_apply)
        # Reusable jammer. JNR is mutated per-forward (init value irrelevant).
        # NOTE: we keep ``symbol_axis=-1`` so
        # the automaton runs along whatever axis the wrapped channel feeds in.
        # For SwinJSCC w/SA the latent is ``(B, N=1024, C=96)`` and the symbol
        # axis is the inner C dimension. The DRFM automaton therefore evolves
        # across 96 per-token features rather than across N=1024 token
        # positions. This is still a form of *structured energy-targeted
        # noise* useful as a training augmentation, but it is NOT the
        # "DRFM-over-spatial-tokens" model that arXiv:2502.02385 describes.
        # It is therefore described as "energy-targeted feature-channel
        # jamming during training" rather than a strict DRFM.
        self.jammer = ReactiveJammer(
            jnr_db=jnr_max_db,
            detect_threshold=detect_threshold,
            react_latency=react_latency,
            hold_duration=hold_duration,
        )

    def _should_apply(self) -> bool:
        if self.jam_prob <= 0.0:
            return False
        if self.always_apply:
            return True
        if not self.training:
            return False
        # Bernoulli per-forward gate.
        return bool(torch.rand(1).item() < self.jam_prob)

    def forward(
        self,
        feature: torch.Tensor,
        chan_param: float,
        avg_pwr: bool = False,
    ) -> torch.Tensor:
        """Channel-pass with optional reactive jamming."""
        noisy = self.base_channel.forward(feature, chan_param, avg_pwr)
        if not self._should_apply():
            return noisy
        jnr = float(torch.empty(1).uniform_(self.jnr_min_db, self.jnr_max_db).item())
        self.jammer.jnr_db = jnr
        # ReactiveJammer expects (B, N) symbol-axis shape; SwinJSCC channel
        # outputs (B, N, C) latent. The jammer's forward() accepts arbitrary
        # shape and treats symbol_axis=-1 by default – match that contract.
        return self.jammer(noisy)

    def extra_repr(self) -> str:
        return (
            f"jam_prob={self.jam_prob}, jnr=U[{self.jnr_min_db}, "
            f"{self.jnr_max_db}], detect={self.detect_threshold}, "
            f"latency={self.react_latency}, hold={self.hold_duration}, "
            f"always_apply={self.always_apply}"
        )
