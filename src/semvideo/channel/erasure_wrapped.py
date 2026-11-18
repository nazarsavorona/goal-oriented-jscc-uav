"""Channel wrapper that composes a base channel with post-channel Bernoulli erasure.

Replaces the train-time monkey-patch on ``SwinJSCC.feature_pass_channel`` in
``scripts/train_swinjscc_dji.py`` (lines 405-420 of the pre-refactor version).
The monkey-patch broke ``state_dict()`` round-trip, blocked TensorRT export,
and drifted from the eval-time patch in the evaluation scripts.

Usage:
    net = SwinJSCC(...)
    net.channel = ChannelWithErasure(net.channel, p_max=0.40)

Acts like ``Dropout``: erasure is applied during training (``self.training``)
and skipped at eval, so the published P66/P68/P70/P71 numbers reproduce
exactly. Set ``always_apply=True`` to force erasure at eval (equivalent to
the eval-time monkey-patch).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from semvideo.channel.packet_loss import BernoulliErasureChannel


class ChannelWithErasure(nn.Module):
    """Wrap a SwinJSCC ``Channel`` with optional post-channel Bernoulli erasure.

    The base channel runs first (AWGN / Rayleigh / etc.), then surviving
    symbols pass through a Bernoulli erasure layer with ``p`` sampled
    uniformly from ``[0, p_max]`` per forward pass.

    Args:
        base_channel: The SwinJSCC ``Channel`` instance to wrap. Forwarded
            as a submodule so ``state_dict()`` round-trips cleanly.
        p_max: Maximum erasure probability. Each forward draws
            ``p ~ U[0, p_max]``. Set to 0 to disable erasure entirely.
        always_apply: If True, erasure runs at eval too. Default False
            (training-only, like Dropout). The training script wants False;
            the packet-loss eval script wants True.

    Note:
        The SwinJSCC ``Channel.forward`` signature is
        ``(feature, chan_param, avg_pwr=False)``, where ``chan_param`` is
        the SNR in dB. We forward all three args transparently.
    """

    def __init__(
        self,
        base_channel: nn.Module,
        p_max: float = 0.0,
        always_apply: bool = False,
    ) -> None:
        super().__init__()
        if p_max < 0.0 or p_max > 1.0:
            raise ValueError(f"p_max must be in [0, 1], got {p_max}")
        self.base_channel = base_channel
        self.p_max = float(p_max)
        self.always_apply = bool(always_apply)
        # Single reusable erasure submodule; p is mutated per forward.
        self.erasure = BernoulliErasureChannel(p=0.0, snr_db=None)

    def _should_apply(self) -> bool:
        if self.p_max <= 0.0:
            return False
        if self.always_apply:
            return True
        return self.training

    def forward(
        self,
        feature: torch.Tensor,
        chan_param: float,
        avg_pwr: bool = False,
    ) -> torch.Tensor:
        """Channel-pass with optional erasure.

        Args:
            feature: Pre-channel latent.
            chan_param: SNR (dB) passed to the base channel.
            avg_pwr: Forwarded to SwinJSCC's Rayleigh-side ``Channel.forward``.

        Returns:
            Post-erasure (or post-channel-only if disabled) latent. Shape
            matches ``feature``.
        """
        noisy = self.base_channel.forward(feature, chan_param, avg_pwr)
        if not self._should_apply():
            return noisy
        p = float(torch.empty(1).uniform_(0.0, self.p_max).item())
        # Mutate the reusable erasure module's p – cheaper than re-instantiation.
        self.erasure.p = p
        return self.erasure(noisy)

    def extra_repr(self) -> str:
        return f"p_max={self.p_max}, always_apply={self.always_apply}"
