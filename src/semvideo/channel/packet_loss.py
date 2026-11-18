"""Packet-loss (erasure) channel models for JSCC pipelines.

Two flavours:
  - BernoulliErasure: i.i.d. per-symbol erasure with probability ``p``
  - BurstErasure: 2-state Markov (Gilbert-Elliott) burst erasure

Erased entries are multiplied by zero (hard erasure) and optional Gaussian
noise is added to surviving entries, matching the analog-symbol channel
abstraction in awgn.py / fading.py.

The forward pass is differentiable in ``x`` via the multiplicative mask;
gradients do not flow through the random mask itself (consistent with
how the AWGN noise tensor is treated in awgn.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BernoulliErasureChannel(nn.Module):
    """I.i.d. per-symbol Bernoulli erasure with optional residual AWGN.

    Each symbol is independently erased (multiplied by 0) with probability
    ``p``, and surviving symbols pass through an AWGN layer.

    Args:
        p: Erasure probability in [0, 1]. None for random per-batch.
        p_min, p_max: Range for random sampling when p is None.
        snr_db: Residual SNR (dB) for surviving symbols (None = lossless).
    """

    def __init__(
        self,
        p: float | None = 0.1,
        p_min: float = 0.0,
        p_max: float = 0.5,
        snr_db: float | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.p_min = p_min
        self.p_max = p_max
        self.snr_db = snr_db

    def _get_p(self, p: float | None) -> float:
        if p is not None:
            return p
        if self.p is not None:
            return self.p
        return float(torch.empty(1).uniform_(self.p_min, self.p_max).item())

    def forward(
        self,
        x: torch.Tensor,
        p: float | None = None,
        snr_db: float | None = None,
    ) -> torch.Tensor:
        """Apply erasure to input symbols.

        Args:
            x: Input symbols, any shape.
            p: Override erasure probability.
            snr_db: Override residual SNR for surviving symbols.

        Returns:
            Erased symbols (same shape as input).
        """
        resolved_p = self._get_p(p)
        if resolved_p <= 0.0:
            mask = torch.ones_like(x)
        elif resolved_p >= 1.0:
            mask = torch.zeros_like(x)
        else:
            mask = (torch.rand_like(x) > resolved_p).to(x.dtype)

        y = x * mask

        resolved_snr = snr_db if snr_db is not None else self.snr_db
        if resolved_snr is not None:
            # Compute signal power over surviving symbols (avoid biasing
            # toward higher noise when most of the mask is zero).
            surviving_count = mask.sum().clamp(min=1.0)
            signal_power = (y * y).sum() / surviving_count
            snr_linear = 10.0 ** (resolved_snr / 10.0)
            noise_var = signal_power / snr_linear
            noise = torch.randn_like(x) * torch.sqrt(noise_var)
            # Noise only on surviving symbols – erased stays at exactly zero
            y = y + noise * mask

        return y

    def extra_repr(self) -> str:
        return f"p={self.p}, p_range=[{self.p_min}, {self.p_max}], snr_db={self.snr_db}"


class BurstErasureChannel(nn.Module):
    """Gilbert-Elliott 2-state Markov burst erasure.

    Two states, Good and Bad. In Good, erasure rate ``p_good`` (typically
    near 0). In Bad, erasure rate ``p_bad`` (typically near 1). Transition
    rates ``g_to_b`` and ``b_to_g`` set the mean burst length and overall
    loss rate. The stationary loss rate is
        p_loss = pi_G * p_good + pi_B * p_bad
    with pi_B = g_to_b / (g_to_b + b_to_g).

    Args:
        g_to_b: Good→Bad transition probability per symbol.
        b_to_g: Bad→Good transition probability per symbol.
        p_good: Erasure rate while in Good state.
        p_bad: Erasure rate while in Bad state.
        snr_db: Residual SNR for surviving symbols (None = lossless).
        symbol_axis: Axis along which the Markov chain advances. Default -1.
    """

    def __init__(
        self,
        g_to_b: float = 0.01,
        b_to_g: float = 0.10,
        p_good: float = 0.0,
        p_bad: float = 1.0,
        snr_db: float | None = None,
        symbol_axis: int = -1,
    ) -> None:
        super().__init__()
        self.g_to_b = g_to_b
        self.b_to_g = b_to_g
        self.p_good = p_good
        self.p_bad = p_bad
        self.snr_db = snr_db
        self.symbol_axis = symbol_axis

    def stationary_loss_rate(self) -> float:
        denom = self.g_to_b + self.b_to_g
        if denom <= 0:
            return self.p_good
        pi_b = self.g_to_b / denom
        pi_g = 1.0 - pi_b
        return pi_g * self.p_good + pi_b * self.p_bad

    def _generate_state_sequence(self, length: int, device, generator=None) -> torch.Tensor:
        """Sample Markov-state sequence of given length. Returns 0=Good, 1=Bad.

        Vectorised: draw all uniforms up front, then run a Python loop with
        scalar comparisons (no per-step GPU↔CPU sync). For typical latent
        sizes (~thousands) this is ~100× faster than per-step torch.rand.
        """
        pi_b = self.g_to_b / (self.g_to_b + self.b_to_g + 1e-12)
        # Pre-sample all transition probabilities up front
        u = torch.rand(length + 1, device="cpu", generator=generator).numpy()
        states = np.empty(length, dtype=bool)
        in_bad = u[0] < pi_b
        g_to_b = self.g_to_b
        b_to_g = self.b_to_g
        for i in range(length):
            states[i] = in_bad
            transition_prob = b_to_g if in_bad else g_to_b
            if u[i + 1] < transition_prob:
                in_bad = not in_bad
        return torch.from_numpy(states).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten everything except symbol_axis
        x_perm = x.movedim(self.symbol_axis, -1)
        flat = x_perm.reshape(-1, x_perm.shape[-1])
        b, n = flat.shape
        mask = torch.ones_like(flat)
        for bi in range(b):
            states = self._generate_state_sequence(n, x.device)
            p_per_symbol = torch.where(
                states,
                torch.full_like(states, self.p_bad, dtype=x.dtype),
                torch.full_like(states, self.p_good, dtype=x.dtype),
            )
            row_mask = (torch.rand(n, device=x.device) > p_per_symbol).to(x.dtype)
            mask[bi] = row_mask
        mask = mask.reshape(x_perm.shape).movedim(-1, self.symbol_axis)
        y = x * mask

        if self.snr_db is not None:
            surviving_count = mask.sum().clamp(min=1.0)
            signal_power = (y * y).sum() / surviving_count
            snr_linear = 10.0 ** (self.snr_db / 10.0)
            noise_var = signal_power / snr_linear
            y = y + torch.randn_like(x) * torch.sqrt(noise_var) * mask

        return y

    def extra_repr(self) -> str:
        return (
            f"g_to_b={self.g_to_b}, b_to_g={self.b_to_g}, "
            f"p_good={self.p_good}, p_bad={self.p_bad}, "
            f"stationary_loss={self.stationary_loss_rate():.3f}, "
            f"snr_db={self.snr_db}"
        )
