"""Additive White Gaussian Noise (AWGN) channel model.

Differentiable channel model for end-to-end JSCC training.
Noise power is computed from the specified SNR and input signal power.
"""

import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """Differentiable AWGN channel for end-to-end JSCC training.

    Adds complex Gaussian noise to input symbols based on the specified SNR.
    Supports both fixed and random SNR during training.

    Args:
        snr_db: Signal-to-noise ratio in dB (None for random per-batch).
        snr_db_min: Minimum SNR for random sampling.
        snr_db_max: Maximum SNR for random sampling.
    """

    def __init__(
        self,
        snr_db: float | None = 10.0,
        snr_db_min: float = 0.0,
        snr_db_max: float = 20.0,
    ) -> None:
        super().__init__()
        self.snr_db = snr_db
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max

    def _get_snr_db(self, snr_db: float | None) -> float:
        """Resolve SNR value: explicit > instance > random."""
        if snr_db is not None:
            return snr_db
        if self.snr_db is not None:
            return self.snr_db
        # Random SNR for training robustness
        return float(torch.empty(1).uniform_(self.snr_db_min, self.snr_db_max).item())

    def forward(self, x: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        """Add AWGN noise to input symbols.

        The noise variance is computed from the input signal power and SNR:
            sigma^2 = signal_power / (10^(snr_db/10))

        Args:
            x: Input symbols (B, N) – real-valued.
            snr_db: Override SNR in dB. If None, uses self.snr_db or random.

        Returns:
            Noisy symbols (B, N) with same shape as input.
        """
        if not self.training and snr_db is None and self.snr_db is not None:
            snr_db = self.snr_db

        resolved_snr = self._get_snr_db(snr_db)

        # Compute signal power per sample
        signal_power = torch.mean(x**2)

        # SNR linear scale
        snr_linear = 10.0 ** (resolved_snr / 10.0)

        # Noise variance
        noise_var = signal_power / snr_linear

        # Generate noise (same shape as input)
        noise = torch.randn_like(x) * torch.sqrt(noise_var)

        return x + noise

    def extra_repr(self) -> str:
        return f"snr_db={self.snr_db}, range=[{self.snr_db_min}, {self.snr_db_max}]"
