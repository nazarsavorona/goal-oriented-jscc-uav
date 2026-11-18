"""Rayleigh fading channel model with multipath propagation.

Models flat Rayleigh fading where each symbol experiences independent
fading coefficient h ~ CN(0, 1), resulting in |h| ~ Rayleigh.
AWGN is added after fading.
"""

import torch
import torch.nn as nn


class RayleighFadingChannel(nn.Module):
    """Differentiable Rayleigh fading channel for wireless simulation.

    Each symbol is multiplied by a complex fading coefficient drawn from
    CN(0, 1), then AWGN is added. Perfect CSI is assumed at the receiver
    (equalization divides by h).

    Args:
        snr_db: Signal-to-noise ratio in dB.
        snr_db_min: Minimum SNR for random sampling.
        snr_db_max: Maximum SNR for random sampling.
        equalize: Whether to perform perfect CSI equalization.
    """

    def __init__(
        self,
        snr_db: float | None = 10.0,
        snr_db_min: float = 0.0,
        snr_db_max: float = 20.0,
        equalize: bool = True,
    ) -> None:
        super().__init__()
        self.snr_db = snr_db
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.equalize = equalize

    def _get_snr_db(self, snr_db: float | None) -> float:
        if snr_db is not None:
            return snr_db
        if self.snr_db is not None:
            return self.snr_db
        return float(torch.empty(1).uniform_(self.snr_db_min, self.snr_db_max).item())

    def forward(self, x: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        """Apply Rayleigh fading and AWGN to input symbols.

        Args:
            x: Input symbols (B, N) – real-valued, treated as real part of complex.
            snr_db: Override SNR in dB.

        Returns:
            Received symbols (B, N) after fading, noise, and optional equalization.
        """
        resolved_snr = self._get_snr_db(snr_db)

        # Rayleigh fading: h = (h_real + j*h_imag) / sqrt(2), |h| ~ Rayleigh
        h_real = torch.randn_like(x) / (2**0.5)
        h_imag = torch.randn_like(x) / (2**0.5)
        h_mag = torch.sqrt(h_real**2 + h_imag**2)  # (B, N)

        # Apply fading (real-valued model: multiply by |h|)
        faded = x * h_mag

        # Add AWGN
        signal_power = torch.mean(x**2)
        snr_linear = 10.0 ** (resolved_snr / 10.0)
        noise_var = signal_power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_var)

        received = faded + noise

        # Perfect CSI equalization
        if self.equalize:
            received = received / (h_mag + 1e-8)

        return received

    def extra_repr(self) -> str:
        return f"snr_db={self.snr_db}, equalize={self.equalize}"
