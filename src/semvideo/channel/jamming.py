"""Active jamming channel model for electronic warfare scenarios.

Combines background AWGN with intentional interference. Supports four jammer
types:

  - "barrage"    – broadband Gaussian interference covering the whole bandwidth
  - "single_tone"– continuous-wave (CW) interferer at a single frequency offset
  - "swept"      – narrowband jammer that hops across the symbol bandwidth
  - "pulsed"     – intermittent jamming, active for `duty_cycle` of symbols
"""

import torch
import torch.nn as nn


class JammingChannel(nn.Module):
    """Differentiable jamming channel combining AWGN with intentional interference.

    Args:
        jammer_type: Type of jammer ("barrage", "single_tone", "swept", "pulsed").
        jnr_db: Jammer-to-noise ratio in dB.
        duty_cycle: Duty cycle for pulsed jammer (0-1).
        snr_db: Background SNR in dB.
    """

    def __init__(
        self,
        jammer_type: str = "barrage",
        jnr_db: float = 10.0,
        duty_cycle: float = 0.5,
        snr_db: float = 10.0,
        tone_freq_norm: float = 0.25,
    ) -> None:
        super().__init__()
        self.jammer_type = jammer_type
        self.jnr_db = jnr_db
        self.duty_cycle = duty_cycle
        self.snr_db = snr_db
        # Normalised frequency for single-tone jammer (cycles per symbol, in [0, 0.5])
        self.tone_freq_norm = tone_freq_norm

    def forward(self, x: torch.Tensor, snr_db: float | None = None) -> torch.Tensor:
        """Apply jamming interference and AWGN to input symbols.

        Args:
            x: Input symbols (B, N).
            snr_db: Override background SNR in dB.

        Returns:
            Jammed and noisy symbols (B, N).
        """
        resolved_snr = snr_db if snr_db is not None else self.snr_db

        # Background AWGN
        signal_power = torch.mean(x**2)
        snr_linear = 10.0 ** (resolved_snr / 10.0)
        noise_var = signal_power / snr_linear
        awgn = torch.randn_like(x) * torch.sqrt(noise_var)

        # Jammer power relative to noise
        jnr_linear = 10.0 ** (self.jnr_db / 10.0)
        jammer_power = noise_var * jnr_linear

        if self.jammer_type == "barrage":
            # Constant wideband jamming – Gaussian interference
            jammer = torch.randn_like(x) * torch.sqrt(jammer_power)

        elif self.jammer_type == "single_tone":
            # Continuous-wave (CW) jammer: deterministic sinusoid. Per-row
            # frequency diversity (audit 2026-06-03 noted that shared
            # tone_freq_norm across batch rows is unphysical) -- we draw
            # a per-row offset in [-tone_freq_norm/4, +tone_freq_norm/4]
            # around the configured normalised frequency, plus per-row
            # random phase so the jammer is not synchronised with the signal.
            n = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
            phase = 2 * torch.pi * torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            freq_jitter = (torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) - 0.5) * (
                self.tone_freq_norm * 0.5
            )
            freq_per_row = self.tone_freq_norm + freq_jitter  # (B, 1)
            tone = torch.cos(2 * torch.pi * freq_per_row * n + phase)  # (B, N)
            # cos has RMS^2 = 1/2; scale by sqrt(2) so the injected power equals
            # barrage at equal nominal JNR (fixes the prior -3 dB single-tone gap).
            jammer = tone.expand_as(x) * torch.sqrt(2.0 * jammer_power)

        elif self.jammer_type == "pulsed":
            # Worst-case partial-band pulsed jamming: jam a duty_cycle fraction of
            # symbol BLOCKS independently within each row (not whole-row on/off),
            # concentrating power into the active pulses. Peak power rises as
            # 1/duty so the TIME-AVERAGE JNR stays jnr_db while a lower duty cycle
            # is a HARDER threat (textbook partial-band result), not a weaker one.
            block = max(1, x.shape[-1] // 20)
            nblk = (x.shape[-1] + block - 1) // block
            blk_mask = (torch.rand(x.shape[0], nblk, device=x.device) < self.duty_cycle).float()
            mask = blk_mask.repeat_interleave(block, dim=1)[:, : x.shape[-1]]
            peak_power = jammer_power / max(self.duty_cycle, 1e-6)
            jammer = torch.randn_like(x) * torch.sqrt(peak_power) * mask

        elif self.jammer_type == "swept":
            # Genuine swept-frequency (chirp) jammer: the instantaneous frequency
            # advances linearly across the symbol axis, sweeping the normalised
            # band over the frame -- narrowband at each instant but covering the
            # band over time (a real frequency hop, not a scattered barrage).
            # Per-row random start frequency + phase decorrelate the batch rows.
            n = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
            f0 = 0.5 * torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            sweep_rate = 0.5 / max(1, x.shape[-1])
            phase = 2 * torch.pi * torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
            inst_phase = 2 * torch.pi * (f0 * n + 0.5 * sweep_rate * n * n) + phase
            chirp = torch.cos(inst_phase)  # (B, N)
            # sqrt(2) so injected power matches barrage at equal nominal JNR
            jammer = chirp.expand_as(x) * torch.sqrt(2.0 * jammer_power)
        else:
            raise ValueError(f"Unknown jammer type: {self.jammer_type}")

        return x + awgn + jammer

    def extra_repr(self) -> str:
        return (
            f"jammer_type={self.jammer_type}, jnr_db={self.jnr_db}, "
            f"duty_cycle={self.duty_cycle}, snr_db={self.snr_db}, "
            f"tone_freq_norm={self.tone_freq_norm}"
        )
