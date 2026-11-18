"""DeepJSCC: CNN-based Joint Source-Channel Coding.

Proven architecture following Bourtsoulatze et al. (2019) "Deep Joint
Source-Channel Coding for Wireless Image Transmission". Uses strided
convolutions for encoding and transposed convolutions for decoding,
with power normalization at the channel bottleneck.

Key advantages over ViT-based JSCC:
    - Stable gradient flow through CNN layers
    - No vanishing gradient through power normalization
    - Fast convergence (< 100 epochs)
    - Adjustable compression via channel_ratio parameter
"""

import torch
import torch.nn as nn


class DeepJSCC(nn.Module):
    """CNN-based Joint Source-Channel Coding model.

    Architecture::

        Input (B, 3, H, W)
          -> Encoder: 5 conv layers (stride-2 downsampling)
          -> Channel bottleneck: (B, C, H/16, W/16)
          -> Power normalization (gradient-friendly)
          -> [Channel model] (AWGN / Rayleigh / Jamming)
          -> Decoder: 5 transposed conv layers (stride-2 upsampling)
          -> Reconstructed image (B, 3, H, W)

    Args:
        channel_ratio: Ratio of channel bottleneck width to base width (256).
            Controls compression: 1/6 gives ~18x, 1/12 gives ~37x.
        base_channels: Base channel width for encoder/decoder.
    """

    def __init__(
        self,
        channel_ratio: float = 1 / 6,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.channel_ratio = channel_ratio
        c = max(1, int(256 * channel_ratio))  # channel bottleneck width
        bc = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(3, bc, 5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(bc, bc * 2, 5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(bc * 2, bc * 4, 5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(bc * 4, bc * 4, 5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(bc * 4, c, 3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(c, bc * 4, 3, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(bc * 4, bc * 4, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(bc * 4, bc * 2, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(bc * 2, bc, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(bc, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to channel symbols.

        Args:
            x: Input image (B, 3, H, W) in [0, 1].

        Returns:
            Normalized channel symbols (B, N_symbols).
        """
        z = self.encoder(x)  # (B, C, H/16, W/16)
        b = z.shape[0]
        z_flat = z.reshape(b, -1)

        # Gradient-friendly power normalization
        with torch.no_grad():
            power = (z_flat**2).mean(dim=-1, keepdim=True)
            scale = 1.0 / (power.sqrt() + 1e-8)
        z_norm = z_flat * scale

        return z_norm

    def decode(self, symbols: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Decode received symbols to image.

        Args:
            symbols: Received channel symbols (B, N_symbols).
            spatial_shape: Original (C, H, W) shape of encoded features.

        Returns:
            Reconstructed image (B, 3, H, W) in [0, 1].
        """
        z = symbols.reshape(-1, *spatial_shape)
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        channel: nn.Module | None = None,
        snr_db: float | None = None,
    ) -> torch.Tensor:
        """Full forward pass: encode -> channel -> decode.

        Args:
            x: Input image (B, 3, H, W) in [0, 1].
            channel: Differentiable channel model. If None, noiseless.
            snr_db: Channel SNR in dB (passed to channel model).

        Returns:
            Reconstructed image (B, 3, H, W) in [0, 1].
        """
        z = self.encoder(x)  # (B, C, H/16, W/16)
        spatial_shape = z.shape[1:]  # (C, H/16, W/16)
        b = z.shape[0]

        # Flatten and normalize power
        z_flat = z.reshape(b, -1)
        with torch.no_grad():
            power = (z_flat**2).mean(dim=-1, keepdim=True)
            scale = 1.0 / (power.sqrt() + 1e-8)
        z_norm = z_flat * scale

        # Pass through channel.
        # Note: deliberately do NOT forward `snr_db` into the channel – `snr_db`
        # is reserved for SNR conditioning (DeepJSCC has none; the argument keeps
        # the calling convention of the training script). The channel uses
        # its own configured SNR, which is what the caller actually controls.
        z_rx = channel(z_norm) if channel is not None else z_norm

        # Reshape and decode
        z_rx = z_rx.reshape(b, *spatial_shape)
        return self.decoder(z_rx)

    def get_compression_stats(self, img_size: tuple[int, int] = (128, 128)) -> dict:
        """Compute compression statistics.

        Args:
            img_size: Input image size (H, W).

        Returns:
            Dict with compression metrics.
        """
        input_pixels = 3 * img_size[0] * img_size[1]
        c = max(1, int(256 * self.channel_ratio))
        spatial_h = img_size[0] // 16
        spatial_w = img_size[1] // 16
        n_symbols = c * spatial_h * spatial_w
        return {
            "input_pixels": input_pixels,
            "n_symbols": n_symbols,
            "compression_ratio": input_pixels / n_symbols,
            "channel_bottleneck": f"{c}x{spatial_h}x{spatial_w}",
        }
