"""Standalone inference latency benchmark for DeepJSCC on any target device.

Designed to run on Jetson Orin NX (or any CUDA/CPU device) with minimal
dependencies. No DJI dataset required -- uses a synthetic random frame if
no image is provided.

Measures two timings:
  - codec only:   encoder forward + (optional channel) + decoder forward
  - full pipeline: resize 1080->512 + codec + resize 512->1080

Usage (laptop, quick sanity check):
    cd semvideo
    .venv/Scripts/python scripts/jetson_inference_bench.py \
        --checkpoint experiments/p13_deepjscc_yolofeat_512_*/best.pt

Usage (Jetson, after copying best.pt):
    python3 jetson_inference_bench.py --checkpoint p13_best.pt --device cuda

Options:
    --checkpoint PATH   Path to .pt checkpoint (required)
    --image PATH        Path to a test image; if omitted uses synthetic 1920x1080
    --snr-db FLOAT      AWGN SNR in dB for timing (default: 10.0)
    --n-warmup INT      Warm-up iterations (default: 20)
    --n-iters INT       Timed iterations (default: 200)
    --input-size INT    Codec input size in pixels, square (default: 512)
    --device STR        "cuda" or "cpu" (default: cuda if available, else cpu)
    --out CSV           Output CSV path (default: jetson_bench_results.csv)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal inline model -- avoids needing the semvideo package on Jetson.
# Mirrors src/semvideo/models/deep_jscc.py exactly.
# ---------------------------------------------------------------------------

class _DeepJSCC(nn.Module):
    def __init__(self, channel_ratio: float = 1 / 6, base_channels: int = 64) -> None:
        super().__init__()
        c = max(1, int(256 * channel_ratio))
        bc = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, bc, 5, stride=2, padding=2), nn.PReLU(),
            nn.Conv2d(bc, bc * 2, 5, stride=2, padding=2), nn.PReLU(),
            nn.Conv2d(bc * 2, bc * 4, 5, stride=2, padding=2), nn.PReLU(),
            nn.Conv2d(bc * 4, bc * 4, 5, stride=2, padding=2), nn.PReLU(),
            nn.Conv2d(bc * 4, c, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(c, bc * 4, 3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(bc * 4, bc * 4, 5, stride=2, padding=2, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(bc * 4, bc * 2, 5, stride=2, padding=2, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(bc * 2, bc, 5, stride=2, padding=2, output_padding=1), nn.PReLU(),
            nn.ConvTranspose2d(bc, 3, 5, stride=2, padding=2, output_padding=1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """Returns (normalized symbols, spatial_shape) for split encoder/decoder use."""
        z = self.encoder(x)
        spatial = z.shape[1:]
        z_flat = z.reshape(z.shape[0], -1)
        with torch.no_grad():
            scale = 1.0 / ((z_flat ** 2).mean(-1, keepdim=True).sqrt() + 1e-8)
        return z_flat * scale, spatial

    def decode(self, symbols: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """Reconstruct image from received symbols."""
        return self.decoder(symbols.reshape(-1, *spatial_shape))

    def forward(self, x: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        z_norm, spatial = self.encode(x)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_std = (z_norm.detach() ** 2).mean().sqrt() / (snr_lin ** 0.5)
        z_rx = z_norm + torch.randn_like(z_norm) * noise_std
        return self.decode(z_rx, spatial)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str) -> tuple[_DeepJSCC, int, dict]:
    """Load DeepJSCC from checkpoint. Returns (model, n_params, config)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    channel_ratio = cfg.get("channel_ratio", 1 / 6)
    base_channels = cfg.get("base_channels", 64)
    model = _DeepJSCC(channel_ratio=channel_ratio, base_channels=base_channels).to(device)
    state = ck.get("model_state_dict", ck.get("model_state", ck))
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, cfg


# ---------------------------------------------------------------------------
# Frame provider
# ---------------------------------------------------------------------------

def get_test_frame(image_path: str | None, height: int = 1080, width: int = 1920) -> np.ndarray:
    """Return HxWx3 uint8 BGR frame. Falls back to synthetic noise if no image."""
    if image_path:
        try:
            import cv2
            frame = cv2.imread(image_path)
            if frame is not None:
                return frame
            print(f"Warning: could not read {image_path}, using synthetic frame.")
        except ImportError:
            try:
                from PIL import Image
                img = np.array(Image.open(image_path).convert("RGB"))
                return img[:, :, ::-1].copy()
            except Exception:
                print("Warning: PIL failed too, using synthetic frame.")
    # Synthetic: natural-image-like values (not pure random noise)
    rng = np.random.default_rng(42)
    frame = rng.integers(30, 220, (height, width, 3), dtype=np.uint8)
    return frame


def frame_to_tensor(frame: np.ndarray, size: int, device: str) -> torch.Tensor:
    """Resize frame to size x size and convert to (1, 3, H, W) float tensor in [0,1]."""
    try:
        import cv2
        small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    except ImportError:
        try:
            from PIL import Image
            pil = Image.fromarray(frame[:, :, ::-1])
            rgb = np.array(pil.resize((size, size), Image.BILINEAR))
        except ImportError:
            # No cv2 or PIL: just crop/pad with numpy
            rgb = frame[:size, :size]
            if rgb.shape[0] < size or rgb.shape[1] < size:
                tmp = np.zeros((size, size, 3), dtype=np.uint8)
                tmp[:rgb.shape[0], :rgb.shape[1]] = rgb
                rgb = tmp
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def tensor_to_frame(t: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """Upsample (1,3,H,W) tensor back to out_h x out_w."""
    arr = (t.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    try:
        import cv2
        return cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    except ImportError:
        try:
            from PIL import Image
            return np.array(Image.fromarray(arr).resize((out_w, out_h), Image.BICUBIC))
        except ImportError:
            return arr


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    model: _DeepJSCC,
    frame: np.ndarray,
    codec_size: int,
    snr_db: float,
    n_warmup: int,
    n_iters: int,
    device: str,
) -> dict:
    h, w = frame.shape[:2]
    x = frame_to_tensor(frame, codec_size, device)

    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad():
            rec = model(x, snr_db=snr_db)
        _ = tensor_to_frame(rec, h, w)
    sync(device)

    times_enc: list[float] = []
    times_dec: list[float] = []
    times_codec: list[float] = []
    times_full: list[float] = []

    for _ in range(n_iters):
        t_full_start = time.perf_counter()
        x_in = frame_to_tensor(frame, codec_size, device)
        sync(device)

        # Encoder
        t_enc_start = time.perf_counter()
        with torch.no_grad():
            z_norm, spatial = model.encode(x_in)
        sync(device)
        t_enc_end = time.perf_counter()

        # Channel (AWGN)
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_std = (z_norm.detach() ** 2).mean().sqrt() / (snr_lin ** 0.5)
        z_rx = z_norm + torch.randn_like(z_norm) * noise_std
        sync(device)

        # Decoder
        t_dec_start = time.perf_counter()
        with torch.no_grad():
            rec = model.decode(z_rx, spatial)
        sync(device)
        t_dec_end = time.perf_counter()

        _ = tensor_to_frame(rec, h, w)
        t_full_end = time.perf_counter()

        times_enc.append((t_enc_end - t_enc_start) * 1000)
        times_dec.append((t_dec_end - t_dec_start) * 1000)
        times_codec.append((t_dec_end - t_enc_start) * 1000)
        times_full.append((t_full_end - t_full_start) * 1000)

    return {
        "enc_median_ms":   float(np.median(times_enc)),
        "enc_fps":         1000.0 / float(np.median(times_enc)),
        "dec_median_ms":   float(np.median(times_dec)),
        "dec_fps":         1000.0 / float(np.median(times_dec)),
        "codec_median_ms": float(np.median(times_codec)),
        "codec_p10_ms":    float(np.percentile(times_codec, 10)),
        "codec_p90_ms":    float(np.percentile(times_codec, 90)),
        "codec_fps":       1000.0 / float(np.median(times_codec)),
        "full_median_ms":  float(np.median(times_full)),
        "full_p10_ms":     float(np.percentile(times_full, 10)),
        "full_p90_ms":     float(np.percentile(times_full, 90)),
        "full_fps":        1000.0 / float(np.median(times_full)),
        "n_iters":         n_iters,
    }


# ---------------------------------------------------------------------------
# Visual comparison
# ---------------------------------------------------------------------------

def _save_visual(
    model: _DeepJSCC,
    frame: np.ndarray,
    codec_size: int,
    snr_db: float,
    device: str,
    out_path: str,
) -> None:
    """Save side-by-side original vs reconstructed PNG with PSNR annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = frame.shape[:2]
    x = frame_to_tensor(frame, codec_size, device)
    with torch.no_grad():
        rec = model(x, snr_db=snr_db)

    # Original resized to codec input (apples-to-apples)
    orig_t = frame_to_tensor(frame, codec_size, device)
    orig_np = (orig_t.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    rec_np = (rec.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    mse = np.mean((orig_np.astype(float) - rec_np.astype(float)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"DeepJSCC P13  |  SNR = {snr_db} dB  |  PSNR = {psnr:.2f} dB", fontsize=14)

    axes[0].imshow(orig_np)
    axes[0].set_title("Original (512x512)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(rec_np)
    axes[1].set_title(f"Reconstructed (512x512)  PSNR={psnr:.2f} dB", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-SNR degradation figure
# ---------------------------------------------------------------------------

def _save_snr_compare(
    model: _DeepJSCC,
    frame: np.ndarray,
    codec_size: int,
    device: str,
    out_path: str,
    snr_levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
) -> None:
    """Save a grid: top row = original crops, bottom row = reconstructed at each SNR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame_to_tensor(frame, codec_size, device)
    orig_np = (x.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    n = len(snr_levels)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle("DeepJSCC P13 - Graceful degradation vs SNR", fontsize=14, y=1.01)

    for col, snr in enumerate(snr_levels):
        with torch.no_grad():
            rec = model(x, snr_db=snr)
        rec_np = (rec.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        mse = np.mean((orig_np.astype(float) - rec_np.astype(float)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")

        axes[0, col].imshow(orig_np)
        axes[0, col].set_title(f"Original  (SNR={snr:.0f} dB)", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(rec_np)
        axes[1, col].set_title(f"Reconstructed\nPSNR={psnr:.2f} dB", fontsize=11)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepJSCC inference latency benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image", default=None, help="Test image path (optional)")
    parser.add_argument("--snr-db", type=float, default=10.0, help="AWGN SNR in dB")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=512,
                        help="Codec input resolution (square)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cuda' or 'cpu' (auto-detect if omitted)")
    parser.add_argument("--out", default="jetson_bench_results.csv")
    parser.add_argument("--save-visual", default=None, metavar="PATH",
                        help="Save before/after comparison PNG to this path")
    parser.add_argument("--compare-snrs", action="store_true",
                        help="Save multi-SNR degradation figure (0/5/10/20 dB) alongside --save-visual")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  DeepJSCC Inference Benchmark")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM       : {total_mem:.1f} GB")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  Input size : {args.input_size}x{args.input_size}")
    print(f"  SNR        : {args.snr_db} dB")
    print(f"  Iterations : {args.n_warmup} warmup + {args.n_iters} timed")
    print()

    model, n_params, cfg = load_model(args.checkpoint, device)
    print(f"  Model params: {n_params:,}")
    print(f"  channel_ratio={cfg.get('channel_ratio', 1/6):.4f}, "
          f"base_channels={cfg.get('base_channels', 64)}")
    print()

    frame = get_test_frame(args.image)
    print(f"  Test frame : {frame.shape[1]}x{frame.shape[0]} "
          f"({'from ' + args.image if args.image else 'synthetic'})")
    print()
    print("  Running benchmark...")

    stats = benchmark(
        model, frame,
        codec_size=args.input_size,
        snr_db=args.snr_db,
        n_warmup=args.n_warmup,
        n_iters=args.n_iters,
        device=device,
    )

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Encoder only:")
    print(f"    Median : {stats['enc_median_ms']:.2f} ms  ({stats['enc_fps']:.1f} fps)")
    print()
    print(f"  Decoder only:")
    print(f"    Median : {stats['dec_median_ms']:.2f} ms  ({stats['dec_fps']:.1f} fps)")
    print()
    print(f"  Codec  (encoder + channel + decoder):")
    print(f"    Median : {stats['codec_median_ms']:.2f} ms  ({stats['codec_fps']:.1f} fps)")
    print(f"    P10/P90: {stats['codec_p10_ms']:.2f} / {stats['codec_p90_ms']:.2f} ms")
    print()
    print(f"  Full pipeline (resize 1080->512 + codec + resize 512->1080):")
    print(f"    Median : {stats['full_median_ms']:.2f} ms  ({stats['full_fps']:.1f} fps)")
    print(f"    P10/P90: {stats['full_p10_ms']:.2f} / {stats['full_p90_ms']:.2f} ms")
    print()

    target_fps = 30.0
    ok = stats["full_fps"] >= target_fps
    symbol = "OK" if ok else "BELOW TARGET"
    print(f"  30 fps target: {symbol} (full pipeline {stats['full_fps']:.1f} fps)")
    print("=" * 60)

    row = {
        "checkpoint": args.checkpoint,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "cpu",
        "n_params": n_params,
        "input_size": args.input_size,
        "snr_db": args.snr_db,
        **stats,
    }
    if args.save_visual:
        _save_visual(model, frame, args.input_size, args.snr_db, device, args.save_visual)
        print(f"\n  Visual saved -> {args.save_visual}")
    if args.compare_snrs:
        snr_path = (Path(args.save_visual).with_suffix("") if args.save_visual
                    else Path("jetson_bench_visual"))
        snr_path = str(snr_path) + "_snr_compare.png"
        _save_snr_compare(model, frame, args.input_size, device, snr_path)
        print(f"  SNR compare saved -> {snr_path}")

    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\n  Results saved -> {args.out}")


if __name__ == "__main__":
    main()
