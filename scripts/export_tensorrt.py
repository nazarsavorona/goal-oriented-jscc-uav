"""Export SwinJSCC w/SA codec to ONNX (and downstream TensorRT) for Jetson.

Strategy:
  1. Load checkpoint (P68 default) via the same `build_swinjscc` path used by the
     CPU eval scripts.
  2. **Detach the ChannelWithErasure / channel layer** before export – the channel
     is *runtime* state (SNR + optional erasure), injected on-device between the
     encoder.engine and decoder.engine. Baking the channel into the export would
     fuse a stochastic op into a deterministic graph, which TensorRT cannot
     represent.
  3. Export encoder and decoder as **two separate ONNX graphs** with stable
     static input shapes (and an optional dynamic batch axis). Two graphs let us
     wire the AWGN noise injection in NumPy/CUDA on the Jetson side without
     re-tracing the model.
  4. Run an ``onnxruntime`` CPU round-trip check on the host: PSNR of the ONNX
     reconstruction vs the PyTorch reconstruction must be ≥ 50 dB on a 16-frame
     batch (i.e. essentially bit-identical) before we hand off to ``trtexec``.

The TensorRT engine build itself runs on the Jetson side (`trtexec --onnx=...
--fp16 --useCudaGraph`).

Usage (run from the repository root):
    docker run --rm --runtime=nvidia \
        -v "$PWD":/workspace -v /path/to/third_party:/third_party \
        -v /path/to/data:/data -w /workspace semvideo:latest \
        python scripts/export_tensorrt.py \
            --checkpoint /workspace/experiments/p68_p49_erasure_aug_p40/best.pt \
            --output-dir /workspace/experiments/wp7_jetson \
            --image-dims 3 512 512 --channel-symbols 96 \
            --opset 17 --dynamic-batch

Jetson recipe (opset 13 -- use this when handing ONNX to an older JetPack /
TensorRT that caps at opset 13, e.g. a Jetson Orin NX on JetPack 5.1.x):
    docker run --rm --runtime=nvidia \
        -v "$PWD":/workspace -v /path/to/third_party:/third_party \
        -v /path/to/data:/data -w /workspace semvideo:latest \
        python scripts/export_tensorrt.py \
            --checkpoint /workspace/experiments/p68_p49_erasure_aug_p40/best.pt \
            --output-dir /workspace/experiments/wp7_jetson \
            --image-dims 3 512 512 --channel-symbols 96 \
            --opset 13 --dynamic-batch
    Note: opset 13 forces PyTorch to decompose nn.LayerNorm into primitives;
    the native LayerNormalization op (opset 17) is unsupported by TensorRT
    builds that cap at opset 13 (older JetPack).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Path plumbing – same convention as the other eval scripts so this can run
# unchanged inside the Docker image that mounts the repository at /workspace.
REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVIDEO_ROOT = Path(__file__).resolve().parents[1]
# Upstream SwinJSCC code is not distributed with this repository (see README, "Third-party
# code"). Set SWINJSCC_DIR, or place it at <parent of this repository>/third_party/SwinJSCC
# (/third_party/SwinJSCC when the repository is mounted at /workspace in the Docker image).
SWINJSCC_DIR = Path(os.environ.get("SWINJSCC_DIR", REPO_ROOT / "third_party" / "SwinJSCC"))
sys.path.insert(0, str(SEMVIDEO_ROOT / "src"))
sys.path.insert(0, str(SWINJSCC_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_swinjscc(checkpoint: Path, image_dims=(3, 512, 512), C: int = 96,
                   channel: str = "awgn") -> nn.Module:
    """Build SwinJSCC w/SA and load the checkpoint with shape-filtered load.

    Mirrors `scripts/eval_swinjscc_dinov2.build_swinjscc` but kept local to
    avoid the chdir() side-effect in that module.
    """
    from net.network import SwinJSCC  # noqa: E402

    sj_args = SimpleNamespace(
        model="SwinJSCC_w/_SA", channel_type=channel, C=str(C),
        multiple_snr="10", distortion_metric="MSE",
        trainset="DIV2K", logger=None,
    )
    cfg = SimpleNamespace(
        seed=42, pass_channel=True, CUDA=torch.cuda.is_available(),
        device=DEVICE, norm=False, print_step=100, plot_step=10000,
        normalize=False, learning_rate=1e-4, image_dims=image_dims,
        batch_size=1, downsample=4, logger=None,
    )
    cfg.encoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]),
        patch_size=2, in_chans=3,
        embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2],
        num_heads=[4, 6, 8, 10], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    cfg.decoder_kwargs = dict(
        model="SwinJSCC_w/_SA", img_size=(image_dims[1], image_dims[2]),
        embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2],
        num_heads=[10, 8, 6, 4], C=C, window_size=8, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
        patch_norm=True,
    )
    net = SwinJSCC(sj_args, cfg).to(DEVICE).eval()

    state = torch.load(str(checkpoint), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # Shape-filtered load – accepts both wrapped (ChannelWithErasure prefix) and
    # unwrapped state dicts; the channel layer keys differ between P49 / P68.
    target = net.state_dict()
    loaded = {}
    for k, v in state.items():
        if k in target and v.shape == target[k].shape:
            loaded[k] = v
        else:
            # Try stripping a one-level wrap prefix (e.g. channel.base_channel.h
            # -> channel.h) for P68 checkpoints saved through ChannelWithErasure.
            stripped = k.replace("channel.base_channel.", "channel.")
            if stripped in target and v.shape == target[stripped].shape:
                loaded[stripped] = v
    target.update(loaded)
    net.load_state_dict(target, strict=False)
    print(f"[load] {checkpoint.name}: {len(loaded)}/{len(target)} keys matched", flush=True)
    return net


class EncoderWrapper(nn.Module):
    """ONNX-exportable wrapper for the SwinJSCC encoder.

    Hardcodes the channel parameter (SNR) and channel-number (C) as fixed
    inputs to the underlying encoder – they're constant per export. The
    forward returns only the latent ``feature`` tensor (no CBR, no channel
    side info), matching the contract the TRT engine will expose.
    """

    def __init__(self, encoder: nn.Module, C: int, snr_db: float,
                 model_name: str = "SwinJSCC_w/_SA") -> None:
        super().__init__()
        self.encoder = encoder
        self.C = int(C)
        self.snr_db = float(snr_db)
        self.model_name = model_name

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # SwinJSCC encoder signature: (image, chan_param, channel_number, model_name)
        # For w/SA the encoder returns a single tensor of shape (B, N, C).
        return self.encoder(image, self.snr_db, self.C, self.model_name)


class DecoderWrapper(nn.Module):
    """ONNX-exportable wrapper for the SwinJSCC decoder.

    Takes the (noisy) latent and returns the reconstruction in [0, 1] range
    (after clamping inside the SwinJSCC decoder).
    """

    def __init__(self, decoder: nn.Module, snr_db: float,
                 model_name: str = "SwinJSCC_w/_SA") -> None:
        super().__init__()
        self.decoder = decoder
        self.snr_db = float(snr_db)
        self.model_name = model_name

    def forward(self, noisy_feature: torch.Tensor) -> torch.Tensor:
        # SwinJSCC decoder signature: (feature, chan_param, model_name).
        return self.decoder(noisy_feature, self.snr_db, self.model_name)


def export_split_onnx(
    net: nn.Module,
    out_dir: Path,
    image_dims: Tuple[int, int, int],
    C: int,
    snr_db: float,
    opset: int,
    dynamic_batch: bool,
) -> Tuple[Path, Path]:
    """Export encoder and decoder as two separate ONNX files.

    Returns (encoder_onnx_path, decoder_onnx_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    enc_path = out_dir / "p68_encoder.onnx"
    dec_path = out_dir / "p68_decoder.onnx"

    # SwinJSCC lazily updates attn_mask buffers on the first forward at a given
    # resolution. Run a dummy forward through the full model first so all buffers
    # are materialised before tracing.
    h, w = image_dims[1], image_dims[2]
    dummy_image = torch.randn(1, *image_dims, device=DEVICE)
    with torch.no_grad():
        # Force the encoder/decoder to update_resolution to (h, w).
        net.encoder.update_resolution(h, w)
        net.decoder.update_resolution(h // (2 ** net.downsample),
                                      w // (2 ** net.downsample))
        _dummy_feat = net.encoder(dummy_image, snr_db, C, "SwinJSCC_w/_SA")
        _ = net.decoder(_dummy_feat, snr_db, "SwinJSCC_w/_SA")

    enc_wrapper = EncoderWrapper(net.encoder, C=C, snr_db=snr_db).eval()
    dec_wrapper = DecoderWrapper(net.decoder, snr_db=snr_db).eval()

    dyn_axes_enc = {"image": {0: "B"}, "feature": {0: "B"}} if dynamic_batch else None
    dyn_axes_dec = {"feature": {0: "B"}, "image": {0: "B"}} if dynamic_batch else None

    print(f"[export] encoder -> {enc_path}", flush=True)
    torch.onnx.export(
        enc_wrapper, (dummy_image,), str(enc_path),
        input_names=["image"], output_names=["feature"],
        dynamic_axes=dyn_axes_enc, opset_version=opset,
        do_constant_folding=True,
    )

    with torch.no_grad():
        dummy_feat = enc_wrapper(dummy_image)

    print(f"[export] decoder -> {dec_path}", flush=True)
    print(f"[export] decoder input latent shape: {tuple(dummy_feat.shape)}",
          flush=True)
    torch.onnx.export(
        dec_wrapper, (dummy_feat,), str(dec_path),
        input_names=["feature"], output_names=["image"],
        dynamic_axes=dyn_axes_dec, opset_version=opset,
        do_constant_folding=True,
    )
    return enc_path, dec_path


def verify_onnx_ops(onnx_path: Path, max_opset: int) -> None:
    """Verify an exported ONNX graph is safe for an opset-capped TensorRT build.

    Loads the model, runs the structural checker, asserts the default-domain
    opset is ``<= max_opset``, and scans every node for op types that exist
    only from opset 17 and are hostile to TensorRT builds that cap at opset 13
    (older JetPack). ``LayerNormalization`` is a hard error (SwinJSCC uses
    nn.LayerNorm heavily; at opset 13 PyTorch decomposes it into primitives);
    ``GroupNormalization`` is warned about but not fatal.

    Raises:
        RuntimeError: if the default opset exceeds ``max_opset`` or any hard
            opset-17-only op type is present.
    """
    import onnx  # lazy import -- server may lack the package

    # Op types that only exist from opset 17 and break TRT builds capped at 13.
    hard_ops = {"LayerNormalization"}
    warn_ops = {"GroupNormalization"}

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # Read the default-domain (ai.onnx, represented as "" or "ai.onnx") opset.
    default_opset = None
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            default_opset = int(imp.version)
            break
    if default_opset is None:
        raise RuntimeError(
            f"[verify] {onnx_path.name}: could not determine default-domain opset"
        )
    if default_opset > max_opset:
        raise RuntimeError(
            f"[verify] {onnx_path.name}: default opset {default_opset} exceeds "
            f"max_opset {max_opset} -- re-export with --opset {max_opset}."
        )

    nodes = list(model.graph.node)
    found_hard = sorted({n.op_type for n in nodes if n.op_type in hard_ops})
    found_warn = sorted({n.op_type for n in nodes if n.op_type in warn_ops})

    for op in found_warn:
        print(f"[verify] {onnx_path.name}: WARNING op '{op}' present -- "
              f"verify your TensorRT build supports it at opset {default_opset}.",
              flush=True)

    if found_hard:
        raise RuntimeError(
            f"[verify] {onnx_path.name}: opset-17-only op(s) {found_hard} present "
            f"-- TensorRT builds capped at opset 13 (older JetPack) cannot consume "
            f"these. Re-export at opset 13 so PyTorch decomposes nn.LayerNorm into "
            f"primitives (ReduceMean/Sub/Pow/Sqrt/Div/Mul/Add)."
        )

    print(f"[verify] {onnx_path.name}: PASS (opset {default_opset}, "
          f"{len(nodes)} nodes, no LayerNormalization)", flush=True)


def round_trip_check(net: nn.Module, enc_path: Path, dec_path: Path,
                     image_dims: Tuple[int, int, int], C: int, snr_db: float,
                     n_samples: int = 16) -> dict:
    """ONNXRuntime CPU round-trip vs PyTorch. Returns metrics dict."""
    import onnxruntime as ort  # lazy import – server may lack the package

    enc_sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])

    diffs = []
    for i in range(n_samples):
        x = torch.rand(1, *image_dims, device=DEVICE)
        with torch.no_grad():
            feat_pt = net.encoder(x, snr_db, C, "SwinJSCC_w/_SA")
            # Channel noise is OUTSIDE the export – for the round-trip we feed
            # the clean latent directly into the decoder, matching what the TRT
            # engines will see on Jetson when the runtime AWGN is disabled.
            rec_pt = net.decoder(feat_pt, snr_db, "SwinJSCC_w/_SA").clamp(0, 1)

        x_np = x.cpu().numpy()
        feat_onnx = enc_sess.run(["feature"], {"image": x_np})[0]
        rec_onnx = dec_sess.run(["image"], {"feature": feat_onnx})[0]
        rec_onnx = np.clip(rec_onnx, 0, 1)

        mse = float(np.mean((rec_pt.cpu().numpy() - rec_onnx) ** 2))
        psnr = float(99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse))
        diffs.append(psnr)

    return {
        "mean_psnr_db": float(np.mean(diffs)),
        "min_psnr_db": float(np.min(diffs)),
        "n_samples": n_samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", default="experiments/wp7_jetson", type=Path)
    ap.add_argument("--image-dims", nargs=3, type=int, default=[3, 512, 512],
                    metavar=("C", "H", "W"))
    ap.add_argument("--channel-symbols", "--C", type=int, default=96)
    ap.add_argument("--snr-db", type=float, default=10.0,
                    help="Hardcoded SNR baked into the export. Runtime SNR is "
                         "injected on Jetson via on-device AWGN between the "
                         "encoder and decoder engines.")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="Mark axis 0 of inputs/outputs as dynamic (for trtexec "
                         "--shapes profile sweep). Static is faster but rebuild "
                         "per batch size.")
    ap.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh"])
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="Skip the ONNXRuntime CPU round-trip check (use only "
                         "when onnxruntime is not installed).")
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        print(f"[error] checkpoint not found: {args.checkpoint}", flush=True)
        return 1

    print(f"[setup] checkpoint: {args.checkpoint}", flush=True)
    print(f"[setup] image_dims: {tuple(args.image_dims)}, C={args.channel_symbols}",
          flush=True)
    print(f"[setup] snr_db={args.snr_db}, channel={args.channel}, opset={args.opset}",
          flush=True)
    print(f"[setup] dynamic_batch={args.dynamic_batch}", flush=True)
    print(f"[setup] out: {args.output_dir}", flush=True)

    # Need chdir into third_party/SwinJSCC for the relative imports inside its
    # encoder.py / decoder.py to resolve. Done here (not at import time) so this
    # module is safely importable from elsewhere.
    os.chdir(str(SWINJSCC_DIR))

    net = build_swinjscc(
        args.checkpoint, image_dims=tuple(args.image_dims),
        C=args.channel_symbols, channel=args.channel,
    )

    enc_path, dec_path = export_split_onnx(
        net, args.output_dir, tuple(args.image_dims),
        C=args.channel_symbols, snr_db=args.snr_db,
        opset=args.opset, dynamic_batch=args.dynamic_batch,
    )
    print(f"[ok] exported encoder + decoder", flush=True)

    # Op-level safety gate for opset-capped TensorRT builds (older JetPack).
    # Missing onnx package only warns -- it does not block the export/round-trip.
    try:
        verify_onnx_ops(enc_path, max_opset=args.opset)
        verify_onnx_ops(dec_path, max_opset=args.opset)
    except ImportError as e:
        print(f"[warn] onnx unavailable, skipping op verification: {e}",
              flush=True)

    if args.skip_roundtrip:
        print("[skip] roundtrip check skipped (--skip-roundtrip)", flush=True)
        return 0

    try:
        metrics = round_trip_check(
            net, enc_path, dec_path, tuple(args.image_dims),
            C=args.channel_symbols, snr_db=args.snr_db, n_samples=16,
        )
    except ImportError as e:
        print(f"[warn] onnxruntime unavailable, skipping round-trip: {e}",
              flush=True)
        return 0

    print(f"[roundtrip] mean PSNR (ONNX vs PyTorch) = "
          f"{metrics['mean_psnr_db']:.2f} dB (min {metrics['min_psnr_db']:.2f}, "
          f"n={metrics['n_samples']})", flush=True)
    # Acceptance criterion: ≥ 50 dB. Below that the export
    # is unsafe to hand off to trtexec.
    threshold = 50.0
    if metrics["min_psnr_db"] < threshold:
        print(f"[fail] min PSNR {metrics['min_psnr_db']:.2f} dB < threshold {threshold} dB"
              f" – investigate (likely an op falling back to a different "
              f"impl in ORT). Do not hand off to trtexec yet.", flush=True)
        return 2

    print(f"[ok] roundtrip ≥ {threshold} dB – safe to push ONNX to Jetson for trtexec",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
