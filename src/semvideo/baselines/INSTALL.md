# Classical baselines – install + run

Separated source-channel pipeline: an ffmpeg codec (H.264 / H.265) over a real LDPC code
(pyldpc, belief-propagation decoding). The paper's separated JPEG + LDPC reference (Table 8)
is produced by `scripts/eval_classical_jamming_clssim.py`, which reuses the LDPC code of
`ldpc_real.py` at block length 1024; the H.264 / H.265 + LDPC sweep below is not reported in
the paper.

## 1. Install

```bash
# Inside the project environment (or the Docker image)
pip install imageio-ffmpeg av pyldpc
```

- `imageio-ffmpeg` ships a bundled static ffmpeg (libx264 + libvpx). Enough for
  H.264, not enough for H.265.
- `pyav` is optional but recommended: gives us a Python binding to libavformat
  for stricter mp4 muxing if we ever need it (the current code uses subprocess
  pipes and `cv2.VideoCapture`).
- `pyldpc` is the LDPC encoder/decoder. Pure NumPy, deterministic by seed.

For H.265 you need a system ffmpeg with `libx265`:

- **Windows**: `winget install ffmpeg` or download from gyan.dev's full build,
  ensure `ffmpeg.exe` is on `PATH`.
- **Linux**: `sudo apt-get install ffmpeg` (Ubuntu 22.04+ ships libx265).
- **macOS**: `brew install ffmpeg`.

BPG (`bpgenc`) is not wired here; it would need a separately built binary.

## 2. Smoke test

```bash
python -m pytest tests/test_classical_baselines.py -v
```

The tests skip gracefully if `imageio-ffmpeg` or `pyldpc` are missing.

## 3. Full sweep on DJI val set

```bash
python scripts/run_classical_sweep.py \
    --frames-dir data/frames/val \
    --max-frames 32 \
    --size 256 \
    --codecs h264 h265 \
    --channels awgn erasure barrage_jam \
    --snr-grid -5 0 5 10 15 20 \
    --crf 23 35 \
    --code-rate 1/2 \
    --modulation bpsk \
    --max-bits 200000 \
    --out experiments/baselines_real/dji_val_real_classical.csv
```

Approximate wall-clock budget on a single CPU core for 32 frames @ 256×256:
- ffmpeg pass: ~0.3 s
- LDPC pass (200k bits, bp_iter=30): ~6 s
- Per cell: ~7 s
- Sweep cells: 2 codecs × 6 SNRs × 2 CRFs × 3 channels ≈ 72 cells ≈ 10 minutes.

Add `--perceptual` to also compute LPIPS + DINOv2 cls_sim (slower; needs the
DINOv2 hub model downloaded).

## 4. Full-bitstream sweep

For a sweep over the whole bitstream use:

```bash
python scripts/run_classical_sweep.py \
    --frames-dir data/frames/val \
    --max-frames 64 --size 256 \
    --codecs h264 h265 --channels awgn \
    --snr-grid -5 -2 0 2 5 8 10 12 15 18 20 \
    --crf 23 28 35 --code-rate 1/2 --modulation bpsk \
    --max-bits 0 \
    --out experiments/baselines_real/snr_sweep.csv
```

`--max-bits 0` means "push the whole MP4 bitstream through LDPC" – full
fidelity. This pass takes ~1 hour on a laptop CPU.

## 5. Known caveats

1. **Block length**. `bpsk_qpsk_ldpc` defaults to n≈240-bit LDPC blocks (the
   paper's JPEG + LDPC reference uses n = 1024). Production 5G NR-LDPC uses
   k∈[40, 8448], n∈[80, 17856]. Short blocks have a softer waterfall, so these
   curves are slightly worse than NR-LDPC: the short code handicaps the
   separated baseline, and a gap measured against it overstates the gap to a
   production code.

2. **BPSK only by default**. The script accepts `--modulation qpsk`. For
   higher-order QAM you'd need to extend `bpsk_qpsk_ldpc`; left as future work.

3. **Rayleigh fading not wired here**. The pipeline supports AWGN, Bernoulli
   bit-flip, and barrage-jam (modelled as effective-SNR drop). For Rayleigh we
   should pre-multiply BPSK symbols by complex Gaussian gain BEFORE the AWGN
   step inside `bpsk_qpsk_ldpc` – tracked as a TODO; see the docstring of
   `_apply_channel` in `codec_under_channel.py`.

4. **Speed knob**. `--max-bits N` truncates the LDPC pass at N bits so the
   sweep is interactive. Set 0 to run the full MP4 bitstream through LDPC for
   full-bitstream numbers.
