# Goal-Oriented JSCC for UAV Reconnaissance Video under Contested RF Channels

Code accompanying the paper

> Nazarii Savorona, Yaroslav Tereshchenko, Bohdan Bobyl.
> **Goal-Oriented Joint Source–Channel Coding for UAV Reconnaissance Video Under Contested RF
> Channels with Detector-Feature Distillation.** *Drones* (MDPI), 2026.

- Paper DOI: **TBD** (to be filled in when the article is published)
- Repository URL: https://github.com/nazarsavorona/goal-oriented-jscc-uav

The repository holds what the paper uses and nothing else: the compact codec, the channel
models with their verification harness, the losses, the training and evaluation scripts, the
separated baselines, the figure builders, the Docker build, and the bicubic-control harness
that the paper's Data Availability Statement refers to. It contains no model weights, no
datasets and no third-party network code.

Corresponding author: Nazarii Savorona, nazarii.savorona@knu.ua

## What is here

| Path | Contents |
|---|---|
| `src/semvideo/models/deep_jscc.py` | the compact convolutional DeepJSCC-style codec (5.5 M parameters at the paper's settings) |
| `src/semvideo/channel/` | differentiable channel models: AWGN (`awgn.py`), Rayleigh fading (`fading.py`), barrage / pulsed / swept / single-tone jammers (`jamming.py`), the reactive jammer with its emission-following, matched-energy and mean-referenced power conventions (`jamming_reactive.py`), i.i.d. and Gilbert–Elliott burst symbol erasure (`packet_loss.py`), and the wrappers that add erasure or reactive jamming to the channel layer of the SwinJSCC-capacity codec during training (`erasure_wrapped.py`, `reactive_jam_wrapped.py`) |
| `src/semvideo/training/` | loss terms: three-scale SSIM (`losses.py`), frozen-detector feature distillation (`detector_feature_loss.py`), DINOv2 patch features (`dinov2_feature_loss.py`), SigLIP patch features (`scenegraph_feature_loss.py`; the file and flag names are historical, see below) |
| `src/semvideo/baselines/` | separated source-channel baseline: ffmpeg codecs and a real LDPC code (see `INSTALL.md` there) |
| `src/semvideo/eval/`, `src/semvideo/metrics/` | paired bootstrap statistics; PSNR / SSIM / LPIPS; the DINOv2 class-token similarity and 8-bit PSNR of the jamming tables (`dinov2_clssim.py`) |
| `scripts/train.py` | training entry point of the compact codec (argparse flags) |
| `scripts/train_swinjscc_dji.py` | trainer of the SwinJSCC-capacity codec chain (imports upstream SwinJSCC, not included) |
| `scripts/mamba_matched_pair.py`, `configs/mambajscc/` | the MambaJSCC matched pair and resolution control of Appendix C.1, with the training configurations of the two DJI trunks (imports upstream MambaJSCC, not included) |
| `scripts/dinov2_metric.py` | the DINOv2 class-token / patch-token similarity metric with the **bicubic no-codec control** |
| `scripts/eval_*.py`, `scripts/analyse_*.py`, `scripts/run_*.py`, `scripts/measure_*.py`, `scripts/diag_*.py` | the evaluators behind the paper's tables (see the mapping below) |
| `scripts/rf_stage1_dump_latents.py`, `rf_stage2_sionna_link.py`, `rf_stage3_decode_score.py` | the frequency-domain OFDM simulation, in three stages |
| `scripts/baseline_under_jamming.py`, `run_classical_clssim_sweep.py`, `run_classical_sweep.py`, `eval_classical_jamming_clssim.py`, `eval_video_codec_baseline.py` | separated JPEG + LDPC baseline and the video-codec (H.264 / H.265 / AV1) baseline |
| `scripts/build_*.py` | figure builders, and `build_visdrone_yolo.py`, which converts VisDrone annotations to the YOLO format |
| `scripts/extract_diverse_frames.py` | frame extraction and the per-source-video temporal train / validation split |
| `scripts/verify_channel_fidelity*.py` | numerical checks of the channel models against their analytical references |
| `scripts/export_tensorrt.py`, `scripts/jetson_*.py`, `scripts/latency_profile_512.py` | ONNX export and latency measurement |
| `tests/` | 200 unit tests of the channel models, the codec, the losses, the metrics, the statistics and the baselines |

Layout: 9 files at the root, 42 scripts, 27 files under `src/semvideo/`, 2 files under `configs/`, 15 files under `tests/`.

## Installation

The Docker image is the reference environment (PyTorch 2.6, CUDA 12.4). It was built and
smoke-tested on 23 November 2026 with ultralytics 8.4.47, opencv-python and
opencv-python-headless 5.1.0.95, transformers 5.24.0, timm 1.0.31 and scipy 1.17.1; later
builds may resolve newer versions of the packages that the Dockerfile does not pin.

```bash
docker build -t semvideo:latest .

docker run --rm --gpus all --ipc=host \
    -v "$PWD":/workspace \
    -e SEMVIDEO_GIT_SHA="$(git rev-parse HEAD)" \
    -v /path/to/data:/data:ro \
    -v /path/to/third_party:/third_party:ro \
    -w /workspace semvideo:latest \
    python scripts/train.py --help
```

On hosts that use the NVIDIA runtime instead of `--gpus`, pass `--runtime=nvidia`; this is
what `docker-compose.yml` does (`DATA_DIR=/path/to/data docker compose run --rm train ...`).

The image contains no `.git`, so `scripts/train.py` records the commit in `config.json` only
when it can run `git rev-parse HEAD` on a mounted repository. Otherwise, for example under
`docker compose`, which mounts the data, `experiments/` and `weights/` but not the
repository, or when git refuses a mount owned by another user, it uses `SEMVIDEO_GIT_SHA`
and writes `unknown` if that is not set. Pass `-e SEMVIDEO_GIT_SHA=$(git rev-parse HEAD)` to
record it in every case.

Without Docker: Python 3.11 or newer, `pip install -e ".[dev]"`, plus the packages listed in
the `Dockerfile` (`ultralytics`, `timm`, `transformers`, `torchmetrics` and others) for the
detector-feature loss and the evaluators. The classical baselines need
`pip install -e ".[baselines]"` and a system `ffmpeg`.

Weights & Biases logging is optional: pass `--no-wandb`, or copy `.env.example` to `.env`
and fill in your own key.

### Where scripts look for data and third-party code

`scripts/train.py` takes `--data-dir` (default `data/frames`, relative to the working
directory); every root must contain `train/` and `val/` subdirectories of frames.

The scripts of the SwinJSCC-capacity chain resolve their defaults against the **parent
directory of the repository**: `<parent>/data/...` and `<parent>/third_party/SwinJSCC`. With
the repository mounted at `/workspace`, as above, these are `/data/...` and
`/third_party/SwinJSCC`. Every such default has a command-line flag, and three environment
variables override the remaining fixed locations:

| Variable | Meaning |
|---|---|
| `SWINJSCC_DIR` | directory of the upstream SwinJSCC code (see "Third-party code") |
| `SEMVIDEO_SOURCE_VIDEO` | a 1080p source video for `eval_swinjscc_dinov2.py` and `latency_profile_512.py` |
| `SEMVIDEO_RECORDS_DIR` | evaluation records read by `build_paper1_threat_figures.py` (default `records/`) |

## Main commands

All commands run from the repository root. The values below illustrate the interface. The
arguments of every training run reported in the paper are stored with its checkpoint
(`config.json`, or the argument dictionary inside the checkpoint) and are part of the
archived evaluation records (Zenodo).

```bash
# unit tests (inside the Docker image run `pip install pytest` first if it is missing)
python -m pytest tests -q

# channel-model verification against analytical references
python scripts/verify_channel_fidelity.py

# compact codec: detector-feature arm and pixel-only arm of a matched pair
python scripts/train.py --model deepjscc --channel awgn --loss perceptual_yolo \
    --yolo-weights yolov8m.pt --yolo-feat-weight 0.5 --img-size 256 --epochs 50 \
    --data-dir /data/dji/frames_512_v1 --seed 42 --no-wandb --exp-name pair_detfeat
python scripts/train.py --model deepjscc --channel awgn --loss mse --img-size 256 --epochs 50 \
    --data-dir /data/dji/frames_512_v1 --seed 42 --no-wandb --exp-name pair_pixel

# paired lift of the two arms with a per-frame bootstrap interval
python scripts/eval_r1_paired.py <detfeat_ckpt> <pixel_ckpt> /data/dji/frames_512_v1/val 375 20

# SwinJSCC-capacity codec (needs the upstream code, see below); later stages continue
# from a checkpoint with --init-from and add --train-erasure-p-max or --train-reactive-jam-prob
python scripts/train_swinjscc_dji.py --train-dir /data/dji/frames_512_v1/train \
    --val-dir /data/dji/frames_512_v1/val --out /workspace/experiments/swin_stage1

# the reactive-augmented model (Table 2, row 8) adds --train-reactive-jam-prob 0.5 with
# JNR ~ U[5, 15] dB; its training-time jammer is the mean-referenced one, which is the
# default of --train-reactive-jam-power-reference (see "Reactive-jammer conventions")

# benchmark sweep (Kodak / DIV2K / VisDrone / any image folder), AWGN or Rayleigh
python scripts/eval_benchmark_sweep.py --checkpoint <ckpt> --benchmark-root /data/kodak \
    --label MYRUN_on_Kodak --snrs 1 4 7 10 13 20

# jamming sweep, optionally with a receiver-side front-end (--rx-defense clip|excise)
python scripts/eval_jamming_cross_corpus.py --p49-ckpt <ckpt> --snr 20 \
    --jammer-types barrage pulsed single_tone swept reactive --jnr-grid 0 5 10 15 --out <dir>

# symbol-erasure sweep
python scripts/eval_erasure_cross_corpus.py --ckpt <abs path> --label P68 \
    --p-grid 0.1 0.3 0.5 --snr 10 --out <abs dir>

# separated JPEG + LDPC baseline under the same jammers (Table 8 settings: 30 frames,
# 15 BP iterations; the script defaults are 100 frames and 30 iterations); it does not
# need the upstream SwinJSCC code
python scripts/eval_classical_jamming_clssim.py --frames-dir /data/dji/frames_512_v1/val \
    --label dji --n-frames 30 --bp-iter 15 --snr-grid 0 5 10 15 20 25 --jnr 10 \
    --ldpc-arm real --out <dir>

# ground-truth detection on VisDrone2019-DET test-dev, then offline scoring
python scripts/eval_visdrone_gt_detection.py --out <dir> --codec p49=<ckpt> --cond awgn:20
python scripts/analyse_visdrone_gt_detection.py <dir> --boot 2000
```

Checkpoint tags that appear in defaults and comments: `ORIG` – the published SwinJSCC
weights, evaluated zero-shot; `P49` – the two-stage cross-corpus codec; `P68` – its
erasure-augmented continuation; `P80` – the reactive-jam-augmented variant; `P12` to `P15` –
compact-codec checkpoints at 512 px (the glob patterns in `dinov2_metric.py` and
`latency_profile_512.py`).

### Options of `scripts/train.py`

`--model deepjscc` (the only codec in this release); `--loss mse` or `--loss perceptual_yolo`
(MSE + three-scale SSIM + frozen-detector feature L1, with `--yolo-feat-weight`,
`--yolo-feat-warmup`, `--yolo-weights`); `--channel` one of `awgn`, `rayleigh`, `mixed`,
`jam_barrage`, `jam_pulsed`, `mixed_jam` (with `--jnr-db`, `--duty-cycle`), and
`--erasure-p-max` for training-time symbol erasure; `--seed` and `--cudnn-deterministic` for
reproducible runs. Every run writes `config.json`, `metrics.csv`, `best.pt`, `last.pt` and
image snapshots under `experiments/<name>_<timestamp>/`.

With `--loss perceptual_yolo` the trainer also uses `--yolo-feat-warmup` (default 3 epochs)
as the length of the learning-rate warm-up, in place of `--warmup-epochs` (default 15). The
paper states this in Section 3.7 and measures its effect with a control; the code is left
exactly as it was when the reported runs were made.

### Reactive-jammer conventions

`ReactiveJammer` (`src/semvideo/channel/jamming_reactive.py`) runs one detect-then-hold
automaton and takes the power convention as `power_reference` (Appendix A of the paper):

- `followed` – emission-following: the interference on each gated symbol is scaled to that
  symbol's own energy. This is the default of the evaluators and the setting behind every
  reactive number of the paper that does not name another one.
- `matched_barrage` – the same gating with the injected energy equalised to an equal-JNR
  barrage.
- `mean` – mean-referenced: the interference on each gated symbol is scaled to the mean
  symbol energy of the block. The reactive-augmented model (Table 2, row 8) was trained
  against this convention, and the paper quotes the values measured in it as such
  (Section 4.4). It is the default of `--train-reactive-jam-power-reference` in
  `train_swinjscc_dji.py`, because the stored arguments of that model predate the flag;
  in `eval_jamming_cross_corpus.py` it is `--reactive-power-reference mean`.

### Names that differ from the paper

- The SigLIP patch-feature term of the SwinJSCC-capacity objective is
  `src/semvideo/training/scenegraph_feature_loss.py`, switched on with
  `--scenegraph-feat-weight 0.05 --scenegraph-model siglip` in `train_swinjscc_dji.py`. The
  names are kept because the arguments stored with the paper's checkpoints use them.
- The DINOv2 term is `--dino-feat-weight 0.05 --dino-model dinov2_vitb14`.
- The stored argument dictionaries were written by a larger internal version of the two
  trainers. They contain further keys, all at their inactive defaults; the options behind
  those keys are not part of this release, and the released trainers reproduce the behaviour
  of the inactive defaults.

### The bicubic-control harness

`scripts/dinov2_metric.py` computes the DINOv2 ViT-S/14 class-token and patch-token cosine
similarity between an original 1080p frame and its reconstruction. Before any codec is
scored it runs the **bicubic control**: the frame is downsampled to 256 px and to 512 px and
upsampled back with bicubic interpolation, with no codec and no channel, which gives the
similarity that the resolution change alone costs. The control needs only a source video.
The defaults of `--frame-start` and `--frame-end` (10850 and 11800) select a segment of the
authors' source video, so for any other video pass a frame range that it contains:

```bash
python scripts/dinov2_metric.py --source /path/to/video.mp4 --frame-start 0 --frame-end 900 \
    --n-frames 30 --out experiments/dinov2_metric
```

Codec rows are added for every compact-codec checkpoint that matches the `CHECKPOINTS` list
at the top of the script.

## Paper tables and figures

Numbering follows the proof of the accepted manuscript and is to be re-checked against the
published version (**TBD**). "Docstring" means that the script's own docstring describes the
measurement; where a row says "see script docstring", the mapping is not pinned by the code
and should be read from there.

| Paper item | Script(s) | Note |
|---|---|---|
| Table 1 (sensitivity to the detector-feature weight) | `train.py`, `eval_deepjscc_cls.py`, `eval_r1_paired.py` | one training run per weight via `--yolo-feat-weight`; see script docstring |
| Table 2 (training protocol of the SwinJSCC-capacity codec) | `train_swinjscc_dji.py` | the table lists the arguments stored in each checkpoint; stages 1 and 2 predate this trainer (Section 3.4); row 8 trains against the mean-referenced reactive jammer (see "Reactive-jammer conventions") |
| Table 3 (matched pairs on the compact codec) | `eval_r1_paired.py` | docstring: paired lift with a per-frame bootstrap interval |
| Table 4 (decomposition of the lift) | `train.py`, `eval_deepjscc_cls.py` | in this trainer an MSE + MS-SSIM arm without the detector term is `--loss perceptual_yolo --yolo-feat-weight 0`; see script docstring |
| Table 5 (lift across the threat suite) | `eval_compact_jamming_erasure.py` | docstring: compact codec across the jamming and erasure grid |
| Table 6 (six corpus × channel cells) | `eval_benchmark_sweep.py` | docstring: benchmark folders × {AWGN, Rayleigh} |
| Table 7 (detector-supervisor comparison) | `eval_supablation_paired.py`, `diag_supervisor_loss_scale.py` | docstring: all arms on one frame list; loss-scale check |
| Table 8 (separated JPEG + LDPC against the codec) | `eval_classical_jamming_clssim.py` (with `run_classical_clssim_sweep.py`, `baseline_under_jamming.py`), `eval_jamming_cross_corpus.py` | docstring: real LDPC arm and analytic threshold arm |
| Table 9 (cost of each jammer mode) | `eval_jamming_cross_corpus.py` | docstring |
| Table 10 (reactive-jam sweep with receiver-side baselines) | `eval_jamming_cross_corpus.py --rx-defense` | see script docstring |
| Table 11 (erasure robustness) | `eval_erasure_cross_corpus.py` | docstring |
| Figure 1 (system overview) | – | schematic, no script |
| Figure 2 (dataset samples) | `build_dataset_samples_figure.py` | needs the DJI validation frames |
| Figure 3 (reconstruction montage) | `build_recon_montage_figure.py`, `build_recon_montage_swin.py` | the script that composes the two sets of panels is not included |
| Figure 4 (reactive-jam generality ladder) | data: `eval_jamming_cross_corpus.py` | the plotting script is not included |
| Figure 5 (jamming heatmap) | `build_paper1_threat_figures.py` | reads evaluation records, see below |
| Figure 6 (OFDM jammer-mode ladder) | data: `rf_stage1_dump_latents.py`, `rf_stage2_sionna_link.py`, `rf_stage3_decode_score.py` | the plotting script is not included; stage 2 runs in a separate Sionna container |
| Figure 7 (erasure curves) | `build_paper1_threat_figures.py` | reads evaluation records, see below |
| Metric anchored to detection | `eval_clssim_detection_anchor.py`, `eval_visdrone_gt_detection.py` (uses `eval_visdrone_map.py`), `analyse_visdrone_gt_detection.py` | docstrings |
| SwinJSCC anchor reproduction on Kodak | `run_swinjscc_kodak.py` | docstring |
| Rician / Doppler / imperfect-CSI sensitivity | `eval_rician_doppler.py` | docstring |
| Video-codec baselines (H.264 / H.265 / AV1) | `eval_video_codec_baseline.py` | docstring |
| Reactive jammer duty cycle and injected power | `measure_reactive_duty.py` | docstring |
| CLIP cross-check of the metric | `eval_deepjscc_cls_clip.py`, `eval_benchmark_sweep.py --metric-backbone clip_vitb32` | docstrings |
| Paired significance of two evaluation runs | `eval_paired_bootstrap.py` | docstring |
| Dataset construction (stride sampling, temporal split) | `extract_diverse_frames.py` | docstring |
| Deployment cost | `latency_profile_512.py`, `export_tensorrt.py`, `jetson_make_reference.py`, `jetson_inference_bench.py` | TensorRT engines are built on the target with `trtexec` |
| Channel-model verification | `verify_channel_fidelity.py`, `verify_channel_fidelity_part2.py`, `tests/test_channel*.py` | docstrings |
| Objective transfer to a MambaJSCC trunk, and its resolution control (Appendix C.1) | `mamba_matched_pair.py`, `configs/mambajscc/` | needs the upstream MambaJSCC code; see "MambaJSCC matched pair" |

`build_paper1_threat_figures.py` opens three sets of evaluation records at import time
(`erasure_figure_source/`, `gt_detection/`, `heldout_testdev/`). They are part of the
archived evaluation records on Zenodo, not of this repository. In that record's records
archive the three sit together in `records/docs/research/reviews_2026_09/`; unpacked at the
repository root, the script finds them there. Otherwise place them directly under
`records/`, or point `SEMVIDEO_RECORDS_DIR` at the directory that contains them.

### MambaJSCC matched pair

`scripts/mamba_matched_pair.py` runs the matched continuation of Appendix C.1: both arms
continue one PSNR-only MambaJSCC trunk trained on the DJI frames for the same 30 epochs with
the same seed and data order, one under MambaJSCC's own MSE objective and one under the
paper's objective (MSE, MS-SSIM at 0.15 and the frozen-YOLOv8m feature term at 0.5, on
MambaJSCC's per-sample-sum scale). Before training it evaluates the trunk on the DJI
validation split at AWGN SNR = 20 dB and stops if the result is not the initialization the
paper reports; `--epochs 0` runs only that evaluation, which is how the resolution control
of the same appendix measures the 256 px and 512 px trunks.

```bash
# the pair, on the paper's three seeds (1024 is MambaJSCC's default)
python scripts/mamba_matched_pair.py --loss DETFEAT --epochs 30 --seed 1024 --out-tag detfeat_s1024
python scripts/mamba_matched_pair.py --loss PSNR    --epochs 30 --seed 1024 --out-tag psnrctl_s1024
# ... and the same with --seed 7 and --seed 1337

# resolution control: a trunk evaluated at 512 px input
python scripts/mamba_matched_pair.py --loss PSNR --epochs 0 --skip-init-gate --out-tag eval_512 \
    --train-config configs/mambajscc/vssm_tiny_DJI512.yaml \
    --init-encoder <encoder.pt> --init-decoder <decoder.pt> --eval-snrs 20
```

What it needs and this repository does not include:

- the upstream MambaJSCC code (<https://github.com/Wireless3C-SJTU/MambaJSCC>) at
  `$MAMBAJSCC_DIR`, by default `<parent of this repository>/third_party/MambaJSCC`, in an
  environment where its selective-scan CUDA kernels are built (the upstream instructions),
  with `ultralytics` and `transformers` added; a CUDA GPU. The script imports MambaJSCC's
  configuration, data loader, channel, network, MS-SSIM and scheduler from that checkout and
  changes the working directory to it; relative paths given on the command line are resolved
  before that;
- the PSNR-only trunks, trained with upstream `main.py` on the training configurations in
  `configs/mambajscc/` (`vssm_tiny_DJI.yaml`, 256 px; `vssm_tiny_DJI512.yaml`, 512 px)
  together with upstream's vssm_tiny model configuration `configs/vssm/vssm_tiny_DIV2K.yaml`.
  Upstream `main.py` reads `configs/vssm/vssm_tiny_<name>.yaml` and
  `configs/train/vssm_tiny_<name>.yaml` for the `config_name` set in its source, so copy both
  files into the checkout under one name. The data directories of the configurations are
  ImageFolder roots, each holding one subdirectory with the DJI frames of that split.

## Data

No data is distributed with this repository.

- **DJI Mavic 3T corpus** (recorded by the authors) – not public.
- **Operator-supplied footage** – not public.
- **VisDrone2019-DET** – obtain from its owners (<https://github.com/VisDrone/VisDrone-Dataset>);
  used under CC BY-NC-SA 3.0, imagery is not redistributed.
- **DIV2K** and **Kodak** – obtain from their owners under their standard academic-benchmark terms.

Requests concerning the non-public material go to the corresponding author. Default paths
such as `/data/dji/frames_512_v1/val` name the layout used in the experiments and exist only
where that material has been obtained.

## Third-party code and licences

Nothing third-party is copied into this repository; each component is obtained from its
owner under its own licence.

- **SwinJSCC** – the network code is imported from the upstream repository
  <https://github.com/semcomm/SwinJSCC> and is **not** included here. Clone it to
  `<parent of this repository>/third_party/SwinJSCC` (`/third_party/SwinJSCC` in the Docker
  layout above), or anywhere else and set `SWINJSCC_DIR`. The scripts import
  `net.network.SwinJSCC` from it and change the working directory to it, so pass absolute
  paths for checkpoints and outputs. The published weights used as the zero-shot reference
  are expected at `checkpoint/SwinJSCC/SwinJSCC w_ SA/` inside that directory; obtain them
  from the upstream authors. Check the upstream repository for its terms of use.
- **MambaJSCC** – used only by `scripts/mamba_matched_pair.py` and **not** included here.
  Clone <https://github.com/Wireless3C-SJTU/MambaJSCC> to
  `<parent of this repository>/third_party/MambaJSCC`, or anywhere else and set
  `MAMBAJSCC_DIR` (see "MambaJSCC matched pair"). The upstream repository states no licence;
  check its terms of use with its authors.
- **YOLOv8 / Ultralytics** – AGPL-3.0. Installed as the `ultralytics` package; detector
  weights (`yolov8m.pt` and the other supervisors) are downloaded by that package and are
  not redistributed. The ground-truth detection scripts use the publicly released YOLOv8s
  detector trained on VisDrone2019-DET train that the paper uses, `mshamrai/yolov8s-visdrone`
  on Hugging Face (licence tag on its model card: OpenRAIL), placed at
  `weights/yolov8s-visdrone.pt` (`--yolo`); it is not redistributed here.
  `scripts/build_visdrone_yolo.py` converts the VisDrone annotations to the YOLO format for
  anyone who trains a replacement; a replacement does not reproduce the paper's values.
- **DINOv2** – Apache-2.0 (loaded from `facebookresearch/dinov2` or `facebook/dinov2-small`).
- **SigLIP** – Apache-2.0 (`google/siglip-*`). The same loss module can also load
  **OWLv2** (Apache-2.0, `google/owlv2-*`), which the paper does not use.
- **CLIP** – MIT (`openai/clip-vit-base-patch32`).
- **VGG-16** ImageNet weights from `torchvision` – used by the LPIPS-style validation metric.
- **pyldpc**, **PyAV** and **ffmpeg** – used by the separated baselines
  (`src/semvideo/baselines/INSTALL.md`).
- **Sionna** – Apache-2.0; used only by `rf_stage2_sionna_link.py`, in its own container.

## Archived evaluation records

The evaluation records, the verification scripts that recompute the paper's tables from
them, the stored training arguments, ten trained checkpoints of the SwinJSCC-capacity codec
and the per-image detector outputs of the VisDrone2019-DET test-dev runs are deposited in a
Zenodo record, DOI [10.5281/zenodo.24247227](https://doi.org/10.5281/zenodo.24247227). The
files of that record are not public: access is restricted and is granted by the
corresponding author on written request. The record is related data, not an archive of this
software release.

## Licence

This repository is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**;
the full text is in [`LICENSE`](LICENSE). The detector-feature loss calls Ultralytics YOLOv8,
which is itself licensed under AGPL-3.0. The licence covers the code in this repository
only, not the third-party components listed above, the datasets, or any model weights.

## Citation

Please cite the paper (the DOI is a placeholder until publication):

```bibtex
@article{savorona2026goaljscc,
  author  = {Savorona, Nazarii and Tereshchenko, Yaroslav and Bobyl, Bohdan},
  title   = {Goal-Oriented Joint Source--Channel Coding for {UAV} Reconnaissance Video Under
             Contested {RF} Channels with Detector-Feature Distillation},
  journal = {Drones},
  year    = {2026},
  doi     = {TBD}
}
```
