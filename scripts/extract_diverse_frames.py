"""Extract diverse frames from DJI drone videos using PySceneDetect.

Instead of uniform stride extraction (which produces many near-duplicate frames),
this script:
1. Detects scene changes using ContentDetector (visual content analysis)
2. Extracts frames at scene boundaries + adaptive sampling within scenes
3. Computes frame-to-frame similarity to skip near-duplicates
4. Produces a more diverse training set from the same video data

Usage:
    cd semvideo
    python scripts/extract_diverse_frames.py
    python scripts/extract_diverse_frames.py --threshold 20 --min-scene-len 30
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False
    print("WARNING: scenedetect not installed, falling back to stride-based extraction")


def compute_frame_hash(frame: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Compute perceptual hash (average hash) for duplicate detection."""
    small = cv2.resize(frame, (hash_size, hash_size))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = gray.mean()
    return (gray > mean).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute normalized Hamming distance between two hashes (0=identical, 1=opposite)."""
    return np.mean(h1 != h2)


def estimate_video_sharpness(video_path: str, samples: int = 10) -> float:
    """Estimate mean Laplacian variance across uniformly sampled frames.

    Used as a video-level quality gate to skip blurry/static promo clips
    before running expensive scene detection.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0.0

    indices = np.linspace(0, total - 1, min(samples, total), dtype=int)
    values = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    return float(np.mean(values)) if values else 0.0


def is_solid_color_frame(frame: np.ndarray, std_threshold: float = 5.0) -> bool:
    """Detect signal-loss / blank frames where all pixels are nearly identical.

    Drone FPV streams produce solid blue/green/black frames during signal
    loss; these carry no information and just bloat the dataset.
    """
    return float(frame.std()) < std_threshold


def is_low_information_frame(
    frame: np.ndarray,
    std_threshold: float = 5.0,
    min_laplacian_var: float = 30.0,
    min_entropy: float = 4.5,
) -> bool:
    """Reject frames that won't teach the model anything.

    Composite gate that catches three failure modes seen in real drone footage:
    1. Solid-color signal-loss frames (std < threshold).
    2. Out-of-focus or texture-only ploughed-field frames (Laplacian var low).
    3. Black/dark night-zoom frames with collapsed histograms (entropy low).

    Defaults are tuned for 128x128-or-larger drone frames; loosen for very
    smooth content like drone hover over still water.
    """
    if float(frame.std()) < std_threshold:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < min_laplacian_var:
        return True
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total <= 0:
        return True
    p = hist / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    return entropy < min_entropy


def detect_scenes(video_path: str, threshold: float = 25.0, min_scene_len: int = 15) -> list:
    """Detect scene boundaries in a video using content-based detection.

    Args:
        video_path: Path to video file.
        threshold: ContentDetector threshold (higher = fewer scenes). Default 25.
        min_scene_len: Minimum scene length in frames.

    Returns:
        List of (start_frame, end_frame) tuples for each scene.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for scene in scene_list:
        start = scene[0].get_frames()
        end = scene[1].get_frames()
        scenes.append((start, end))

    return scenes


def extract_scene_frames(
    video_path: str,
    scenes: list,
    output_dir: Path,
    target_size: int = 128,
    frames_per_scene: int = 3,
    similarity_threshold: float = 0.20,
    prefix: str = "",
    max_per_prefix: int = 0,
    min_laplacian_var: float = 30.0,
    min_entropy: float = 4.5,
    max_recent: int = 300,
) -> int:
    """Extract diverse frames from detected scenes.

    For each scene:
    - Always extract first and last frame (scene boundaries)
    - Sample additional frames uniformly within scene
    - Skip frames too similar to previously extracted ones (perceptual hash)
    - Skip frames that fail the low-information gate (blur/entropy)
    - Stop extracting from this video once ``max_per_prefix`` frames are saved
      (0 = unlimited) – used to defeat single-video dominance.

    Args:
        video_path: Path to video.
        scenes: List of (start, end) frame tuples.
        output_dir: Directory to save extracted frames.
        target_size: Resize frames to this square size.
        frames_per_scene: Target frames per scene.
        similarity_threshold: Skip frame if Hamming distance < this to any recent frame.
        prefix: Filename prefix (e.g., mission_camera).
        max_per_prefix: Cap on frames to extract from this video (0 = unlimited).
        min_laplacian_var: Minimum focus measure for a frame to be kept.
        min_entropy: Minimum grayscale histogram entropy (bits) to be kept.
        max_recent: Window size for perceptual-hash duplicate detection.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    recent_hashes: list[np.ndarray] = []

    for scene_idx, (start, end) in enumerate(scenes):
        if max_per_prefix and extracted >= max_per_prefix:
            break
        scene_len = end - start
        if scene_len < 3:
            continue

        # Sample frame indices within this scene
        if scene_len <= frames_per_scene:
            sample_indices = list(range(start, end))
        else:
            step = scene_len / frames_per_scene
            sample_indices = [int(start + i * step) for i in range(frames_per_scene)]
            # Always include scene boundaries
            if start not in sample_indices:
                sample_indices[0] = start
            if end - 1 not in sample_indices:
                sample_indices[-1] = end - 1

        for frame_idx in sample_indices:
            if max_per_prefix and extracted >= max_per_prefix:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if is_low_information_frame(
                frame, min_laplacian_var=min_laplacian_var, min_entropy=min_entropy,
            ):
                continue

            # Compute hash and check similarity
            frame_hash = compute_frame_hash(frame)
            is_duplicate = False
            for prev_hash in recent_hashes:
                if hamming_distance(frame_hash, prev_hash) < similarity_threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Resize and save
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            # Center crop to square, then resize
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            cropped = frame[top : top + min_dim, left : left + min_dim]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

            filename = f"{prefix}_scene{scene_idx:04d}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            recent_hashes.append(frame_hash)
            if len(recent_hashes) > max_recent:
                recent_hashes.pop(0)
            extracted += 1

    cap.release()
    return extracted


def extract_uniform_fallback(
    video_path: str,
    output_dir: Path,
    target_size: int = 128,
    stride: int = 30,
    similarity_threshold: float = 0.20,
    prefix: str = "",
    max_per_prefix: int = 0,
    min_laplacian_var: float = 30.0,
    min_entropy: float = 4.5,
    max_recent: int = 300,
) -> int:
    """Fallback: extract frames with stride + duplicate + low-information filtering.

    Used when scene detection finds no cuts (slow drone hover, static scenes).
    Default stride raised to 30 (was 5) so a 25-minute static clip yields
    hundreds rather than thousands of redundant frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    recent_hashes: list[np.ndarray] = []

    for frame_idx in range(0, total_frames, stride):
        if max_per_prefix and extracted >= max_per_prefix:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if is_low_information_frame(
            frame, min_laplacian_var=min_laplacian_var, min_entropy=min_entropy,
        ):
            continue

        frame_hash = compute_frame_hash(frame)
        is_duplicate = any(
            hamming_distance(frame_hash, h) < similarity_threshold
            for h in recent_hashes[-max_recent:]
        )
        if is_duplicate:
            continue

        h, w = frame.shape[:2]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        cropped = frame[top : top + min_dim, left : left + min_dim]
        resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

        filename = f"{prefix}_frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_dir / filename), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        recent_hashes.append(frame_hash)
        extracted += 1

    cap.release()
    return extracted


def _extract_one_video(
    video_path: str,
    output_dir: Path,
    prefix: str,
    *,
    stride: int,
    size: int,
    threshold: float,
    min_scene_len: int,
    frames_per_scene: int,
    similarity: float,
    max_per_prefix: int,
    min_laplacian_var: float,
    min_entropy: float,
) -> int:
    """Extract frames from a single video using stride or scene-based sampling.

    When ``stride > 0`` we walk the video at fixed intervals and let the dedup
    + low-information gates filter out near-duplicates and uninformative frames.
    For drone footage this is dramatically more productive than scene-based
    sampling (scene boundaries on continuous drone flights are mostly artificial
    and cap each segment at frames_per_scene regardless of segment length).

    When ``stride == 0`` and PySceneDetect is available we fall back to the
    scene-based path with a uniform fallback if no cuts are detected. Useful
    only for actual cut-edited media; not recommended for drone capture.
    """
    if stride > 0:
        return extract_uniform_fallback(
            video_path, output_dir, target_size=size, stride=stride,
            similarity_threshold=similarity, prefix=prefix,
            max_per_prefix=max_per_prefix,
            min_laplacian_var=min_laplacian_var,
            min_entropy=min_entropy,
        )
    if HAS_SCENEDETECT:
        scenes = detect_scenes(video_path, threshold, min_scene_len)
        if scenes:
            return extract_scene_frames(
                video_path, scenes, output_dir,
                target_size=size, frames_per_scene=frames_per_scene,
                similarity_threshold=similarity, prefix=prefix,
                max_per_prefix=max_per_prefix,
                min_laplacian_var=min_laplacian_var,
                min_entropy=min_entropy,
            )
    # Either no scenedetect or zero scenes detected
    return extract_uniform_fallback(
        video_path, output_dir, target_size=size, stride=30,
        similarity_threshold=similarity, prefix=prefix,
        max_per_prefix=max_per_prefix,
        min_laplacian_var=min_laplacian_var,
        min_entropy=min_entropy,
    )


def process_flat_folder(
    data_dir: Path,
    output_base: Path,
    size: int,
    threshold: float,
    min_scene_len: int,
    frames_per_scene: int,
    similarity: float,
    min_sharpness: float,
    stride: int = 60,
    max_per_prefix: int = 0,
    min_laplacian_var: float = 20.0,
    min_entropy: float = 4.0,
    extensions: tuple[str, ...] = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"),
) -> dict[str, int]:
    """Extract frames from videos sitting flat in a single folder.

    Used for non-DJI sources where videos aren't organized into mission/camera
    subdirs. Reuses the same scene detection + perceptual hash dedup as the
    DJI flow. Skips videos whose mean sharpness falls below ``min_sharpness``
    (drops promo/static clips like SKYDEFENDER) and per-frame solid-color
    artifacts (signal-loss frames).
    """
    videos = sorted(p for p in data_dir.iterdir() if p.suffix in extensions)
    print(f"Flat mode: scanning {len(videos)} videos in {data_dir}")

    stats: dict[str, int] = {}
    for vpath in videos:
        sharp = estimate_video_sharpness(str(vpath))
        prefix = vpath.stem
        if sharp < min_sharpness:
            print(f"\n  SKIP {vpath.name}: sharpness {sharp:.0f} < {min_sharpness}")
            stats[f"{prefix}_skipped_low_sharpness"] = 0
            continue

        cap = cv2.VideoCapture(str(vpath))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        duration = n_frames / fps if fps > 0 else 0
        print(f"\n  {vpath.name}: {n_frames} frames, {duration:.0f}s, sharpness={sharp:.0f}")

        n_extracted = _extract_one_video(
            str(vpath), output_base / "train", prefix,
            stride=stride, size=size, threshold=threshold,
            min_scene_len=min_scene_len, frames_per_scene=frames_per_scene,
            similarity=similarity, max_per_prefix=max_per_prefix,
            min_laplacian_var=min_laplacian_var, min_entropy=min_entropy,
        )
        print(f"    Extracted {n_extracted} diverse frames -> train/")
        stats[prefix] = n_extracted
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diverse frames from drone videos")
    parser.add_argument("--data-dir", default="data/dji")
    parser.add_argument("--output-dir", default="data/dji/frames_128_v3")
    parser.add_argument("--size", type=int, default=128, help="Output frame size")
    parser.add_argument("--threshold", type=float, default=3.0, help="Scene detection threshold (low for drone footage)")
    parser.add_argument("--min-scene-len", type=int, default=15, help="Min scene length in frames")
    parser.add_argument("--frames-per-scene", type=int, default=3,
                        help="Frames to extract per scene (was 5; lowered to fight redundancy)")
    parser.add_argument("--similarity", type=float, default=0.20,
                        help="Duplicate hash threshold (was 0.08; raised to catch slow-hover near-dupes)")
    parser.add_argument("--cameras", default="W,Z", help="Camera types to use (S=IR, T=thermal, W=wide, Z=zoom)")
    parser.add_argument("--night-cameras", default="W",
                        help="Cameras to use for night missions only – Z is useless at night")
    parser.add_argument("--flat", action="store_true",
                        help="Treat data-dir as a flat folder of videos (no DJI mission structure)")
    parser.add_argument("--min-sharpness", type=float, default=40.0,
                        help="(--flat only) Skip videos whose mean Laplacian variance is below this")
    parser.add_argument("--max-per-prefix", type=int, default=300,
                        help="Cap frames extracted per source video – defeats single-clip dominance (0 = unlimited)")
    parser.add_argument("--min-laplacian-var", type=float, default=20.0,
                        help="Per-frame focus measure threshold (drops blurry / texture-only frames)")
    parser.add_argument("--min-entropy", type=float, default=4.0,
                        help="Per-frame grayscale histogram entropy threshold in bits (drops black/dark frames)")
    parser.add_argument("--stride", type=int, default=60,
                        help="Sampling stride in frames. Default 60 = 2s at 30fps. "
                             "Set to 0 to use scene-detection-based sampling instead (slower, fewer frames). "
                             "For drone footage stride sampling is recommended – scene cuts are mostly artificial.")
    parser.add_argument("--night-min-entropy", type=float, default=None,
                        help="Override --min-entropy for night-mission extraction only "
                             "(legit night frames have lower entropy than daytime; default: same as --min-entropy)")
    parser.add_argument("--night-only", action="store_true",
                        help="Skip daytime missions AND val-split. Used to extract test_night/ without "
                             "disturbing an existing train/val split.")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of train to move into val")
    parser.add_argument("--split-mode", choices=["temporal", "random"], default="temporal",
                        help="temporal = per-source tail-15%% (no leakage). random = old buggy behavior.")
    parser.add_argument("--val-buffer", type=int, default=5,
                        help="Number of frames to DELETE between train tail and val head per source. "
                             "Extra safety: even if dedup misses a near-duplicate at the train/val boundary, "
                             "this guarantees they're separated by val-buffer * stride source frames. "
                             "Default 5 stride positions = ~10s separation at stride=60.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)
    cameras = [c.strip() for c in args.cameras.split(",")]

    if args.flat:
        stats = process_flat_folder(
            data_dir=data_dir,
            output_base=output_base,
            size=args.size,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            frames_per_scene=args.frames_per_scene,
            similarity=args.similarity,
            min_sharpness=args.min_sharpness,
            stride=args.stride,
            max_per_prefix=args.max_per_prefix,
            min_laplacian_var=args.min_laplacian_var,
            min_entropy=args.min_entropy,
        )
        _finalize_split(output_base, stats, args.val_split, args.split_mode, args.val_buffer)
        return

    # Find all mission directories
    missions = sorted(data_dir.glob("DJI_*"))
    print(f"Found {len(missions)} missions")

    # Classify missions
    daytime_missions = [m for m in missions if "2224" not in m.name]  # Night mission has 2224
    night_missions = [m for m in missions if "2224" in m.name]

    # Effective entropy for night extraction: override if user passed --night-min-entropy
    night_entropy = args.night_min_entropy if args.night_min_entropy is not None else args.min_entropy

    total_extracted = 0
    stats: dict[str, int] = {}

    # Process daytime missions -> train + val (skipped when --night-only)
    if args.night_only:
        print("\n[--night-only] Skipping daytime missions; only test_night/ will be (re-)extracted.")
        daytime_missions = []
    for mission_dir in daytime_missions:
        mission_name = mission_dir.name
        print(f"\n{'='*60}")
        print(f"Mission: {mission_name} (daytime)")
        print(f"{'='*60}")

        for cam in cameras:
            # Find video file for this camera
            videos = list(mission_dir.glob(f"*_{cam}*.MP4")) + list(mission_dir.glob(f"*_{cam}*.mp4"))
            if not videos:
                print(f"  No {cam}-camera video found, skipping")
                continue

            video_path = str(videos[0])
            prefix = f"{mission_name}_{cam}"

            # Get video info
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = n_frames / fps if fps > 0 else 0
            cap.release()

            print(f"\n  Camera {cam}: {n_frames} frames, {duration:.0f}s, {fps:.0f} fps")

            print(f"  Extracting (stride={args.stride}, lap_var>={args.min_laplacian_var}, "
                  f"entropy>={args.min_entropy}, sim<{args.similarity})...")
            n_extracted = _extract_one_video(
                video_path, output_base / "train", prefix,
                stride=args.stride, size=args.size, threshold=args.threshold,
                min_scene_len=args.min_scene_len, frames_per_scene=args.frames_per_scene,
                similarity=args.similarity, max_per_prefix=args.max_per_prefix,
                min_laplacian_var=args.min_laplacian_var, min_entropy=args.min_entropy,
            )

            print(f"  Extracted {n_extracted} diverse frames -> train/")
            stats[f"{mission_name}_{cam}"] = n_extracted
            total_extracted += n_extracted

    # Process night missions -> test_night (use night_cameras only; Z-zoom at night is useless)
    night_cameras = [c.strip() for c in args.night_cameras.split(",")]
    for mission_dir in night_missions:
        mission_name = mission_dir.name
        print(f"\n{'='*60}")
        print(f"Mission: {mission_name} (NIGHT, cameras={night_cameras})")
        print(f"{'='*60}")

        for cam in night_cameras:
            videos = list(mission_dir.glob(f"*_{cam}*.MP4")) + list(mission_dir.glob(f"*_{cam}*.mp4"))
            if not videos:
                continue

            video_path = str(videos[0])
            prefix = f"{mission_name}_{cam}"

            print(f"  Camera {cam} (stride={args.stride}, entropy>={night_entropy})...")
            n_extracted = _extract_one_video(
                video_path, output_base / "test_night", prefix,
                stride=args.stride, size=args.size, threshold=args.threshold,
                min_scene_len=args.min_scene_len, frames_per_scene=args.frames_per_scene,
                similarity=args.similarity, max_per_prefix=args.max_per_prefix,
                min_laplacian_var=args.min_laplacian_var, min_entropy=night_entropy,
            )
            print(f"  Extracted {n_extracted} frames -> test_night/")
            stats[f"{mission_name}_{cam}_night"] = n_extracted
            total_extracted += n_extracted

    if args.night_only:
        # Don't disturb existing train/val. Just print summary.
        _print_summary(output_base, stats)
    else:
        _finalize_split(output_base, stats, args.val_split, args.split_mode, args.val_buffer)


def _parse_prefix_and_frame(filename: str) -> tuple[str, int]:
    """Parse '{prefix}[_scene{NNNN}]_frame{NNNNNN}.jpg' -> (prefix, frame_number).

    The prefix is the source-video identifier. Used to split by source so
    val frames never share a parent video with train frames.
    """
    import re
    m = re.match(r"(.+?)(?:_scene\d+)?_frame(\d+)\.jpg$", filename)
    if not m:
        return filename, 0
    return m.group(1), int(m.group(2))


def _finalize_split(
    output_base: Path,
    stats: dict[str, int],
    val_fraction: float,
    mode: str = "temporal",
    val_buffer: int = 5,
) -> None:
    """Move a fraction of train/ into val/ and print extraction summary.

    Modes:
        - "temporal": for each source video, hold out the LAST val_fraction of
          extracted frames (sorted by frame number) for val. The ``val_buffer``
          frames immediately before the val tail are DELETED, creating a
          leakage-proof gap between the last train frame and the first val
          frame for every source.
        - "random": old behaviour, kept only for ablation / regression checks.
          Do not use for new datasets.

    Args:
        val_buffer: Number of frames to delete between train and val per source.
            Each "frame" is one stride position, so the actual time gap is
            val_buffer * stride source frames (~val_buffer * stride / fps seconds).
    """
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    if not train_dir.exists():
        _print_summary(output_base, stats)
        return

    val_dir.mkdir(parents=True, exist_ok=True)
    all_train = sorted(train_dir.glob("*.jpg"))
    if not all_train:
        _print_summary(output_base, stats)
        return

    moved = 0
    deleted = 0
    if mode == "random":
        np.random.seed(42)
        n_val = max(1, int(len(all_train) * val_fraction))
        val_indices = np.random.choice(len(all_train), n_val, replace=False)
        for idx in val_indices:
            src = all_train[idx]
            src.rename(val_dir / src.name)
            moved += 1
    else:
        # temporal: group by prefix, sort by frame number, hold out tail with buffer
        by_prefix: dict[str, list[tuple[int, Path]]] = {}
        for p in all_train:
            pref, fnum = _parse_prefix_and_frame(p.name)
            by_prefix.setdefault(pref, []).append((fnum, p))

        for prefix, items in by_prefix.items():
            items.sort(key=lambda t: t[0])
            n = len(items)
            n_val = int(round(n * val_fraction))
            # Need at least val_buffer + 4 frames before we'll split a source.
            if n < val_buffer + 4 or n_val == 0:
                continue
            buffer_start = max(0, n - n_val - val_buffer)
            buffer_end = n - n_val
            # Delete buffer-zone frames (between train tail and val head)
            for _, src in items[buffer_start:buffer_end]:
                src.unlink()
                deleted += 1
            # Move val frames
            for _, src in items[n - n_val:]:
                src.rename(val_dir / src.name)
                moved += 1

    print(f"\nMoved {moved} frames from train -> val ({mode} split, ~{int(val_fraction * 100)}%)")
    if deleted:
        print(f"Deleted {deleted} frames as train/val boundary buffer (val_buffer={val_buffer})")
    _print_summary(output_base, stats)


def _print_summary(output_base: Path, stats: dict[str, int]) -> None:
    """Print extraction summary plus per-prefix split distribution."""
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    train_count = len(list((output_base / "train").glob("*.jpg"))) if (output_base / "train").exists() else 0
    val_count = len(list((output_base / "val").glob("*.jpg"))) if (output_base / "val").exists() else 0
    night_count = len(list((output_base / "test_night").glob("*.jpg"))) if (output_base / "test_night").exists() else 0

    print(f"Train:      {train_count:,} frames")
    print(f"Val:        {val_count:,} frames")
    print(f"Test night: {night_count:,} frames")
    print(f"Total:      {train_count + val_count + night_count:,} frames")
    print(f"\nPer-video extracted (before split):")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v} frames")
    print(f"\nOutput: {output_base}")

    # Verify split integrity: warn if any source video has frames in both train and val
    train_prefixes: set[str] = set()
    val_prefixes: set[str] = set()
    for p in (output_base / "train").glob("*.jpg") if (output_base / "train").exists() else []:
        train_prefixes.add(_parse_prefix_and_frame(p.name)[0])
    for p in (output_base / "val").glob("*.jpg") if (output_base / "val").exists() else []:
        val_prefixes.add(_parse_prefix_and_frame(p.name)[0])
    leaked = train_prefixes & val_prefixes
    if leaked:
        print(f"\n  Note: {len(leaked)} source(s) split temporally between train and val:")
        for p in sorted(leaked):
            print(f"    {p}")
        print(f"  (Tail-end frames are in val; this is expected with mode=temporal.)")

    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    train_count = len(list((output_base / "train").glob("*.jpg"))) if (output_base / "train").exists() else 0
    val_count = len(list((output_base / "val").glob("*.jpg"))) if (output_base / "val").exists() else 0
    night_count = len(list((output_base / "test_night").glob("*.jpg"))) if (output_base / "test_night").exists() else 0

    print(f"Train:      {train_count:,} frames")
    print(f"Val:        {val_count:,} frames")
    print(f"Test night: {night_count:,} frames")
    print(f"Total:      {train_count + val_count + night_count:,} frames")
    print(f"\nPer-video breakdown:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v} frames")
    print(f"\nOutput: {output_base}")


if __name__ == "__main__":
    main()
