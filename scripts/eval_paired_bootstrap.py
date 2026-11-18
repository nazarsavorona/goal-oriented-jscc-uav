"""Paired bootstrap + Wilcoxon significance for two eval runs.

Loads per-image ``cls_sim`` (or another column) from two evaluation
directories, aligns the rows by frame identifier, and reports the mean
difference ``A - B`` with a 95 percent paired-bootstrap confidence
interval, a parametric paired-t CI cross-check, and a Wilcoxon
signed-rank p-value. The verdict ``distinguishable`` is True iff the
bootstrap CI excludes 0 AND the Wilcoxon p-value is below ``alpha``.

The loader auto-detects three ``per_frame.csv`` schemas produced over the
project history:

1. ``batch,i_in_batch,cls_sim,psnr_db`` (single channel/SNR per file).
   Frame id = ``"{batch}_{i_in_batch}"``.
2. ``label,channel,snr,image,psnr,cls_sim,patch_sim`` (planA / benchmark).
   Frame id = ``image``; rows filtered by ``--channel`` and ``--snr``.
3. ``label,bench,jammer,jnr,snr,image,cls_sim,patch_sim,psnr`` (reactive
   jamming sweeps). Frame id = ``image``; rows filtered by ``--label``,
   ``--bench``, ``--jammer``, ``--jnr`` and ``--snr``.

Each ``--dir`` argument may be a directory (``per_frame.csv`` is appended)
or a direct path to a CSV file.

Example:
    python scripts/eval_paired_bootstrap.py \\
        --dir-a experiments/planA_p49zs_dji --label-a P49 \\
        --dir-b experiments/planA_swin512v40_dji --label-b SwinJSCC-512 \\
        --channel awgn --snr 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semvideo.eval.stats import compare_paired  # noqa: E402

# Mapping of float-valued filter columns -> CLI arg names, for tolerant
# numeric matching (e.g. "10" matches "10.0").
_FLOAT_FILTERS = {"snr", "jnr"}


def _read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV into a header list and a list of row dicts.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple ``(fieldnames, rows)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no header.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"no such CSV: {csv_path}")
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"empty / header-less CSV: {csv_path}")
        return list(reader.fieldnames), list(reader)


def _frame_id(row: dict[str, str], schema: str) -> str:
    """Compute the per-image alignment key for a row.

    Args:
        row: A CSV row mapping column name to string value.
        schema: One of ``"old"``, ``"image"``.

    Returns:
        A stable frame identifier string.
    """
    if schema == "old":
        return f"{row['batch']}_{row['i_in_batch']}"
    return row["image"]


def _matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    """Test whether a row satisfies all active filters.

    Numeric filters (``snr``/``jnr``) match by float value so that
    ``"10"`` and ``"10.0"`` are treated as equal.

    Args:
        row: A CSV row mapping column name to string value.
        filters: Mapping of column name to required value (string).

    Returns:
        True iff every active filter matches.
    """
    for col, want in filters.items():
        if want is None or col not in row:
            continue
        have = row[col]
        if col in _FLOAT_FILTERS:
            if float(have) != float(want):
                return False
        elif have != want:
            return False
    return True


def load_metric_by_frame(
    path: Path,
    metric: str,
    filters: dict[str, str],
) -> dict[str, float]:
    """Load a per-image metric keyed by frame id, applying row filters.

    Args:
        path: Directory containing ``per_frame.csv`` or a direct CSV path.
        metric: Column name to extract (e.g. ``"cls_sim"``).
        filters: Active channel/snr/label/jnr/etc. filters.

    Returns:
        Mapping from frame id to metric value. The last matching row wins
        if a frame id is duplicated (none expected in practice).

    Raises:
        ValueError: If the schema is unrecognised, the metric column is
            absent, or no rows survive filtering.
    """
    csv_path = path / "per_frame.csv" if path.is_dir() else path
    fieldnames, rows = _read_rows(csv_path)
    fields = set(fieldnames)

    if {"batch", "i_in_batch"} <= fields:
        schema = "old"
    elif "image" in fields:
        schema = "image"
    else:
        raise ValueError(
            f"unrecognised per_frame schema in {csv_path}: columns={fieldnames}"
        )

    if metric not in fields:
        raise ValueError(f"metric column '{metric}' not in {csv_path} (have {fieldnames})")

    # Only apply filters that are actually columns of this file.
    active = {k: v for k, v in filters.items() if v is not None and k in fields}

    out: dict[str, float] = {}
    for row in rows:
        if not _matches(row, active):
            continue
        out[_frame_id(row, schema)] = float(row[metric])

    if not out:
        raise ValueError(
            f"no rows in {csv_path} match filters {active} (schema={schema})"
        )
    return out


def align(
    a: dict[str, float], b: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect two frame-keyed metric dicts into aligned arrays.

    Args:
        a: Frame id -> metric for model A.
        b: Frame id -> metric for model B.

    Returns:
        Tuple ``(arr_a, arr_b, keys)`` where ``keys`` is the sorted list of
        common frame ids and the arrays are aligned to it.

    Raises:
        ValueError: If the frame-id intersection is empty.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError(
            "no common frame ids between the two runs - check schema/filters; "
            f"A has {len(a)} frames, B has {len(b)} frames"
        )
    arr_a = np.array([a[k] for k in keys], dtype=np.float64)
    arr_b = np.array([b[k] for k in keys], dtype=np.float64)
    return arr_a, arr_b, keys


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir-a", required=True, type=Path, help="eval dir or CSV for model A")
    p.add_argument("--dir-b", required=True, type=Path, help="eval dir or CSV for model B")
    p.add_argument("--label-a", default="A", help="display label for model A")
    p.add_argument("--label-b", default="B", help="display label for model B")
    p.add_argument("--metric", default="cls_sim", help="per-image metric column (default cls_sim)")
    # Filters (applied only when the column is present in a given file).
    p.add_argument("--channel", default=None, help="filter rows by channel (image schema)")
    p.add_argument("--snr", default=None, help="filter rows by SNR dB")
    p.add_argument("--jammer", default=None, help="filter rows by jammer name (reactive schema)")
    p.add_argument("--jnr", default=None, help="filter rows by JNR dB (reactive schema)")
    p.add_argument("--row-label-a", default=None, help="filter A rows by label column value")
    p.add_argument("--row-label-b", default=None, help="filter B rows by label column value")
    p.add_argument("--bench", default=None, help="filter rows by bench column (reactive schema)")
    p.add_argument("--n-boot", type=int, default=10_000, help="bootstrap resamples")
    p.add_argument("--conf-level", type=float, default=0.95, help="confidence level")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)

    common = {
        "channel": args.channel,
        "snr": args.snr,
        "jammer": args.jammer,
        "jnr": args.jnr,
        "bench": args.bench,
    }
    filters_a = dict(common, label=args.row_label_a)
    filters_b = dict(common, label=args.row_label_b)

    a = load_metric_by_frame(args.dir_a, args.metric, filters_a)
    b = load_metric_by_frame(args.dir_b, args.metric, filters_b)
    arr_a, arr_b, keys = align(a, b)

    res = compare_paired(
        arr_a, arr_b, n_boot=args.n_boot, conf_level=args.conf_level, seed=args.seed
    )
    pct = int(round(args.conf_level * 100))
    print(f"A = {args.label_a}   B = {args.label_b}   metric = {args.metric}")
    print(f"paired frames: {res.n}")
    print(f"mean A      : {res.mean_a:.5f}")
    print(f"mean B      : {res.mean_b:.5f}")
    print(f"mean delta (A - B): {res.mean_diff:+.5f}")
    print(f"{pct}% bootstrap CI : [{res.ci_low:+.5f}, {res.ci_high:+.5f}]  (n_boot={args.n_boot})")
    print(f"{pct}% paired-t  CI : [{res.ci_low_t:+.5f}, {res.ci_high_t:+.5f}]")
    print(f"Wilcoxon W  : {res.wilcoxon_stat:.1f}")
    print(f"Wilcoxon p  : {res.wilcoxon_p:.3e}")
    print(f"distinguishable @ alpha={1 - args.conf_level:g}: {res.distinguishable}")
    # Machine-readable one-liner (pipe-delimited) for table assembly.
    print(
        "TSV\t"
        f"{args.label_a}\t{args.label_b}\t{res.n}\t{res.mean_a:.5f}\t{res.mean_b:.5f}\t"
        f"{res.mean_diff:+.5f}\t{res.ci_low:+.5f}\t{res.ci_high:+.5f}\t{res.wilcoxon_p:.3e}\t"
        f"{int(res.distinguishable)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
