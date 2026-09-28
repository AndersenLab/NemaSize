"""
Parses a CellProfiler headless-run log (produced by
build_cellprofiler_speed_test.py + `docker run ... cellprofiler -c -r ...`)
and reports per-image wall-clock timing in the same
mean/SD-over-batches format as NemaSize's source_code/speed_meter.py, so the
two tools' speeds can be reported side by side in the paper.

NemaSize's speed_meter only times the model.predict() call itself (image is
already loaded into memory before tick_start(), and nothing is saved to disk
until after tick_end() -- see detect_and_crop_rois.py / skeletonize_worms.py).
To match that "processing only" scope, this script excludes CellProfiler's
LoadData module (reads the image + metadata off disk) and the
RescaleIntensity/OverlayOutlines/SaveImages modules (build + write the QC
overlay PNG) from the "processing" total by default; use --include-io to get
the full per-image time instead.

Usage:
    python parse_cellprofiler_speed_log.py <path/to/run_timing.log> [--batches K]
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

START_END_RE = re.compile(r"Start:\s*(\S+).*?End:\s*(\S+)", re.DOTALL)

# \s+ (not a literal space) because PowerShell's `*>>` redirection word-wraps
# stdout at console width, splitting a single log record across two lines.
LINE_RE = re.compile(
    r"Image\s*#\s*(\d+),\s*module\s+(\S+)\s*#\s*\d+:\s*CPU_time\s*=\s*([\d.]+)\s*secs,\s*"
    r"Wall_time\s*=\s*([\d.]+)\s*secs"
)

# Disk I/O and QC-visualization modules -- not analogous to anything inside
# NemaSize's timed model.predict() window, so excluded from "processing only".
IO_AND_VIZ_MODULES = {"LoadData", "RescaleIntensity", "OverlayOutlines", "SaveImages"}


def parse_per_image_per_module(log_path: Path) -> dict[int, dict[str, float]]:
    """Return {image_idx: {module_name: wall_seconds}} (module times summed if repeated)."""
    # PowerShell's Out-File/redirection writes UTF-16LE (with BOM); fall back to utf-8.
    raw = log_path.read_bytes()
    try:
        text = raw.decode("utf-16")
    except UnicodeError:
        text = raw.decode("utf-8", errors="ignore")

    per_image: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for m in LINE_RE.finditer(text):
        img_idx = int(m.group(1))
        module = m.group(2)
        wall = float(m.group(4))
        per_image[img_idx][module] += wall
    return per_image


def parse_total_wall_clock_sec(log_path: Path) -> float | None:
    """Read the Start:/End: bracket around the whole `docker run` call.

    This mirrors NemaSize's total_wall_clock_sec (subprocess-to-subprocess,
    includes engine/process startup, all image I/O, and final CSV writes --
    everything the per-module sums in parse_per_image_per_module() miss).
    """
    raw = log_path.read_bytes()
    try:
        text = raw.decode("utf-16")
    except UnicodeError:
        text = raw.decode("utf-8", errors="ignore")

    m = START_END_RE.search(text)
    if not m:
        return None
    start = datetime.fromisoformat(re.sub(r"(\.\d{6})\d*", r"\1", m.group(1)))
    end = datetime.fromisoformat(re.sub(r"(\.\d{6})\d*", r"\1", m.group(2)))
    return (end - start).total_seconds()


def summarize(samples: list[float], num_batches: int, name: str) -> dict:
    n = len(samples)
    k = max(1, min(num_batches, n))
    base, rem = divmod(n, k)
    batch_seconds = []
    lo = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        hi = lo + size
        batch_seconds.append(sum(samples[lo:hi]))
        lo = hi

    mean_sec = statistics.fmean(batch_seconds)
    std_sec = statistics.stdev(batch_seconds) if len(batch_seconds) > 1 else 0.0
    total_sec = sum(batch_seconds)

    return {
        "name": name,
        "n_samples": n,
        "n_batches": k,
        "batch_seconds_mean": mean_sec,
        "batch_seconds_std": std_sec,
        "batch_seconds_min": min(batch_seconds),
        "batch_seconds_max": max(batch_seconds),
        "batch_seconds_median": statistics.median(batch_seconds),
        "total_seconds": total_sec,
        "throughput_items_per_sec": n / total_sec if total_sec > 0 else None,
        "per_item_seconds_mean": total_sec / n if n else None,
        "per_item_seconds_min": min(samples),
        "per_item_seconds_max": max(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path)
    ap.add_argument("--batches", type=int, default=1,
                     help="Number of contiguous batches for mean/SD (default 1 = "
                          "single trial, matching a per-image sample list).")
    ap.add_argument("--warmup", type=int, default=1,
                     help="Exclude the first N images from the summary as warmup "
                          "(mirrors speed_meter.py's SPEED_WARMUP). Default 1, "
                          "since image #1 shows a one-time first-call cost.")
    ap.add_argument("--include-io", action="store_true",
                     help="Include LoadData/RescaleIntensity/OverlayOutlines/"
                          "SaveImages in the reported total (default: excluded, "
                          "to match NemaSize's model.predict()-only timing).")
    ap.add_argument("--out", type=Path, default=None,
                     help="Where to write speed_CellProfiler_summary.json "
                          "(default: alongside the log file).")
    args = ap.parse_args()

    per_image_modules = parse_per_image_per_module(args.log_path)
    all_indices = sorted(per_image_modules)
    print(f"Parsed {len(all_indices)} images from {args.log_path}")

    # Per-module breakdown (mean across all parsed images, incl. warmup).
    module_totals: dict[str, float] = defaultdict(float)
    for modules in per_image_modules.values():
        for mod, sec in modules.items():
            module_totals[mod] += sec
    print("\nPer-module mean seconds/image (across all parsed images):")
    for mod, total in sorted(module_totals.items(), key=lambda kv: -kv[1]):
        tag = " [I/O or QC overlay -- excluded from processing-only]" if mod in IO_AND_VIZ_MODULES else ""
        print(f"  {mod:28s} {total / len(all_indices):8.2f} s{tag}")

    full_per_image = {i: sum(m.values()) for i, m in per_image_modules.items()}
    processing_per_image = {
        i: sum(sec for mod, sec in m.items() if mod not in IO_AND_VIZ_MODULES)
        for i, m in per_image_modules.items()
    }
    per_image = full_per_image if args.include_io else processing_per_image
    label = "CellProfiler_full" if args.include_io else "CellProfiler_processing_only"

    print(f"\nUsing {'FULL (incl. I/O + overlay)' if args.include_io else 'PROCESSING-ONLY (excl. I/O + overlay)'} per-image time:")
    for i in all_indices:
        print(f"  image #{i}: {per_image[i]:.2f} s")

    warmup_indices = all_indices[: args.warmup]
    kept_indices = all_indices[args.warmup:]
    samples = [per_image[i] for i in kept_indices]

    summary = summarize(samples, args.batches, name=label)
    summary["warmup_images"] = warmup_indices
    summary["warmup_seconds"] = sum(per_image[i] for i in warmup_indices)
    summary["excluded_modules"] = [] if args.include_io else sorted(IO_AND_VIZ_MODULES)
    print(json.dumps(summary, indent=2))

    total_wall_clock_sec = parse_total_wall_clock_sec(args.log_path)
    if total_wall_clock_sec is not None:
        n_all = len(all_indices)
        summary["total_wall_clock_sec"] = total_wall_clock_sec
        summary["total_wall_clock_per_image_sec"] = total_wall_clock_sec / n_all if n_all else None
        print(
            f"\nTrue full-pipeline wall clock (Start:/End: bracket, mirrors "
            f"NemaSize's total_wall_clock_sec): {total_wall_clock_sec:.2f} s "
            f"for {n_all} images = {total_wall_clock_sec / n_all:.2f} s/image"
        )
    else:
        print("\nNo Start:/End: bracket found in log -- total_wall_clock_sec unavailable.")

    out_path = args.out or args.log_path.parent / f"speed_{label}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
