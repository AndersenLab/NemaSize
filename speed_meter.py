"""
Lightweight speed-measurement helper for paper-publication benchmarking.

Records per-call wall-clock time of YOLO ``model.predict()`` invocations, then
partitions the recorded samples into K contiguous batches.  The mean and
standard deviation are computed across those K batch samples (one batch = one
trial), which is what the NemaSeg paper reports for inference timing.

Designed to be opt-in: when ``enabled=False`` the helper short-circuits in
~one branch per call, so leaving it disabled has negligible overhead and does
not change any existing stdout/stderr output.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from typing import Any


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433 (local import keeps cold-path light)
        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_sync() -> None:
    import torch
    torch.cuda.synchronize()


def _gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def resolve_device(device_arg: str) -> str:
    """Map ``"auto"`` / ``""`` to ``"cuda"`` or ``"cpu"`` based on availability."""
    if device_arg in ("auto", "", None):
        return "cuda" if _cuda_available() else "cpu"
    if device_arg == "cuda":
        return "cuda" if _cuda_available() else "cpu"
    return device_arg  # "cpu", "0", "0,1", etc.


def device_label(device_arg: str) -> str:
    """Short label suitable for filename suffix: ``cpu`` or ``gpu``."""
    resolved = resolve_device(device_arg)
    return "gpu" if resolved.startswith("cuda") or resolved.isdigit() else "cpu"


class SpeedMeter:
    """Records per-call timings and writes batched mean/SD summaries."""

    def __init__(self, name: str, enabled: bool, device: str = "auto") -> None:
        self.name = name
        self.enabled = enabled
        self.device = resolve_device(device)
        self._is_cuda = self.device.startswith("cuda") or self.device.isdigit()
        self._samples_sec: list[float] = []
        self._warmup_sec: float = 0.0
        self._warmup_items: int = 0
        self._t0: float | None = None
        self._tick_t0: float | None = None

    # -- warmup --------------------------------------------------------
    def warmup_start(self) -> None:
        if not self.enabled:
            return
        if self._is_cuda:
            _cuda_sync()
        self._t0 = time.perf_counter()

    def warmup_end(self, n_items: int) -> None:
        if not self.enabled or self._t0 is None:
            return
        if self._is_cuda:
            _cuda_sync()
        self._warmup_sec += time.perf_counter() - self._t0
        self._warmup_items += n_items
        self._t0 = None

    # -- per-call timing ----------------------------------------------
    def tick_start(self) -> None:
        if not self.enabled:
            return
        if self._is_cuda:
            _cuda_sync()
        self._tick_t0 = time.perf_counter()

    def tick_end(self) -> None:
        if not self.enabled or self._tick_t0 is None:
            return
        if self._is_cuda:
            _cuda_sync()
        self._samples_sec.append(time.perf_counter() - self._tick_t0)
        self._tick_t0 = None

    # -- summary -------------------------------------------------------
    def finalize(
        self,
        output_dir: str,
        num_batches: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        os.makedirs(output_dir, exist_ok=True)
        n = len(self._samples_sec)

        if n == 0:
            sys.stderr.write(
                f"[speed] {self.name}: no samples recorded; nothing to summarize.\n"
            )
            summary = {
                "name": self.name,
                "n_samples": 0,
                "n_batches": 0,
                "device": self.device,
                "gpu_name": _gpu_name(),
                "warmup_sec": self._warmup_sec,
                "warmup_items": self._warmup_items,
            }
            if extra_metadata:
                summary.update(extra_metadata)
            with open(os.path.join(output_dir, f"speed_{self.name}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            return summary

        # Partition samples into K contiguous batches.
        k = max(1, min(num_batches, n))
        if k != num_batches:
            sys.stderr.write(
                f"[speed] {self.name}: only {n} samples; reducing batches "
                f"from {num_batches} to {k}.\n"
            )

        # Distribute samples evenly: the first (n % k) batches get one extra
        # item so batch sizes differ by at most 1. This avoids inflating the
        # last batch when n is not divisible by k (which would otherwise bias
        # mean/SD upward for that final trial).
        base, rem = divmod(n, k)
        batch_seconds: list[float] = []
        batch_counts: list[int] = []
        lo = 0
        for i in range(k):
            size = base + (1 if i < rem else 0)
            hi = lo + size
            batch_seconds.append(sum(self._samples_sec[lo:hi]))
            batch_counts.append(size)
            lo = hi

        # Write raw per-batch CSV.
        raw_path = os.path.join(output_dir, f"speed_{self.name}_raw.csv")
        with open(raw_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch_idx", "n_items", "seconds"])
            for i, (sec, cnt) in enumerate(zip(batch_seconds, batch_counts)):
                w.writerow([i, cnt, f"{sec:.6f}"])

        mean_sec = statistics.fmean(batch_seconds)
        std_sec = statistics.stdev(batch_seconds) if len(batch_seconds) > 1 else 0.0
        total_sec = sum(batch_seconds)
        mean_items = statistics.fmean(batch_counts)

        summary: dict[str, Any] = {
            "name": self.name,
            "device": self.device,
            "gpu_name": _gpu_name(),
            "n_samples": n,
            "n_batches": k,
            "batch_size_mean_items": mean_items,
            "batch_seconds_mean": mean_sec,
            "batch_seconds_std": std_sec,
            "batch_seconds_min": min(batch_seconds),
            "batch_seconds_max": max(batch_seconds),
            "batch_seconds_median": statistics.median(batch_seconds),
            "total_seconds": total_sec,
            "throughput_items_per_sec": n / total_sec if total_sec > 0 else None,
            "per_item_seconds_mean": total_sec / n,
            "warmup_sec": self._warmup_sec,
            "warmup_items": self._warmup_items,
        }
        if extra_metadata:
            summary.update(extra_metadata)

        json_path = os.path.join(output_dir, f"speed_{self.name}_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        sys.stderr.write(
            f"[speed] {self.name}: {k} batches × ~{mean_items:.1f} items, "
            f"mean = {mean_sec*1000:.2f} ms, std = {std_sec*1000:.2f} ms, "
            f"per-item = {summary['per_item_seconds_mean']*1000:.2f} ms "
            f"(device={self.device})\n"
        )
        sys.stderr.write(f"[speed] {self.name}: wrote {raw_path} and {json_path}\n")
        return summary
