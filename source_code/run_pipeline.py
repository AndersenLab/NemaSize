"""
Full NemaSeg Pipeline
=====================
Runs the two-stage pipeline end-to-end:

  Stage 1 – detect_and_crop_rois.py
      YOLO detection model finds worm bounding boxes in raw images,
      crops square ROIs, and writes:
          <ROI_OUTPUT_DIR>/images/          ← cropped ROI images
          <ROI_OUTPUT_DIR>/roi_catalog.json ← geometry metadata

  Stage 2 – skeletonize_worms.py
      YOLO segmentation model segments each worm ROI, extracts the
      centerline skeleton, and writes per-worm length/width measurements
      to a CSV in <SKELETON_OUTPUT_DIR>.

Usage:
    python run_pipeline.py
    (or via SLURM – see SLURM_scripts/test_run.sh)
"""

import argparse
import json
import os
import sys
import subprocess
import time

# Windows + conda: torch and numpy/MKL each ship their own libiomp5md.dll,
# which collide when both are loaded in the same process. Setting this before
# any torch import (and inheriting it into subprocesses) silences the
# "OMP: Error #15" abort. Safe for benchmarking; revisit only if numerical
# correctness on this stack ever becomes suspect.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# YOLO detection model paths
DETECT_MODEL_CLUSTER = r"/scratch/eande106/ZihaoJohnLi/NemaSeg_Project/weights/worm_find/best.pt"
DETECT_MODEL_LOCAL   = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\segment\worm_seg_train\weights\best.pt"

# YOLO segmentation model paths
SEG_MODEL_CLUSTER = r"/scratch/eande106/ZihaoJohnLi/NemaSeg_Project/weights/worm_roi_seg/best.pt"
SEG_MODEL_LOCAL   = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"


# Default project path (used when no CLI argument is provided)
DEFAULT_PROJECT_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Speed_test\WormBodyDetection.v10i.yolo26\train"


# ---------------------------------------------------------------------------
# In-script speed-measurement config (used as argparse defaults).
# Edit these to benchmark without passing CLI flags; CLI flags still override.
# ---------------------------------------------------------------------------
MEASURE_SPEED      = False     # True = enable benchmarking for both stages
SPEED_NUM_BATCHES  = 20        # number of batches (trials) for mean/SD
SPEED_WARMUP       = 5         # warmup predict() calls excluded from samples
SPEED_DEVICE       = "cuda"    # "auto" | "cpu" | "cuda"
SPEED_OUTPUT_DIR   = None      # None = <project>/NemaSize_output/speed_<cpu|gpu>


# ===========================================================================
# PIPELINE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the full NemaSeg detect → skeletonize pipeline."
    )
    parser.add_argument(
        "project_path", nargs="?", default=DEFAULT_PROJECT_PATH,
        help="Root project directory. Sub-folders raw_images/, inference_rois/, "
             "and skeleton/ are derived automatically.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local model paths instead of cluster paths.",
    )
    parser.add_argument(
        "--detect-model", type=str, default=None,
        help="Override path to YOLO detection model (.pt). "
             "Falls back to env var NEMASIZE_DETECT_MODEL, then --local/cluster defaults.",
    )
    parser.add_argument(
        "--seg-model", type=str, default=None,
        help="Override path to YOLO segmentation model (.pt). "
             "Falls back to env var NEMASIZE_SEG_MODEL, then --local/cluster defaults.",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["all", "1", "2", "detect", "skeleton"],
        help="Which stage(s) to run: 'all' (default), '1'/'detect' for ROI "
             "detection only, '2'/'skeleton' for skeletonization only. "
             "Stage 2 alone requires a pre-existing inference_rois/ folder.",
    )
    parser.add_argument(
        "--per-image-timeout-sec", type=float, default=60.0,
        help="Stage 2: if > 0, skip any image whose skeletonization exceeds "
             "this many seconds (POSIX/Linux only, via SIGALRM). "
             "Default 60s. Set to 0 to disable.",
    )
    parser.add_argument(
        "--measure-speed", action="store_true", default=MEASURE_SPEED,
        help="Enable inference-time benchmarking for both stages (opt-in). "
             f"Default {MEASURE_SPEED} (set in-script via MEASURE_SPEED).",
    )
    parser.add_argument(
        "--speed-output-dir", type=str, default=SPEED_OUTPUT_DIR,
        help="Where to write speed_*.csv/json files. "
             "Defaults to <project>/NemaSize_output/speed_<cpu|gpu>.",
    )
    parser.add_argument(
        "--speed-num-batches", type=int, default=SPEED_NUM_BATCHES,
        help=f"Number of batches (trials) for mean/SD. Default {SPEED_NUM_BATCHES}.",
    )
    parser.add_argument(
        "--speed-warmup", type=int, default=SPEED_WARMUP,
        help=f"Warmup predict() calls excluded from samples. Default {SPEED_WARMUP}.",
    )
    parser.add_argument(
        "--speed-device", type=str, default=SPEED_DEVICE,
        choices=["auto", "cpu", "cuda"],
        help="Device for speed measurement: auto|cpu|cuda. "
             "Output dir is auto-suffixed with _cpu or _gpu.",
    )
    args = parser.parse_args()

    run_stage1 = args.stage in ("all", "1", "detect")
    run_stage2 = args.stage in ("all", "2", "skeleton")

    # Precedence: CLI flag > env var > --local/cluster default
    DETECT_MODEL_PATH = (
        args.detect_model
        or os.environ.get("NEMASIZE_DETECT_MODEL")
        or (DETECT_MODEL_LOCAL if args.local else DETECT_MODEL_CLUSTER)
    )
    SEG_MODEL_PATH = (
        args.seg_model
        or os.environ.get("NEMASIZE_SEG_MODEL")
        or (SEG_MODEL_LOCAL if args.local else SEG_MODEL_CLUSTER)
    )

    project_path = args.project_path.rstrip("/\\")

    # Derived paths
    detect_input_folder = os.path.join(project_path, "raw_images")
    output_root         = os.path.join(project_path, "NemaSize_output")
    roi_output_folder   = os.path.join(project_path, "inference_rois")  # intermediate
    skeleton_output_dir = os.path.join(output_root, "skeleton")

    # Validate paths
    if not os.path.isdir(project_path):
        print(f"❌ Project path does not exist: {project_path}")
        sys.exit(1)

    # Stage 1 input mode: raw_images/ folder, or a batch_*.txt manifest when
    # raw_images/ is absent (used by the NemaSize-nf batched workflow, where
    # each batch dir contains exactly one batch_<i>.txt).
    stage1_image_list: str | None = None
    if run_stage1:
        if os.path.isdir(detect_input_folder):
            pass  # folder mode
        else:
            import glob as _glob
            _candidates = sorted(_glob.glob(os.path.join(project_path, "batch_*.txt")))
            if len(_candidates) == 1:
                stage1_image_list = _candidates[0]
                print(f"  raw_images/ not found; using manifest: {stage1_image_list}")
            elif len(_candidates) == 0:
                print(
                    f"❌ Stage 1 input not found: neither {detect_input_folder} "
                    f"nor batch_*.txt in {project_path}"
                )
                sys.exit(1)
            else:
                print(
                    f"❌ Multiple batch_*.txt files in {project_path}; ambiguous:"
                )
                for _c in _candidates:
                    print(f"     {_c}")
                sys.exit(1)

    print("=" * 60)
    print("NemaSeg Full Pipeline")
    print("=" * 60)
    print(f"Project path : {project_path}")
    print(f"Stage(s)     : {args.stage}")

    # Resolve script paths relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve speed output directory (auto-suffix with _cpu / _gpu so CPU and
    # GPU runs don't overwrite each other when invoked back-to-back).
    speed_output_dir: str | None = None
    if args.measure_speed:
        try:
            sys.path.insert(0, script_dir)
            from speed_meter import device_label as _dev_label
            suffix = _dev_label(args.speed_device)
        except Exception:
            suffix = args.speed_device if args.speed_device != "auto" else "auto"
        base = args.speed_output_dir or os.path.join(output_root, "speed")
        speed_output_dir = f"{base}_{suffix}"
        os.makedirs(speed_output_dir, exist_ok=True)
        sys.stderr.write(
            f"[speed] pipeline: device={args.speed_device}, "
            f"output_dir={speed_output_dir}\n"
        )

    stage_times: dict[str, float] = {}
    pipeline_t0 = time.perf_counter()

    # Map speed-device choice to YOLO's device string for both stages.
    # Only forwarded when --measure-speed is set; otherwise stages keep their
    # hardcoded defaults.
    def _yolo_device_arg(choice: str) -> str:
        if choice == "cpu":
            return "cpu"
        if choice == "cuda":
            return "0"
        return ""  # auto
    yolo_device = _yolo_device_arg(args.speed_device) if args.measure_speed else None

    # ------------------------------------------------------------------
    # Stage 1 – Detection & ROI cropping
    # ------------------------------------------------------------------
    if run_stage1:
        print("\n[Stage 1] Running detect_and_crop_rois ...")
        if stage1_image_list:
            print(f"  Input  : {stage1_image_list} (manifest)")
        else:
            print(f"  Input  : {detect_input_folder}")
        print(f"  Output : {roi_output_folder}")
        print(f"  Model  : {DETECT_MODEL_PATH}")

        stage1_cmd = [
            sys.executable, os.path.join(script_dir, "detect_and_crop_rois.py"),
            "--output-folder", roi_output_folder,
            "--model-path",    DETECT_MODEL_PATH,
        ]
        if stage1_image_list:
            stage1_cmd += ["--image-list", stage1_image_list]
        else:
            stage1_cmd += ["--input-folder", detect_input_folder]
        if args.measure_speed:
            stage1_cmd += [
                "--measure-speed",
                "--speed-output-dir", speed_output_dir,
                "--speed-num-batches", str(args.speed_num_batches),
                "--speed-warmup", str(args.speed_warmup),
                "--device", yolo_device,
            ]

        _t = time.perf_counter()
        ret = subprocess.run(stage1_cmd)
        stage_times["stage1_sec"] = time.perf_counter() - _t
        if ret.returncode != 0:
            print(f"\n❌ Stage 1 failed (exit code {ret.returncode}).")
            sys.exit(ret.returncode)

        print(f"\n[Stage 1] Complete.")
    else:
        print("\n[Stage 1] Skipped (use --stage all or --stage 1 to run).")

    roi_images_dir   = os.path.join(roi_output_folder, "images")
    roi_catalog_path = os.path.join(roi_output_folder, "roi_catalog.json")
    print(f"  ROI images : {roi_images_dir}")
    print(f"  Catalog    : {roi_catalog_path}")

    # ------------------------------------------------------------------
    # Stage 2 – Skeletonization
    # ------------------------------------------------------------------
    if run_stage2:
        if not os.path.isdir(roi_images_dir):
            print(f"\n❌ ROI images folder not found: {roi_images_dir}")
            print("    Run Stage 1 first, or check the project path.")
            sys.exit(1)
        if not os.path.isfile(roi_catalog_path):
            print(f"\n❌ ROI catalog not found: {roi_catalog_path}")
            print("    Run Stage 1 first, or check the project path.")
            sys.exit(1)

        print("\n[Stage 2] Running skeletonize_worms ...")
        print(f"  Input  : {roi_images_dir}")
        print(f"  Output : {skeleton_output_dir}")
        print(f"  Model  : {SEG_MODEL_PATH}")

        stage2_cmd = [
            sys.executable, os.path.join(script_dir, "skeletonize_worms.py"),
            "--model-path",       SEG_MODEL_PATH,
            "--image-dir",        roi_images_dir,
            "--output-dir",       skeleton_output_dir,
            "--roi-catalog-path", roi_catalog_path,
        ]
        if args.per_image_timeout_sec and args.per_image_timeout_sec > 0:
            stage2_cmd += [
                "--per-image-timeout-sec", str(args.per_image_timeout_sec)
            ]
        if args.measure_speed:
            stage2_cmd += [
                "--measure-speed",
                "--speed-output-dir", speed_output_dir,
                "--speed-num-batches", str(args.speed_num_batches),
                "--speed-warmup", str(args.speed_warmup),
                "--speed-device", args.speed_device,
                "--device", yolo_device,
            ]

        _t = time.perf_counter()
        ret = subprocess.run(stage2_cmd)
        stage_times["stage2_sec"] = time.perf_counter() - _t
        if ret.returncode != 0:
            print(f"\n❌ Stage 2 failed (exit code {ret.returncode}).")
            sys.exit(ret.returncode)

        print(f"\n[Stage 2] Complete.")
        print(f"  Results saved to: {skeleton_output_dir}")
    else:
        print("\n[Stage 2] Skipped (use --stage all or --stage 2 to run).")

    print("\n" + "=" * 60)
    print("Pipeline finished successfully.")
    print("=" * 60)

    # ---- write end-to-end speed summary (single trial; no mean/SD here) ----
    if args.measure_speed:
        total_wall = time.perf_counter() - pipeline_t0
        pipeline_summary = {
            "device_arg": args.speed_device,
            "stage1_sec": stage_times.get("stage1_sec"),
            "stage2_sec": stage_times.get("stage2_sec"),
            "total_wall_clock_sec": total_wall,
            "project_path": project_path,
            "detect_model": DETECT_MODEL_PATH,
            "seg_model": SEG_MODEL_PATH,
            "speed_num_batches": args.speed_num_batches,
            "speed_warmup": args.speed_warmup,
            "stage": args.stage,
        }
        out_path = os.path.join(speed_output_dir, "speed_pipeline_summary.json")
        with open(out_path, "w") as f:
            json.dump(pipeline_summary, f, indent=2)
        sys.stderr.write(
            f"[speed] pipeline: stage1={pipeline_summary['stage1_sec']}, "
            f"stage2={pipeline_summary['stage2_sec']}, "
            f"total={total_wall:.2f}s\n"
        )
        sys.stderr.write(f"[speed] pipeline: wrote {out_path}\n")


if __name__ == "__main__":
    main()
