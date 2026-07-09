"""
Detect and Crop ROI Pipeline

This script runs a YOLO detection model on a folder of images, then for each
detected bounding box crops a square ROI using the same logic as
create_roi_dataset.py (so downstream processes built on that script continue
to work unchanged).

Pipeline:
  1. Run YOLO detection on every image in the input folder.
  2. For each detected bounding box:
       - ROI center  = bounding-box center (cx, cy)
       - ROI size    = roi_size_scale_factor * max(bbox_width, bbox_height)
       - Enforce minimum roi_size of 32 px
  3. Crop the square ROI (with black-border padding when the ROI extends
     outside the image).
  4. Resize the cropped ROI to output_image_size × output_image_size.
  5. Save the resized image to <output_folder>/images/<stem>_roi_<idx>.jpg
  6. Write a JSON catalog that records, for every ROI, its exact position
     and size in the ORIGINAL image - needed for mapping predictions back.

JSON catalog format (one file per run: <output_folder>/roi_catalog.json):
  {
    "<stem>_roi_<idx>": {
      "source_image":     "<absolute path to original image>",
            "original_width":   <int>,              # original source image width (px)
            "original_height":  <int>,              # original source image height (px)
      "detection_index":  <int>,
      "class_id":         <int>,
      "class_name":       "<str>",
      "confidence":       <float>,
      "bbox_original":    [x1, y1, x2, y2],   # detection bbox in orig. image px
      "roi_top_left":     [x, y],              # top-left of ROI in orig. image px
                                                # (may be negative if padded)
      "roi_size_original": <int>,              # ROI side length in orig. image px
      "output_image_size": <int>,              # side length of the saved ROI image
      "scale_factor":     <float>              # output_image_size / roi_size_original
    },
    ...
  }
"""

import argparse
import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from speed_meter import SpeedMeter, device_label

# ---------------------------------------------------------------------------
# ROI cropping (identical to create_roi_dataset.py)
# ---------------------------------------------------------------------------

def crop_roi(image, centroid, roi_size):
    """
    Crop a square ROI centered on the centroid.

    Identical to the function in create_roi_dataset.py so that downstream
    processes that relied on that script see geometrically consistent ROIs.

    Args:
        image:    Input image (H x W x C numpy array)
        centroid: (x, y) pixel coordinates of the ROI center
        roi_size: Side length of the square ROI in pixels

    Returns:
        roi:         Cropped (and black-padded if needed) square image
        top_left_xy: (x1, y1) top-left corner of the ROI in original-image
                     coordinates (may be negative when padding is applied)
    """
    img_height, img_width = image.shape[:2]
    cx, cy = centroid
    half_size = roi_size / 2

    # ROI boundaries in original-image coordinates
    x1 = int(cx - half_size)
    y1 = int(cy - half_size)
    x2 = int(cx + half_size)
    y2 = int(cy + half_size)

    # Compute padding needed when the ROI extends outside the image
    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - img_width)
    pad_bottom = max(0, y2 - img_height)

    # Clip to valid image region
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(img_width,  x2)
    y2_clip = min(img_height, y2)

    roi = image[y1_clip:y2_clip, x1_clip:x2_clip]

    # Restore the square shape with black borders when clipping was needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return roi, (x1, y1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_and_crop(
    input_folder=None,
    output_folder=None,
    model_path=None,
    roi_size_scale_factor=1.2,
    output_image_size=640,
    conf_threshold=0.25,
    iou_threshold=None,      # None = use ultralytics default (0.7)
    device="",               # "" = auto, "cpu", "0", etc.
    image_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    # ---- speed-measurement (opt-in; default off keeps original behavior) ----
    measure_speed=False,
    speed_output_dir=None,
    speed_num_batches=10,
    speed_warmup=5,
    save_debug_images=False, # Save the image fed to YOLO to <output_folder>/debug_input/
    max_center_dist_ratio=None, # Filter detections whose center is farther than this
                                # fraction of D/2 (image radius) from the image center,
                                # where D is the side length of the square image.
                                # e.g. 0.9 keeps only detections inside a circle of
                                # radius 0.9 * D/2 centered on the image.
                                # None = no filtering.
    image_list_path=None,    # Newline-separated file of absolute image paths.
                             # Mutually exclusive with input_folder.
):
    """
    Run YOLO detection and crop square ROIs for every detected object.

    Args:
        input_folder:         Folder containing input images.
                              Ignored when image_list_path is given.
        output_folder:        Destination folder; images/ sub-folder is created.
        model_path:           Path to the YOLO weights (.pt file).
        roi_size_scale_factor:Scale applied to the longer bbox side to get ROI size.
        output_image_size:    Final ROI image will be resized to this × this (px).
        conf_threshold:       Minimum detection confidence to keep.
        iou_threshold:        NMS IoU threshold.
        device:               Inference device passed to YOLO.
        image_extensions:     Tuple of accepted file extensions.
        image_list_path:      Manifest file of absolute image paths, one per line.
                              When provided, takes precedence over input_folder.
    """
    # ---- imports ----
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        )

    # ---- setup output ----
    images_out = os.path.join(output_folder, "images")
    os.makedirs(images_out, exist_ok=True)

    if save_debug_images:
        debug_out = os.path.join(output_folder, "debug_input")
        os.makedirs(debug_out, exist_ok=True)

    # ---- load model ----
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # ---- gather images ----
    # Two input modes:
    #   manifest:   image_list_path = newline-separated file of absolute paths
    #   folder:     input_folder    = directory scanned via os.listdir
    # Internal representation is a single list of absolute paths so the rest
    # of the function is mode-agnostic.
    if image_list_path is not None:
        with open(image_list_path, "r") as _lf:
            image_paths = [
                ln.strip() for ln in _lf if ln.strip()
            ]
        image_paths = [
            p for p in image_paths
            if os.path.splitext(p)[1].lower() in image_extensions
        ]
        source_label = f"manifest {image_list_path}"
    else:
        if input_folder is None:
            raise ValueError(
                "detect_and_crop: either input_folder or image_list_path is required"
            )
        input_folder = os.path.normpath(input_folder)
        image_paths = sorted([
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        source_label = input_folder

    if len(image_paths) == 0:
        print(f"No images found in {source_label}")
        return {}

    # ---- speed meter (no-op when measure_speed=False) ----
    speed_meter = SpeedMeter(
        name="detect", enabled=measure_speed, device=device or "auto"
    )
    if measure_speed:
        import sys as _sys
        _sys.stderr.write(
            f"[speed] detect: device={speed_meter.device}, "
            f"num_batches={speed_num_batches}, warmup={speed_warmup}, "
            f"images={len(image_paths)}\n"
        )
        # Warmup: run predict() on the first W images (no postprocessing kept).
        w_count = min(int(speed_warmup), len(image_paths))
        speed_meter.warmup_start()
        for _wpath in image_paths[:w_count]:
            _wimg = cv2.imread(_wpath, cv2.IMREAD_UNCHANGED)
            if _wimg is None:
                continue
            if _wimg.dtype != np.uint8:
                _wimg = cv2.normalize(_wimg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if _wimg.ndim == 2:
                _wimg = cv2.cvtColor(_wimg, cv2.COLOR_GRAY2BGR)
            _wkw = dict(source=_wimg, conf=conf_threshold, device=device, verbose=False)
            if iou_threshold is not None:
                _wkw["iou"] = iou_threshold
            model.predict(**_wkw)
        speed_meter.warmup_end(w_count)

    print(f"\nFound {len(image_paths)} image(s) in {source_label}")
    print(f"ROI scale factor : {roi_size_scale_factor}")
    print(f"Output image size: {output_image_size}x{output_image_size}")
    print(f"Confidence thresh: {conf_threshold}")
    if max_center_dist_ratio is not None:
        print(f"Max center dist  : {max_center_dist_ratio} x D/2 (image radius)")
    print()

    # ---- catalog dict ----
    catalog = {}   # roi_key -> metadata dict

    total_rois = 0

    for img_path in tqdm(image_paths, desc="Detecting & cropping", mininterval=10):
        img_filename = os.path.basename(img_path)
        # IMREAD_UNCHANGED preserves the original bit depth (e.g. 16-bit TIFs)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            continue

        # Normalize non-8-bit images (e.g. 16-bit microscopy TIFs) to 8-bit for YOLO.
        if image.dtype != np.uint8:
            #print(f"  Normalizing {img_path} from {image.dtype} to uint8.")
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Ensure image is 3-channel BGR (IMREAD_UNCHANGED may return grayscale)
        if image.ndim == 2:
            #print(f"  Converting grayscale {img_path} to BGR.")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_height, img_width = image.shape[:2]
        stem = os.path.splitext(img_filename)[0]

        # ---- save debug image (what YOLO will see) ----
        if save_debug_images:
            debug_path = os.path.join(debug_out, f"{stem}_yolo_input.png")
            cv2.imwrite(debug_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # ---- run YOLO detection ----
        predict_kwargs = dict(
            source=image,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        if iou_threshold is not None:
            predict_kwargs["iou"] = iou_threshold
        speed_meter.tick_start()
        results = model.predict(**predict_kwargs)
        speed_meter.tick_end()

        # results is a list (one element per image)
        result = results[0]
        boxes = result.boxes  # ultralytics Boxes object

        if boxes is None or len(boxes) == 0:
            # No detections in this image
            continue

        # ---- process each detection ----
        for det_idx, box in enumerate(boxes):
            # Bounding box in original image pixels (xyxy format)
            x1_bb, y1_bb, x2_bb, y2_bb = box.xyxy[0].tolist()
            bbox_w = x2_bb - x1_bb
            bbox_h = y2_bb - y1_bb

            # ROI center = bounding-box center
            cx = (x1_bb + x2_bb) / 2.0
            cy = (y1_bb + y2_bb) / 2.0

            # ---- distance-from-center filter ----
            if max_center_dist_ratio is not None:
                img_cx = img_width  / 2.0
                img_cy = img_height / 2.0
                image_radius = img_width / 2.0  # D/2 for a square image
                dist = ((cx - img_cx) ** 2 + (cy - img_cy) ** 2) ** 0.5
                if dist / image_radius > max_center_dist_ratio:
                    continue  # skip — outside the valid circular area

            # ROI size = scale_factor * longer side of bbox
            roi_size = int(max(bbox_w, bbox_h) * roi_size_scale_factor)
            roi_size = max(roi_size, 32)   # enforce minimum

            # Confidence and class
            confidence = float(box.conf[0])
            class_id   = int(box.cls[0])
            class_name = model.names.get(class_id, str(class_id))

            # ---- crop ROI (identical logic to create_roi_dataset.py) ----
            roi_image, roi_top_left = crop_roi(image, (cx, cy), roi_size)

            # ---- resize to target output size ----
            # Use INTER_AREA when shrinking (best for downscaling),
            # INTER_LANCZOS4 when enlarging (highest quality upscaling).
            interp = cv2.INTER_AREA if roi_size > output_image_size else cv2.INTER_LANCZOS4
            roi_resized = cv2.resize(
                roi_image,
                (output_image_size, output_image_size),
                interpolation=interp,
            )

            # ---- save image (lossless PNG, compression=0 for fastest I/O) ----
            roi_key          = f"{stem}_roi_{det_idx}"
            roi_img_filename = f"{roi_key}.png"
            roi_img_path     = os.path.join(images_out, roi_img_filename)
            cv2.imwrite(roi_img_path, roi_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # ---- catalog entry ----
            catalog[roi_key] = {
                "source_image":      img_path,
                "original_width":    img_width,
                "original_height":   img_height,
                "detection_index":   det_idx,
                "class_id":          class_id,
                "class_name":        class_name,
                "confidence":        round(confidence, 6),
                # Bounding box from YOLO (original image px)
                "bbox_original":     [
                    round(x1_bb, 2), round(y1_bb, 2),
                    round(x2_bb, 2), round(y2_bb, 2),
                ],
                # ROI geometry in original image coordinates
                "roi_center_original": [round(cx, 2), round(cy, 2)],
                "roi_top_left":      [roi_top_left[0], roi_top_left[1]],
                "roi_size_original": roi_size,
                # Output metadata
                "output_image_size": output_image_size,
                "scale_factor":      round(output_image_size / roi_size, 6),
                "output_image_path": roi_img_path,
            }

            total_rois += 1

    # ---- save catalog ----
    catalog_path = os.path.join(output_folder, "roi_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nDone. {total_rois} ROI image(s) saved to: {images_out}")
    print(f"Catalog written to: {catalog_path}")

    # ---- finalize speed meter (writes nothing when disabled) ----
    if measure_speed:
        _so_dir = speed_output_dir or os.path.join(output_folder, "speed")
        speed_meter.finalize(
            output_dir=_so_dir,
            num_batches=int(speed_num_batches),
            extra_metadata={
                "model_path": model_path,
                "input_source": source_label,
                "image_count": len(image_paths),
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "device_arg": device or "auto",
                "device_label": device_label(device or "auto"),
            },
        )

    return catalog


# ---------------------------------------------------------------------------
# Configuration & entry point
# ---------------------------------------------------------------------------

def main():
    # ========== CONFIGURATION ==========

    # Folder of input images (any mix of jpg/png/etc.)
    #input_folder = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyDetection.v10i.coco-segmentation\valid\images"
    input_folder = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20240606-RIL-HTA1-AOS/raw_images"
    #input_folder = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\WormBodyDetection.v10i.yolo26\train\images"

    # Where to save ROI images and the catalog JSON
    #output_folder = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\inference_rois"
    output_folder = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20240606-RIL-HTA1-AOS/inference_rois"
    #output_folder = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\inference_rois"

    # YOLO detection model weights
    #model_path = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\segment\worm_seg_train\weights\best.pt"
    model_path = r"/scratch/eande106/ZihaoJohnLi/NemaSeg_Project/weights/worm_find/best.pt"

    # Manifest mode (--image-list): newline-separated absolute paths.
    # Takes precedence over input_folder when set via CLI.
    image_list_path = None

    # Apply CLI path overrides (if provided)
    if __name__ == "__main__" and _cli_override:
        input_folder    = _cli_args.input_folder
        output_folder   = _cli_args.output_folder
        model_path      = _cli_args.model_path
        image_list_path = getattr(_cli_args, "image_list", None)

    # ROI size = roi_size_scale_factor * longer side of bounding box
    # 1.2 = 20% padding around the longest bbox side (same default as create_roi_dataset.py)
    roi_size_scale_factor = 1.3

    # All ROI images are resized to this square dimension before saving
    output_image_size = 640   # pixels

    # YOLO detection thresholds
    conf_threshold = 0.25
    iou_threshold  = None  # None = use ultralytics default (0.7); set a float to override

    # Inference device: "" = ultralytics auto-select (GPU if available, else CPU),
    # "cpu" = force CPU, "0" = first GPU, "0,1" = multi-GPU, etc.
    # Default "" makes the script work in CPU-only environments (e.g. Docker
    # CPU image) without crashing on `device=0`.
    device = ""

    # CLI device override (after the local default so --device wins).
    if __name__ == "__main__" and getattr(_cli_args, "device", None) is not None:
        device = _cli_args.device

    # Save the normalized image fed to YOLO for debugging (set False to disable)
    save_debug_images = False

    # Filter detections outside a circular valid area centered on the image.
    # The radius threshold is expressed as a fraction of D/2 (half the image side).
    # e.g. 0.9 keeps detections within a circle of radius 0.9 * D/2. (0.97 is the absolute maximum of well edge)
    # None = disabled.
    max_center_dist_ratio = 0.94

    # ===================================

    print("=" * 60)
    print("Detect-and-Crop ROI Pipeline")
    print("=" * 60)
    print(f"Input folder     : {input_folder}")
    print(f"Output folder    : {output_folder}")
    print(f"Model            : {model_path}")
    print(f"Scale factor     : {roi_size_scale_factor}")
    print(f"Output image size: {output_image_size}x{output_image_size}")
    print(f"Conf threshold   : {conf_threshold}")
    print(f"IoU threshold    : {iou_threshold}")
    print("=" * 60)

    detect_and_crop(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        roi_size_scale_factor=roi_size_scale_factor,
        output_image_size=output_image_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device,
        save_debug_images=save_debug_images,
        max_center_dist_ratio=max_center_dist_ratio,
        image_list_path=image_list_path,
        measure_speed=bool(globals().get("_cli_args") and getattr(_cli_args, "measure_speed", False)),
        speed_output_dir=(getattr(_cli_args, "speed_output_dir", None) if globals().get("_cli_args") else None),
        speed_num_batches=(getattr(_cli_args, "speed_num_batches", 10) if globals().get("_cli_args") else 10),
        speed_warmup=(getattr(_cli_args, "speed_warmup", 5) if globals().get("_cli_args") else 5),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect worms and crop square ROIs from raw images."
    )
    parser.add_argument("--input-folder", type=str, default=None,
                        help="Folder of input images.")
    parser.add_argument("--image-list", type=str, default=None,
                        help="Newline-separated file of absolute image paths. "
                             "Mutually exclusive with --input-folder; if both "
                             "are given, --image-list wins.")
    parser.add_argument("--output-folder", type=str, default=None,
                        help="Destination folder for ROI images and catalog.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to YOLO detection weights (.pt).")
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device for YOLO: '' (auto), 'cpu', '0', etc.")
    parser.add_argument("--measure-speed", action="store_true",
                        help="Enable inference-time benchmarking (opt-in). "
                             "Default off — original behavior unchanged.")
    parser.add_argument("--speed-output-dir", type=str, default=None,
                        help="Where to write speed_detect_*.csv/json. "
                             "Defaults to <output_folder>/speed.")
    parser.add_argument("--speed-num-batches", type=int, default=10,
                        help="Number of batches (trials) for mean/SD. Default 10.")
    parser.add_argument("--speed-warmup", type=int, default=5,
                        help="Warmup predict() calls excluded from samples. Default 5.")

    _cli_args = parser.parse_args()
    _cli_override = (
        (_cli_args.input_folder or _cli_args.image_list)
        and _cli_args.output_folder
        and _cli_args.model_path
    )

    main()
