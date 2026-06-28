# NemaSize / NemaSeg

Automated *C. elegans* body-length and width measurement from microscope
images.

Written by **Zihao John Li** (Andersen Lab, Johns Hopkins University).

---

## What this repo contains

A two-stage YOLO pipeline that turns raw well-plate images into per-worm
length and width measurements:

```
raw images ──► [Stage 1] YOLO detection  ──► per-worm ROI crops
                                              │
                                              ▼
                            [Stage 2] YOLO-Seg segmentation
                                              │
                                              ▼
                            centerline skeletonization
                                              │
                                              ▼
                                  worm_lengths.csv (µm)
                                  contour + skeleton (.txt)
```

### End users

If you just want to **run the pipeline on your own images**, use the
Docker image — you do not need Python, CUDA, or any of the source code:

> 📘 **Beta-tester guide:** [docker/USER_GUIDE.md](docker/USER_GUIDE.md)
>
> ```bash
> docker pull zihaojohnli/nemasize:cpu        # or :gpu
> docker run --rm -v /path/to/my_experiment:/data zihaojohnli/nemasize:cpu /data
> ```

### Developers / model trainers

If you want to retrain the detection or segmentation models, augment
data, or modify the pipeline, keep reading.

---

## Repository layout

| File / folder | Purpose |
|---|---|
| `run_pipeline.py` | End-to-end pipeline runner (detect → segment → skeletonize) |
| `detect_and_crop_rois.py` | Stage 1: YOLO detection + ROI cropping |
| `skeletonize_worms.py` | Stage 2: segmentation + centerline + length/width CSV |
| `create_roi_dataset.py` | Build a YOLO-seg training set from ROI crops |
| `train_yolo_segmentation.py` | Train the YOLO segmentation model |
| `train_rfdetr_segmentation.py` | (Optional) RF-DETR segmentation trainer — experimental |
| `dataset_manager.py` | Local dataset utilities: split, augment, visualize, COCO checks |
| `augment_data.py` | CLI wrapper around the augmentation routines |
| `convert_coco_to_yolo_seg.py` | Convert COCO annotations to YOLO-seg format |
| `visualize_predictions.py` / `visualize_contour_skeleton.py` | QC visualizations |
| `mask_well_imgs.py` | Mask out non-well regions in raw plate images |
| `docker/` | Dockerfiles, runtime requirements, end-user guide |
| `SLURM_scripts/` | Example HPC submission scripts |
| `name_lookup/`, `perform_test/` | Auxiliary analysis scripts |
| `runs/` | Training run outputs |

---

## Quick start (running the pipeline from source)

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Organize your data

```
my_experiment/
└── raw_images/
    ├── plate_001.tif
    ├── plate_002.tif
    └── ...
```

### 3. Run

```bash
# Local machine (uses the local model paths defined in run_pipeline.py)
python run_pipeline.py /path/to/my_experiment --local

# HPC / cluster (uses the cluster model paths)
python run_pipeline.py /path/to/my_experiment

# Override model weights directly
python run_pipeline.py /path/to/my_experiment \
    --detect-model /path/to/detect.pt \
    --seg-model    /path/to/seg.pt
```

Models can also be supplied via environment variables
`NEMASIZE_DETECT_MODEL` and `NEMASIZE_SEG_MODEL` (this is how the Docker
image points at its bundled weights).

### 4. Outputs

```
my_experiment/
├── inference_rois/
│   ├── images/              ← per-worm ROI crops (.png)
│   └── roi_catalog.json     ← ROI geometry for back-mapping
└── NemaSize_output/
    └── skeleton/
        ├── worm_lengths.csv         ← Length_um, Width_um per worm
        └── contour_skeleton_txt/    ← per-worm [CONTOUR] + [SKELETON]
```

See [docker/USER_GUIDE.md §8](docker/USER_GUIDE.md#8-understand-the-outputs)
for the full output schema.

---

## Training your own models

### Detection (Stage 1)

Stage 1 is a stock Ultralytics YOLO detection model. Train it with the
Ultralytics CLI/API on a COCO- or YOLO-format dataset of full-frame
plate images annotated with worm bounding boxes.

### Segmentation (Stage 2)

```bash
python train_yolo_segmentation.py
```

Edit the dataset path and hyperparameters at the top of
`train_yolo_segmentation.py`. The training set is typically the output
of `create_roi_dataset.py` (per-worm crops with polygon masks).

### Dataset utilities

`dataset_manager.py` handles split, augment, visualize, and format
conversion in one place. Most users only need:

```python
from dataset_manager import DatasetManager
manager = DatasetManager()

# Split + augment + visualize in one call
manager.process_dataset(
    dataset_path="./datasets/your_dataset",
    do_split=True, val_ratio=0.2,
    do_augment=True, augmentation_factor=3,
    visualize=True,
)
```

Supported annotation formats: **COCO** (recommended), YOLO, Pascal VOC.
Augmentations: rotation, intensity, Gaussian noise, and combinations
thereof — annotations are transformed automatically.

See [AUGMENTATION_GUIDE.md](AUGMENTATION_GUIDE.md) for details and
`dataset_manager.py` for the full API.

---

## Distribution

The pipeline is published as a Docker image:

| Tag | Description |
|---|---|
| `zihaojohnli/nemasize:cpu` | CPU-only (~2.5 GB on disk) |
| `zihaojohnli/nemasize:gpu` | NVIDIA CUDA 12.1 (~9 GB on disk) |
| `zihaojohnli/nemasize:1.0.1-beta-cpu` / `-gpu` | Pinned versioned tags (use these for published research) |

Build / publish instructions: [docker/DOCKER.md](docker/DOCKER.md).

The image is also Singularity- and Nextflow-compatible (includes
`procps`, built `--platform linux/amd64`).

---

## Status

NemaSize is currently in **public beta**. Bug reports, edge cases, and
feedback on the user guide are very welcome.

**Contact**

- Zihao John Li — <lizihaojohn@outlook.com>
- Erik Andersen (PI) — <erik.andersen@gmail.com>
