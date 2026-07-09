# NemaSize

Source code and data accompanying the manuscript:

"**Multiscale learning and topological analysis across complex postures enable robust nematode size quantification in pharmacological assays**"

Authors: Zihao (John) Li, Amanda O. Shaver, Michael E.G. Sauria, Jack Weinstein, Maya K. Mastronardo, Nikita S. Jhaveri, Kate Stone, Rachel Choo, Colin Lilley, Esha Sharma, Rohan Shrishrimal, Grayson Benson, Ariel Shi, Cecilia Soko, and Erik C. Andersen*

Affiliation: Department of Biology, Johns Hopkins University, Baltimore, MD 21218, USA

The manuscript will be deposited on bioRxiv.
This repository can be cloned from: <https://github.com/AndersenLab/NemaSize>

---

## Overview of NemaSize

NemaSize is AI-aided pipeline to measure nematode body sizes across complex posuture using multiscale learning and topology-aware skeletonization. NemaSize use a two-stage pipeline that turns raw well images into length and width measurements for individual worms:

```
raw images ──► [Stage 1] YOLO26-WF detection  ──► per-worm ROI crops
                                              │
                                              ▼
                            [Stage 2] YOLO26-WS high-resolution segmentation
                                              │
                                              ▼
                            Topology-aware centerline skeletonization
                                              │
                                              ▼
                                  worm lengths and widths (.csv)
                                  contour + skeleton (.txt)
```

---

## Repository layout

The repo is organized into four top-level modules:

| Module | Purpose |
|---|---|
| [`source_code/`](source_code/) | Source code for the pipeline: training, detection, segmentation, and skeletonization |
| [`figure_replication/`](figure_replication/) | Self-contained data and scripts to replicate the paper figures |
| [`docker/`](docker/) | Source code for building the docker container and instrusctions for deployment |
| [`misc/`](misc/) | Archieve of auxiliary scripts not required to run the pipeline |

### `source_code/`

| File / folder | Purpose |
|---|---|
| `run_pipeline.py` | Main end-to-end pipeline runner (detect → segment → skeletonize) |
| `detect_and_crop_rois.py` | Stage 1: YOLO detection → ROI cropping |
| `skeletonize_worms.py` | Stage 2: segmentation → centerline → length/width CSV |
| `speed_meter.py` | Speed benchmarking helper |
| `create_roi_dataset.py` | Build ROI image sets from full-well images for training YOLO26-WS |
| `train_yolo_segmentation.py` | Train the YOLO segmentation models |
| `dataset_manager.py` | Dataset utilities: split, augment, and visualization |
| `augment_data.py` | data augmentation |
| `convert_coco_to_yolo_seg.py` | Convert COCO annotations to YOLO format |
| `visualize_predictions.py` | Visualizations of YOLO inference results |
| `visualize_contour_skeleton.py` | Visualizations of contours and skeletons output from the pipeline |
| `requirements.txt` | Required dependencies for the full pipeline |
| `examples/` | Usage examples for `dataset_manager.py` (flatten/restore categories) |

### `figure_replication/`

| Folder | Purpose |
|---|---|
| `main_measurement_figures/` | Data + script to regenerate the NemaSize-vs-CellProfiler-vs-GT comparison figures |
| `dose_response_figures/` | Dose-response / EC10-50-90 figures |
| `roi_fov_calculations/` | Field-of-view calculations reported for select figures |

### `misc/`

| Folder | Purpose |
|---|---|
| `full_evaluation_pipeline/` | Full evaluation codebase behind `figure_replication/main_measurement_figures` (mask matching, stats) |
| `experimental_rfdetr/` | (Optional) RF-DETR segmentation trainer/inference — experimental, not part of the documented pipeline |
| `debug/` | One-off debugging scripts |
| `exploratory_analysis/` | Exploratory QC plotting scripts |
| `dataset_bookkeeping/` | Filename lookup / HPC data-wrangling utilities |
| `standalone_utilities/` | Standalone tools not wired into the pipeline (`mask_well_imgs.py`, `config.example.json`) |
| `SLURM_scripts/` | Example HPC submission scripts |

---

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


## Quick start (running the pipeline from source)

### 1. Install

```bash
pip install -r source_code/requirements.txt
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
python source_code/run_pipeline.py /path/to/my_experiment --local

# HPC / cluster (uses the cluster model paths)
python source_code/run_pipeline.py /path/to/my_experiment

# Override model weights directly
python source_code/run_pipeline.py /path/to/my_experiment \
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
python source_code/train_yolo_segmentation.py
```

Edit the dataset path and hyperparameters at the top of
`train_yolo_segmentation.py`. The training set is typically the output
of `create_roi_dataset.py` (per-worm crops with polygon masks).

### Dataset utilities

`dataset_manager.py` handles split, augment, visualize, and format
conversion in one place. Most users only need (run from inside `source_code/`,
or add that folder to your `PYTHONPATH`):

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

See [source_code/AUGMENTATION_GUIDE.md](source_code/AUGMENTATION_GUIDE.md) for
details and `dataset_manager.py` for the full API.

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
