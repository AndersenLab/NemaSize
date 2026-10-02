# NemaSize

Source code and data accompanying the manuscript:

"**Multiscale learning and topological analysis across complex postures enable robust nematode size quantification in pharmacological assays**"

**Authors**: Zihao (John) Li, Amanda O. Shaver, Michael E.G. Sauria, Jack Weinstein, Maya K. Mastronardo, Nikita S. Jhaveri, Kate Stone, Rachel Choo, Colin Lilley, Esha Sharma, Rohan Shrishrimal, Grayson Benson, Ariel Shi, Cecilia Soko, and Erik C. Andersen*

*Corresponding author

**Affiliation**: Department of Biology, Johns Hopkins University, Baltimore, MD 21218, USA

The manuscript has been deposited on [bioRxiv](https://doi.org/10.64898/2026.07.12.738008).
This repository can be cloned from: <https://github.com/AndersenLab/NemaSize>

---

## Overview of NemaSize

NemaSize is an AI-aided pipeline to measure nematode body sizes across complex posture using multiscale learning and topology-aware skeletonization. NemaSize uses a two-stage pipeline that turns raw well images into length and width measurements for individual worms:

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
| [`figure_replication/`](figure_replication/) | Self-contained data, bundled `output/` folders, and scripts to regenerate the paper figures — has its own lightweight `requirements.txt` and needs no access to the full pipeline, models, or raw experimental data |
| [`docker/`](docker/) | Source code for building the Docker container and instructions for deployment |
| [`misc/`](misc/) | Miscellaneous or deprecated files produced during pipeline development (not required to run the pipeline) |

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
| `augment_data.py` | Data augmentation |
| `convert_coco_to_yolo_seg.py` | Convert COCO annotations to YOLO format |
| `visualize_predictions.py` | Visualizations of YOLO inference results |
| `visualize_contour_skeleton.py` | Visualizations of contours and skeletons output from the pipeline |
| `flatten_categories.py` | Merge all dataset categories into a single category |
| `restore_categories.py` | Restore dataset categories from an automatic backup |
| `requirements.txt` | Required dependencies for the full pipeline |

### `figure_replication/`

| File / folder | Purpose |
|---|---|
| `Figs_4_and_S1/` | Data and scripts to regenerate Figs. 4 and S1 |
| `Fig_5/` | Data and scripts to regenerate Fig. 5 |
| `Fig_6/` | Data and script to regenerate Fig. 6 (NemaSize annotations on the public BBBC010 and BBBC011 datasets) |
| `requirements.txt` | Lightweight dependencies required by both figure folders |

See [Reproducing the paper figures](#reproducing-the-paper-figures) below for how to run these scripts.

### `docker/`

| File | Purpose |
|---|---|
| `Dockerfile.cpu` | Build the image running on CPU |
| `Dockerfile.gpu` | Build the image running on GPU |
| `requirements-runtime.txt` | Runtime dependencies bundled into the image |
| `USER_GUIDE.md` | User guide for running the pipeline using Docker container |
| `MAINTAINER_GUIDE.md` | Build / publish instructions for maintainers |

### `misc/`

| Folder | Purpose |
|---|---|
| `performance_evaluation/` | Scripts for evaluating pipeline performance |
| `experimental_rfdetr/` | RF-DETR trainer/inference - an experimental alternative segmentation method |
| `debug/` | *Ad hoc* debugging scripts |
| `preliminary_results_visualization/` | Preliminary scripts for results visualization |
| `dataset_bookkeeping/` | Data management utilities |
| `preliminary_utilities/` | Preliminary utilities for pipeline construction |
| `roi_fov_calculations/` | Field-of-view calculations for figure images |
| `SLURM_scripts/` | Preliminary HPC submission scripts |

---

## Reproducing the paper figures

Each figure has its own self-contained folder under `figure_replication/`
with bundled input data and an `output/` folder.

### 1. Install required dependencies

```bash
pip install -r figure_replication/requirements.txt
```

### 2. Run

Each script resolves its data/output paths relative to its own location, so
run it from inside its folder:

```bash
# Figs. 4 and S1
cd figure_replication/Figs_4_and_S1
python generate_figures.py

# Fig. 5
cd figure_replication/Fig_5
python generate_Fig_5A.py
python generate_Fig_5D_and_E.py
python calculate_ec_estimates.py

# Fig. 6
cd figure_replication/Fig_6
python generate_Fig_6.py
```

### 3. Outputs

Each script generates figures (`.svg`/`.png`) and any accompanying
statistics (`.txt`/`.csv`) to the `output/` folder, e.g.
`figure_replication/Fig_5/output/Fig_5A.png`.

---

## Instructions for running NemaSize

There are two options to run the NemaSize pipeline:
- [Option 1: Using Docker container](#option-1-using-docker-container)
- [Option 2: Using source code](#option-2-using-source-code)

### Option 1: Using Docker container

If you just want to run the pipeline on your own images, the easiest option is to use the
Docker image. You do not need to install dependencies or use any of the source code.

Follow the Docker instructions: [docker/USER_GUIDE.md](docker/USER_GUIDE.md)

### Option 2: Using source code

If you want to retrain the detection or segmentation models, augment
data, or modify the pipeline, run from source code.

#### 1. Install

```bash
pip install -r source_code/requirements.txt
```

#### 2. Organize your data

```
my_experiment/
└── raw_images/
    ├── plate_001.tif
    ├── plate_002.tif
    └── ...
```

#### 3. Download the pretrained models

The trained YOLO26-WF (detection) and YOLO26-WS (segmentation) weights are distributed as
[GitHub Release](https://github.com/AndersenLab/NemaSize/releases/tag/v1.0.0) assets:

```bash
curl -L -o YOLO26-WF.pt https://github.com/AndersenLab/NemaSize/releases/download/v1.0.0/YOLO26-WF.pt
curl -L -o YOLO26-WS.pt https://github.com/AndersenLab/NemaSize/releases/download/v1.0.0/YOLO26-WS.pt
```

Save the files to disk. You'll point the pipeline at them in the next step.

#### 4. Run

```bash
python source_code/run_pipeline.py /path/to/my_experiment \
    --detect-model /path/to/YOLO26-WF.pt \
    --seg-model    /path/to/YOLO26-WS.pt
```

Models can also be supplied via environment variables
`NEMASIZE_DETECT_MODEL` and `NEMASIZE_SEG_MODEL`.

> Internal note: `run_pipeline.py` also has hardcoded default model paths
> for the authors' own machines/HPC - cluster paths by default, or local
> paths if `--local` is passed. These paths won't exist for other users, so
> always pass `--detect-model`/`--seg-model` (or the env vars above) instead.

#### 5. Outputs

```
my_experiment/
├── raw_images/                       (unchanged input)
├── inference_rois/
│   ├── images/                       (cropped per-worm ROIs, .png)
│   └── roi_catalog.json              (ROI geometry for mapping back to full-well images)
└── NemaSize_output/
    └── skeleton/
        ├── worm_sizes.csv           ★ main results table
        └── contour_skeleton_txt/
            └── <image>_roi_<n>.txt   per-worm contour + skeleton coords
```

##### `worm_sizes.csv` columns

One row per detected worm.

| Column | Meaning |
|---|---|
| `Filename` | ROI filename (`<image>_roi_<n>.png`) |
| `Date` | Date parsed from the source filename (e.g. `20260226`) |
| `Metadata_Experiment` | Experiment tag parsed from filename (e.g. `cryassays`) |
| `Metadata_Plate` | Plate tag parsed from filename (e.g. `p002`) |
| `Magnification` | Magnification tag parsed from filename (e.g. `m2X`) |
| `Metadata_Well` | Well ID parsed from filename (e.g. `F07`) |
| `Worm_ID` | Per-image ROI index (0, 1, 2, …) |
| `Length_um` | Worm centerline length in micrometers |
| `Width_um` | Mean body width in micrometers |
| `Topology_Warnings` | flag codes for topology fallbacks encountered during centerline extraction; empty if none fired |

> **Filename convention:** the metadata columns above are populated by
> parsing the source image name as
> `YYYYMMDD-<experiment>-<plate>-<magnification>_<well>.tif`
> (e.g. `20260226-cryassays-p002-m2X_F07.tif`). If your filenames
> follow a different scheme, those metadata columns may be blank or
> incorrect — only `Filename`, `Worm_ID`, `Length_um`, and `Width_um`
> are guaranteed.

> **Units:** `Length_um` and `Width_um` are already in micrometers. The
> pipeline applies a built-in pixel-to-µm scale based on the
> `m<magnification>` tag in the filename. If your filenames don't
> carry magnification, you'll need to apply the conversion yourself
> from pixel coordinates (see the per-worm `.txt` files below).

##### `contour_skeleton_txt/<image>_roi_<n>.txt`

One file per worm. Plain text with two sections:

```
[CONTOUR]
x y          ← outline polygon, normalized to [0, 1] of the ROI image
x y
...
[SKELETON]
x y          ← centerline polyline, normalized to [0, 1] of the ROI image
x y
...
```

To recover pixel coordinates, multiply by the ROI's width/height
(available in `inference_rois/roi_catalog.json`). To recover original
full-image coordinates, additionally apply the ROI's offset from the
catalog.

##### `inference_rois/`

- **`images/*.png`** — each detected worm cropped from the original
  image. Useful for visual quality control of the detector and for re-running just
  the segmentation/skeleton stage.
- **`roi_catalog.json`** — ROI bounding boxes and offsets, indexed by
  source image. Required if you want to map results back to the
  original full-resolution coordinates.

##### Visual quality control

This release does **not** generate annotated overlay images
automatically. To inspect segmentation quality, you can use `visualize_contour_skeleton.py`
from `source_code/` (not bundled in the Docker runtime image).

---

### Training your own models

Both YOLO26-WF (Stage 1) and YOLO26-WS (Stage 2) are Ultralytics YOLO26x-seg models trained with the same script. They differ only in what data they're trained on:

- **YOLO26-WF** was trained on full-well images annotated with per-worm
  polygon masks.
- **YOLO26-WS** was trained on the cropped ROI worm images with polygon
  masks.

```bash
python source_code/train_yolo_segmentation.py
```

Edit the dataset path and hyperparameters at the top of
`train_yolo_segmentation.py`, then point it to your custom datasets:

- a full-well image dataset to train YOLO26-WF
- the ROI dataset produced by `create_roi_dataset.py` to train YOLO26-WS

#### Dataset utilities

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

Supported annotation formats: COCO, YOLO, Pascal VOC.
Augmentations: rotation, intensity, Gaussian noise, and combinations thereof. Annotations are transformed automatically.

See [source_code/AUGMENTATION_GUIDE.md](source_code/AUGMENTATION_GUIDE.md) for
details and `dataset_manager.py` for the full API.

---

## Parallelization on High-Performance Computing (HPC) cluster

For large datasets, a [Nextflow](https://www.nextflow.io/) pipeline,
**NemaSize-nf**, wraps this pipeline to parallelize runs on an HPC cluster:

> <https://github.com/AndersenLab/NemaSize-nf>

---

## License

This repository is released under the [Apache License 2.0](LICENSE).

---

## Citation

If you use NemaSize in your research, please cite:

> Li Z, Shaver AO, Sauria MEG, Weinstein J, Mastronardo MK, Jhaveri NS,
> Stone K, Choo R, Lilley C, Sharma E, Shrishrimal R, Benson G, Shi A,
> Soko C, Andersen EC. *Multiscale learning and topological analysis
> across complex postures enable robust nematode size quantification in
> pharmacological assays*. bioRxiv, 2026.

doi: [https://doi.org/10.64898/2026.07.12.738008](https://doi.org/10.64898/2026.07.12.738008)

Please also cite the repository itself if you use the code, the
pretrained models, or the Docker container:

> NemaSize (v1.0.0) [Computer software]. Andersen Lab, Johns Hopkins
> University. <https://github.com/AndersenLab/NemaSize>

---

## Contact

Bug reports, edge cases, and feedback are very welcome.

- Zihao (John) Li: <lizihaojohn@outlook.com>
- Erik C. Andersen (PI): <erik.andersen@gmail.com>
