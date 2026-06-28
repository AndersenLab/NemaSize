# NemaSize Docker Guide

A complete, beginner-friendly walkthrough for **packaging NemaSize as a Docker
image** and **distributing it to other labs** via Docker Hub.

There are two audiences in this document:
1. **You (the maintainer)** — sections [1](#1-install-docker) → [6](#6-publish-to-docker-hub).
2. **End users (other labs)** — section [7](#7-end-user-instructions-share-this-with-other-labs).

---

## What goes into the image (and what does NOT)

The image is intentionally minimal — only what `run_pipeline.py` actually needs.

**Source files copied into the image (4 only):**
- `run_pipeline.py` — entry point
- `detect_and_crop_rois.py` — Stage 1
- `skeletonize_worms.py` — Stage 2
- `visualize_predictions.py` — supplies `get_image_files` to Stage 2

**Python packages installed** (see `docker/requirements-runtime.txt`):
`opencv-python`, `numpy`, `tqdm`, `scipy`, `scikit-image`, `skan`, `networkx`,
`ultralytics`, plus `torch` + `torchvision` (CPU wheels for `:cpu`, CUDA from
the base image for `:gpu`).

**System packages installed** (apt):
`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1` (OpenCV / scikit-image
runtime), plus `procps` (provides `ps`, required by **Nextflow** for task
process monitoring — omit and Nextflow trace metrics will be empty).

**Excluded from the image** (via explicit COPY + `.dockerignore`):
- Training scripts (`train_*.py`), RF-DETR (`rfdetr_inference.py`,
  `rfdetr` package), all dataset-management / debug / plotting scripts
- Sub-folders `name_lookup/`, `perform_test/`, `SLURM_scripts/`, `runs/`,
  `datasets/`, `NemaSize_output/`
- Heavy unused deps: `matplotlib`, `pandas`, `shapely`, `PyYAML`, `rfdetr`,
  `supervision`, `transformers`, `timm`, …
- Docs (`*.md`), git history, IDE configs, Python caches, the original
  `requirements.txt`, stray top-level `*.pt` files

If you later need an excluded script *inside* the container, add it to the
`COPY` list in the Dockerfile and rebuild.

---

## What is Docker, in 60 seconds

| Term | Plain-English meaning |
|---|---|
| **Image** | A frozen snapshot containing OS + Python + your code + dependencies. Read-only. |
| **Container** | A running process started from an image. Isolated from the host. |
| **Dockerfile** | A text recipe that builds an image, layer by layer. |
| **Registry** | A server that hosts images. Docker Hub is the default public one. |
| **Volume / bind mount** | A folder on the host made visible inside the container so the container can read/write user data. |
| **Tag** | A label like `nemasize:cpu` or `nemasize:1.0.0`. |

Other labs will not need to install Python, CUDA, ultralytics, or anything else
— just Docker. They run **one command**, the container processes their images,
and results land back on their disk.

---

## 1. Install Docker

- **Windows / Mac**: install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Launch it once so the daemon starts.
- **Linux**: install Docker Engine via your distro's package manager.

Verify:

```powershell
docker --version
docker run --rm hello-world
```

If `hello-world` prints a greeting, Docker is working.

---

## 2. Prepare your weights

The Dockerfiles bake the trained models into the image. Create a `weights/`
folder in the repo root with **two files at exactly these names**:

```
NemaSeg/
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── requirements-runtime.txt
│   └── DOCKER.md
├── .dockerignore        ← MUST stay at repo root
├── weights/
│   ├── detect.pt   ← YOLO detector  (was: runs/segment/worm_seg_train/weights/best.pt)
│   └── seg.pt      ← YOLO ROI segmenter (was: runs/.../roi_seg_train_fix_overlap/weights/best.pt)
├── run_pipeline.py
├── detect_and_crop_rois.py
├── skeletonize_worms.py
├── visualize_predictions.py
└── ...
```

> **Why `.dockerignore` is at the root, not in `docker/`:** Docker reads
> `.dockerignore` from the root of the *build context*, which is whatever
> directory you pass as the last argument to `docker build`. Since we need
> the context to be the repo root (so Docker can see all the source files),
> `.dockerignore` must live there too.

PowerShell:

```powershell
cd C:\Users\jl200\source\repos\NemaSeg
mkdir weights -ErrorAction SilentlyContinue
Copy-Item "C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\segment\worm_seg_train\weights\best.pt" weights\detect.pt
Copy-Item "C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt" weights\seg.pt
```

> **Note:** add `weights/` to `.gitignore` so model files are not committed to git.
> They live only inside the Docker image.

---

## 3. Build the CPU image

From the repo root:

```powershell
docker build --platform linux/amd64 -f docker/Dockerfile.cpu -t nemasize:cpu .
```

What this does, line by line:

- `--platform linux/amd64` — force x86_64 build (matters if the build host is ARM, e.g. Apple Silicon; HPC/SLURM nodes are almost always amd64).
- `-f docker/Dockerfile.cpu` — use the CPU recipe (lives in `docker/`).
- `-t nemasize:cpu` — tag the resulting image `nemasize` with version `cpu`.
- `.` — send the current directory (filtered by `.dockerignore`) as the build context.

> **Always run `docker build` from the repo root**, not from inside `docker/`.
> The Dockerfiles `COPY` source files using paths relative to the repo root.

First build downloads ~1 GB of base layers and installs PyTorch CPU. Subsequent
builds are fast thanks to layer caching.

When it finishes:

```powershell
docker images nemasize
```

You should see your image (~2–3 GB).

---

## 4. Build the GPU image (optional)

Only useful if **you** have an NVIDIA GPU on this build machine — Docker can
build a GPU image without a GPU, but you cannot **test** it without one.

```powershell
docker build --platform linux/amd64 -f docker/Dockerfile.gpu -t nemasize:gpu .
```

The GPU image is larger (~5–8 GB) because it includes CUDA + cuDNN runtime
libraries.

---

## 5. Test the image locally

The container expects the user's project folder mounted at `/data`. Inside that
folder there must be a `raw_images/` subdirectory — exactly what
`run_pipeline.py` already requires.

Replace the host path below with one of your real test datasets:

```powershell
docker run --rm `
  -v "C:\path\to\some\project:/data" `
  nemasize:cpu `
  /data
```

Explanation:
- `--rm` — auto-delete the container when it exits (image stays).
- `-v HOST:CONTAINER` — bind-mount your project folder into `/data`.
- `nemasize:cpu` — the image to run.
- `/data` — argument passed to `run_pipeline.py` (the `project_path`).

Outputs land in `C:\path\to\some\project\NemaSize_output\` on your host, just
like running the script natively.

### GPU run

```powershell
docker run --rm --gpus all `
  -v "C:\path\to\some\project:/data" `
  nemasize:gpu `
  /data
```

`--gpus all` requires Docker Desktop with WSL2 + NVIDIA drivers (Windows) or
`nvidia-container-toolkit` (Linux).

### Useful flags during testing

| Flag | Purpose |
|---|---|
| `-it` | Interactive terminal (e.g. for debugging). |
| `--entrypoint bash` | Drop into a shell instead of running the pipeline. |
| `--name nemasize-test` | Give the container a stable name. |

Example shell-in:

```powershell
docker run --rm -it --entrypoint bash -v "C:\path\to\project:/data" nemasize:cpu
```

---

## 6. Publish to Docker Hub

### 6a. One-time setup

1. Create a free account at <https://hub.docker.com>.
2. Log in from your machine:

   ```powershell
   docker login
   ```

### 6b. Tag with your Docker Hub username + version

Replace `YOURNAME` with your Docker Hub username and pick a version
(use [semver](https://semver.org), e.g. `1.0.0`):

```powershell
docker tag nemasize:cpu YOURNAME/nemasize:1.0.0-cpu
docker tag nemasize:cpu YOURNAME/nemasize:cpu
docker tag nemasize:cpu YOURNAME/nemasize:latest

docker tag nemasize:gpu YOURNAME/nemasize:1.0.0-gpu
docker tag nemasize:gpu YOURNAME/nemasize:gpu
```

A tag is just a name — no extra disk used.

### 6c. Push

```powershell
docker push YOURNAME/nemasize:1.0.0-cpu
docker push YOURNAME/nemasize:cpu
docker push YOURNAME/nemasize:latest

docker push YOURNAME/nemasize:1.0.0-gpu
docker push YOURNAME/nemasize:gpu
```

The first push uploads everything; later pushes only upload changed layers.

### 6d. Verify

Visit `https://hub.docker.com/r/YOURNAME/nemasize`. Anyone in the world can now
`docker pull YOURNAME/nemasize:cpu`.

### 6e. Releasing new versions

When you retrain a model or fix a bug:

```powershell
docker build --platform linux/amd64 -f docker/Dockerfile.cpu -t YOURNAME/nemasize:1.1.0-cpu -t YOURNAME/nemasize:cpu -t YOURNAME/nemasize:latest .
docker push --all-tags YOURNAME/nemasize
```

---

## 7. End-user instructions (share this with other labs)

> Copy-paste this section into your README or release notes.

### Requirements
- Docker Desktop (Windows/Mac) or Docker Engine (Linux).
- For GPU acceleration: NVIDIA GPU + recent driver.

### Get the image (~2 GB CPU / ~6 GB GPU; one-time download)

```bash
docker pull YOURNAME/nemasize:cpu          # CPU version
docker pull YOURNAME/nemasize:gpu          # GPU version (needs --gpus all)
```

### Prepare your data

Put the microscope images you want to analyze in a folder structured like:

```
my_experiment/
└── raw_images/
    ├── plate_001.tif
    ├── plate_002.tif
    └── ...
```

### Run

**Linux / Mac:**

```bash
docker run --rm \
  -v "$(pwd)/my_experiment:/data" \
  YOURNAME/nemasize:cpu \
  /data
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}\my_experiment:/data" `
  YOURNAME/nemasize:cpu `
  /data
```

**GPU:** add `--gpus all` and use the `:gpu` tag.

### Outputs

Results are written back to your host folder:

```
my_experiment/
├── raw_images/
├── inference_rois/                # cropped worm ROIs
└── NemaSize_output/
    └── skeleton/                  # CSV measurements + visualizations
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `permission denied` on Linux outputs | Add `--user $(id -u):$(id -g)` to the `docker run` command. |
| `could not select device driver "" with capabilities: [[gpu]]` | Install nvidia-container-toolkit (Linux) or update Docker Desktop + NVIDIA driver (Windows). |
| Out of memory on CPU run | Process fewer images at a time, or use the GPU image. |

---

## Appendix: Common Docker commands

```powershell
docker images                      # list images
docker ps                          # list running containers
docker ps -a                       # list all (including exited) containers
docker rm <container_id>           # remove a container
docker rmi <image_id>              # remove an image
docker logs <container_id>         # view container output
docker system df                   # show disk usage
docker system prune                # reclaim space (careful)
```
