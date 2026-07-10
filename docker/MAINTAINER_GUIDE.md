# NemaSize Docker Maintainer Guide

A complete, beginner-friendly walkthrough for **packaging NemaSize as a Docker
image** and **distributing it to other labs** via Docker Hub.

This document is for **maintainers** building and publishing the Docker
image. If you just want to *run* the pipeline via Docker, see
[USER_GUIDE.md](USER_GUIDE.md) instead.

---

## What goes into the image (and what does NOT)

The image is intentionally minimal — only what `run_pipeline.py` actually needs.

**Source files copied into the image (from `source_code/`, 5 total):**
- `source_code/run_pipeline.py` — entry point
- `source_code/detect_and_crop_rois.py` — Stage 1
- `source_code/skeletonize_worms.py` — Stage 2
- `source_code/visualize_predictions.py` — supplies `get_image_files` to Stage 2
- `source_code/speed_meter.py` — optional benchmarking helper

**Python packages installed** (see `docker/requirements-runtime.txt`):
`opencv-python`, `numpy`, `tqdm`, `scipy`, `scikit-image`, `skan`, `networkx`,
`ultralytics`, plus `torch` + `torchvision` (CPU wheels for `:cpu`, CUDA from
the base image for `:gpu`).

**System packages installed** (apt):
`libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender1` (OpenCV / scikit-image
runtime), plus `procps` (provides `ps`, required by **Nextflow** for task
process monitoring — omit and Nextflow trace metrics will be empty).

**Excluded from the image** (via explicit COPY + `.dockerignore`):
- Everything under `misc/` (training-adjacent, experimental RF-DETR, debug,
  exploratory-analysis, dataset-bookkeeping, and SLURM scripts) and
  `figure_replication/` (paper-figure data/scripts)
- Other `source_code/` files not explicitly COPYed above (`train_*.py`,
  `dataset_manager.py`, `augment_data.py`, etc.)
- Heavy unused deps: `matplotlib`, `pandas`, `shapely`, `PyYAML`, `rfdetr`,
  `supervision`, `transformers`, `timm`, …
- Docs (`*.md`), git history, IDE configs, Python caches, `source_code/requirements.txt`,
  stray top-level `*.pt` files

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
NemaSize/
├── docker/
│   ├── Dockerfile.cpu
│   ├── Dockerfile.gpu
│   ├── requirements-runtime.txt
│   └── MAINTAINER_GUIDE.md
├── .dockerignore        ← MUST stay at repo root
├── weights/
│   ├── detect.pt   ← YOLO26-WF detector
│   └── seg.pt      ← YOLO26-WS ROI segmenter
├── source_code/
│   ├── run_pipeline.py
│   ├── detect_and_crop_rois.py
│   ├── skeletonize_worms.py
│   ├── visualize_predictions.py
│   ├── speed_meter.py
│   └── ...
└── ...
```

> **Why `.dockerignore` is at the root, not in `docker/`:** Docker reads
> `.dockerignore` from the root of the *build context*, which is whatever
> directory you pass as the last argument to `docker build`. Since we need
> the context to be the repo root (so Docker can see all the source files),
> `.dockerignore` must live there too.

Use the same pretrained weights end users download from the
[GitHub Release](https://github.com/AndersenLab/NemaSize/releases/tag/v1.0.0)
(or point these at a newer locally-trained checkpoint before a new release):

PowerShell:

```powershell
cd C:\path\to\NemaSize
mkdir weights -ErrorAction SilentlyContinue
curl.exe -L -o weights\detect.pt https://github.com/AndersenLab/NemaSize/releases/download/v1.0.0/YOLO26-WF.pt
curl.exe -L -o weights\seg.pt    https://github.com/AndersenLab/NemaSize/releases/download/v1.0.0/YOLO26-WS.pt
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

## 7. Share the image with other labs

Don't write end-user instructions here — [USER_GUIDE.md](USER_GUIDE.md)
already covers installation, running the pipeline, understanding outputs,
troubleshooting, and FAQ for end users. Once you've pushed a new tag,
just make sure `USER_GUIDE.md`'s image tags/version references are current
(see §6b above for the tags you just pushed).

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
