# NemaSize — Installation & Usage Guide (Beta)

Automated *C. elegans* body-length and width measurement from microscope images.
Two-stage pipeline: **YOLO detection → ROI segmentation → centerline skeletonization**.

This guide is for **end users (other labs)**. You do **not** need to install
Python, CUDA, PyTorch, or any other scientific package — only Docker.

> **Beta software.** Please report bugs, unexpected results, or unclear
> documentation to **Zihao John Li** (<lizihaojohn@outlook.com>), with **Erik
> Andersen** (<erik.andersen@gmail.com>) cc'd.

---

## Table of contents

1. [System requirements](#1-system-requirements)
2. [Pre-flight checklist](#2-pre-flight-checklist)
3. [Install Docker](#3-install-docker)
4. [(GPU users only) Verify GPU access](#4-gpu-users-only-verify-gpu-access)
5. [Pull the NemaSize image](#5-pull-the-nemasize-image)
6. [Prepare your data](#6-prepare-your-data)
7. [Run the pipeline](#7-run-the-pipeline)
8. [Understand the outputs](#8-understand-the-outputs)
9. [Updating to new versions](#9-updating-to-new-versions)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)
12. [Reporting issues](#12-reporting-issues)

---

## 1. System requirements

### Minimum (CPU image)

| Requirement | Value |
|---|---|
| OS | Windows 10/11, macOS 11+, or Linux (kernel 4.x+) |
| RAM | 8 GB (16 GB recommended for large datasets) |
| Disk | 5 GB free (image + outputs) |
| Internet | Required once for image download (~2.5 GB) |
| GPU | **Not required** |

### Recommended (GPU image, 5–20× faster)

| Requirement | Value |
|---|---|
| GPU | NVIDIA GPU with ≥4 GB VRAM (8 GB recommended) |
| Driver | NVIDIA driver ≥ 525.x (any driver from 2023 onward supports CUDA 12.1) |
| Disk | 12 GB free (image is ~9 GB) |
| Internet | Required once for image download (~6 GB) |

> **Apple Silicon Macs (M1/M2/M3):** the CPU image works but runs through
> Rosetta emulation (≈2× slower than a native amd64 CPU). The GPU image is
> NVIDIA-only; no Mac Metal/MPS support.

---

## 2. Pre-flight checklist

Before installing, verify the following on your machine. Each is a
one-line command — run them in order.

### Windows (PowerShell, run as your normal user)

```powershell
# 1. Confirm 64-bit Windows 10/11
[System.Environment]::OSVersion.Version
[System.Environment]::Is64BitOperatingSystem    # must print: True

# 2. Confirm virtualization is enabled (required for Docker Desktop)
Get-ComputerInfo -Property "HyperVRequirementVirtualizationFirmwareEnabled"
# Expect: True. If False, enable VT-x / AMD-V in BIOS.

# 3. (GPU users) Confirm NVIDIA driver
nvidia-smi
# Expect: a table showing your GPU, driver version, and (top-right)
# the maximum supported CUDA version. That number must be ≥ 12.1.
# If "command not found" or version is too old, see "Installing or
# updating the NVIDIA driver" below.
```

### macOS (Terminal)

```bash
# 1. Confirm OS version (need 11+)
sw_vers

# 2. Confirm chip type
uname -m            # x86_64 (Intel) or arm64 (Apple Silicon)
```

### Linux (any shell)

```bash
# 1. Confirm kernel + distro
uname -r ; cat /etc/os-release | head -3

# 2. Confirm 64-bit
uname -m            # must be x86_64

# 3. (GPU users) Confirm NVIDIA driver
nvidia-smi
```

If any check fails, fix it before proceeding.

### Installing or updating the NVIDIA driver (GPU users only)

The GPU image targets **CUDA 12.1**, which requires **NVIDIA driver
version ≥ 525.60.13** (Linux) or **≥ 528.33** (Windows). Any driver
released from early 2023 onward will work; newer drivers are
backward-compatible. You do **NOT** need to install the CUDA Toolkit on
the host — only the driver. CUDA itself ships inside the Docker image.

**How to read `nvidia-smi`:** the top-right cell shows the *maximum* CUDA
version your driver supports. As long as that number is **≥ 12.1**,
you're fine — even if it says 12.4, 13.x, etc.

#### Windows

1. Identify your GPU (Start → Device Manager → Display adapters), or run:
   ```powershell
   Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
   ```
2. Download the **Game Ready** or **Studio** driver from
   <https://www.nvidia.com/Download/index.aspx>:
   - Product Type: GeForce / RTX / Quadro / Tesla (whichever matches)
   - Operating System: **Windows 11 64-bit** (or Windows 10 64-bit)
   - Download Type: Studio Driver (more stable) or Game Ready Driver
3. Run the installer (accept the default "Express" install).
4. Restart Windows.
5. Re-run `nvidia-smi` — it should now print the driver table.

> **Tip:** if you already have **GeForce Experience** or **NVIDIA App**
> installed, just open it → *Drivers* tab → *Check for updates*.

#### Linux (Ubuntu / Debian)

```bash
# Easiest: use the distro-managed driver (works for most users)
sudo apt-get update
ubuntu-drivers devices                       # see what's recommended
sudo ubuntu-drivers autoinstall              # install the recommended driver
sudo reboot
nvidia-smi                                   # verify after reboot
```

If `ubuntu-drivers` isn't available (Debian, RHEL, etc.), follow NVIDIA's
official guide for your distro:
<https://www.nvidia.com/en-us/drivers/unix/>

#### Linux (RHEL / Rocky / CentOS / Fedora)

Use the official NVIDIA CUDA repository (driver only):
<https://developer.nvidia.com/cuda-downloads> → choose your distro →
follow the "driver" install path (skip the toolkit).

#### macOS

NVIDIA drivers are **not supported on macOS** — Apple removed CUDA
compatibility in 2018. Use the CPU image (`zihaojohnli/nemasize:cpu`)
instead.

#### What to do if `nvidia-smi` works on the host but not in Docker

That's a separate issue (NVIDIA Container Toolkit / Docker Desktop GPU
support), not a driver problem — see [Section 4](#4-gpu-users-only-verify-gpu-access).

---

## 3. Install Docker

### Windows / macOS

1. Download **Docker Desktop** from https://www.docker.com/products/docker-desktop/
2. Run the installer (accept defaults — WSL2 backend on Windows).
3. Restart your computer if prompted.
4. Launch Docker Desktop. Wait for the whale icon in the system tray to turn solid (~30 s).

### Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER          # so you can run docker without sudo
# Log out and back in for the group change to take effect
```

### Verify Docker works (all platforms)

```bash
docker --version
docker run --rm hello-world
```

You should see a "Hello from Docker!" message. If yes, proceed.

---

## 4. (GPU users only) Verify GPU access

The container needs to see your GPU through Docker's NVIDIA runtime.

### Windows

Docker Desktop ≥ 4.27 includes NVIDIA Container Toolkit support natively.
Just verify:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see the same GPU table that `nvidia-smi` prints on the host.

### Linux

You must install the NVIDIA Container Toolkit separately:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If the GPU shows up inside the test container, you're ready.

---

## 5. Pull the NemaSize image

### Choose your tag

| Tag | Size (download / disk) | Use when |
|---|---|---|
| `zihaojohnli/nemasize:cpu` | ~1 GB / ~2.5 GB | No GPU, or just trying things out |
| `zihaojohnli/nemasize:gpu` | ~3 GB / ~9 GB | NVIDIA GPU available; 5–20× faster |
| `zihaojohnli/nemasize:1.0.1-beta-cpu` | same as `:cpu` | **Reproducibility** — pin this exact build |
| `zihaojohnli/nemasize:1.0.1-beta-gpu` | same as `:gpu` | **Reproducibility** — pin this exact build |
| `zihaojohnli/nemasize:latest` | same as `:cpu` | Default; equivalent to `:cpu` |

> **For published research, ALWAYS use a versioned tag** (e.g.
> `1.0.1-beta-cpu`) so your analysis is reproducible. The moving tags
> (`:cpu`, `:gpu`, `:latest`) will change when new builds are pushed.

### Pull

```bash
docker pull zihaojohnli/nemasize:cpu          # CPU users
# OR
docker pull zihaojohnli/nemasize:gpu          # GPU users
```

This is a one-time download. Verify:

```bash
docker images zihaojohnli/nemasize
```

---

## 6. Prepare your data

The pipeline expects a single project folder with raw images in a
`raw_images/` subdirectory:

```
my_experiment/
└── raw_images/
    ├── plate_001.tif
    ├── plate_002.tif
    ├── plate_003.tif
    └── ...
```

**Supported image formats:** `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
(case-insensitive).

**Image content expectations:**
- One well per image with square aspect ratio
- The detector was trained at ~2× magnification; very different
  magnifications may need a custom-trained model

**You do NOT need to:**
- Pre-process, normalize, or convert image formats
- Create empty output folders (the pipeline creates them)
- Sort or rename files

---

## 7. Run the pipeline

The container reads from / writes to the project folder via a "bind mount"
(`-v <host>:<container>`). Outputs land directly back on your disk.

### CPU run

**Linux / macOS:**

```bash
docker run --rm \
  -v "$(pwd)/my_experiment:/data" \
  zihaojohnli/nemasize:cpu \
  /data
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  -v "${PWD}\my_experiment:/data" `
  zihaojohnli/nemasize:cpu `
  /data
```

**Windows (cmd.exe):**

```cmd
docker run --rm -v "%cd%\my_experiment:/data" zihaojohnli/nemasize:cpu /data
```

### GPU run

Add `--gpus all` and use the `:gpu` tag:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/my_experiment:/data" \
  zihaojohnli/nemasize:gpu \
  /data
```

### What the flags mean

| Flag | Meaning |
|---|---|
| `--rm` | Remove the container automatically when it exits (image stays). |
| `-v HOST:/data` | Mount your project folder into the container at `/data`. |
| `--gpus all` | Make all NVIDIA GPUs visible inside the container. |
| `zihaojohnli/nemasize:cpu` | The image to run. |
| `/data` | The argument passed to the pipeline (project root). |

### Live progress

The container prints a `tqdm` progress bar for each stage. Output is
streamed live (line-buffered), so you can monitor it directly.

### Stopping a run

Press `Ctrl+C`. The container shuts down cleanly; partial output stays on
disk and can be inspected.

---

## 8. Understand the outputs

After a successful run, your project folder will look like:

```
my_experiment/
├── raw_images/                       (unchanged input)
├── inference_rois/
│   ├── images/                       (cropped per-worm ROIs, .png)
│   └── roi_catalog.json              (ROI geometry for back-mapping)
└── NemaSize_output/
    └── skeleton/
        ├── worm_lengths.csv          ★ main results table
        └── contour_skeleton_txt/
            └── <image>_roi_<n>.txt   per-worm contour + skeleton coords
```

See [../README.md §5. Outputs](../README.md#5-outputs) for the full output
schema (`worm_lengths.csv` columns, the `.txt` contour/skeleton format,
`roi_catalog.json`, and visual QC options).

---

## 9. Updating to new versions

When a new version is released:

```bash
docker pull zihaojohnli/nemasize:cpu        # or :gpu
```

Docker downloads only changed layers, so updates are usually small (tens
of MB) unless model weights or CUDA were updated.

To **roll back** to a specific version:

```bash
docker pull zihaojohnli/nemasize:1.0.1-beta-cpu
docker run --rm -v "$(pwd)/my_experiment:/data" \
    zihaojohnli/nemasize:1.0.1-beta-cpu /data
```

---

## 10. Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Launch it and wait for the whale icon to
turn solid.

### "permission denied" writing outputs (Linux)
Container writes as root by default. Use:
```bash
docker run --rm --user $(id -u):$(id -g) -v "$(pwd)/my_experiment:/data" \
    zihaojohnli/nemasize:cpu /data
```

### "could not select device driver \"\" with capabilities: [[gpu]]"
- **Windows:** update Docker Desktop and your NVIDIA driver, then restart.
- **Linux:** install nvidia-container-toolkit (Section 4).

### "Raw images folder not found: /data/raw_images"
Your bind mount is wrong. The folder you mount must contain a
`raw_images/` subdirectory. Run `ls /path/to/my_experiment` and confirm
it's there.

### "invalid reference format" from Docker
The path in your `-v` flag has unquoted spaces or backslashes. **Wrap the
whole path in double quotes** as shown in Section 7.

### "CUDA out of memory"
Your GPU has less VRAM than the model needs (~2 GB combined). Use the
CPU image instead, or upgrade the GPU.

### Pipeline runs but outputs look wrong
- Confirm input images contain *C. elegans* worms at roughly the
  magnification expected by the model.
- Open a few `inference_rois/images/*.png` files — do they show centered,
  isolated worms? If not, the detector misfired.
- Spot-check `worm_lengths.csv` — are `Length_um` values in a plausible
  range (adult worms are typically ~1000µm)? Implausible values
  usually mean the segmentation collapsed or merged multiple worms.
- Report unusual cases to the maintainer (Section 12).

### Image pull is very slow / hangs
Docker Hub may rate-limit anonymous pulls. Run `docker login` with a free
Docker Hub account; signed-in pulls are faster and have higher quotas.

### "WARNING: The requested image's platform (linux/amd64) does not match the detected host platform"
You're on Apple Silicon (arm64). The image will still run via emulation —
this is just an informational warning, not an error.

---

## 11. FAQ

**Q: Do I need to install Python, CUDA, PyTorch?**
A: No. Everything is inside the container.

**Q: Can I run multiple experiments in parallel?**
A: Yes. Each `docker run` is isolated. Use different host folders for each
run. On a single GPU, only one container can use it at full speed at once.

**Q: Are my images uploaded anywhere?**
A: No. The container has no network access to your data. All processing
happens locally on your machine; only the *image download* uses the
internet (one time).

**Q: Can I use my own trained models instead of the bundled ones?**
A: Yes — override them at runtime:
```bash
docker run --rm \
  -v "$(pwd)/my_experiment:/data" \
  -v "$(pwd)/my_models:/models" \
  zihaojohnli/nemasize:cpu \
  /data --detect-model /models/my_detect.pt --seg-model /models/my_seg.pt
```

**Q: How do I cite NemaSize?**
A: *(citation info goes here once published)*

**Q: Does it work on Singularity / Apptainer (HPC clusters)?**
A: Yes. Convert with:
```bash
singularity pull nemasize_cpu.sif docker://zihaojohnli/nemasize:cpu
singularity run --bind /path/to/my_experiment:/data nemasize_cpu.sif /data
```

**Q: Can I use this with Nextflow / Snakemake?**
A: Yes. The image includes `procps` for process metric collection. Example
Nextflow process:
```groovy
process NEMASIZE {
    container 'zihaojohnli/nemasize:cpu'
    input:
        path project_dir
    output:
        path "${project_dir}/NemaSize_output/skeleton/*.csv"
    script:
        "/opt/nemasize/run_pipeline.py ${project_dir}"
}
```

---

## 12. Reporting issues

When something doesn't work, please collect and send:

1. **Image tag you ran**: e.g. `zihaojohnli/nemasize:1.0.1-beta-cpu`
2. **Full error output** from your terminal
3. **Output of**:
   ```bash
   docker --version
   docker info | grep -E "Server Version|Operating System|Architecture"
   docker image inspect zihaojohnli/nemasize:cpu --format "{{.Id}} {{.Os}}/{{.Architecture}}"
   ```
4. **A small example dataset** that reproduces the problem (1–3 images
   are usually enough), or a description of the input characteristics.

Send to **Zihao John Li** <lizihaojohn@outlook.com> (cc **Erik Andersen**
<erik.andersen@gmail.com>).

---

*Last updated: May 2026 · Version 1.0.1-beta*
