"""
Builds a flat, self-contained working directory that lets the toxin
CellProfiler pipeline (from AndersenLab/cellprofiler-nf) be run headlessly
with `docker run cellprofiler/cellprofiler` directly -- i.e. without Nextflow
or Slurm/Singularity -- so its wall-clock speed can be compared against
NemaSize on the SAME set of images (the NemaSize speed-test dataset).

This mirrors what cellprofiler-nf's config_CP_input_toxin process does
(substitute pipeline placeholders + build metadata.csv), but keeps everything
in one flat folder and puts every input image into a single Metadata_Group
so a single `cellprofiler -c -r -p pipeline.cppipe -i . -o out` call times
the whole set in one process, comparable to how NemaSize is timed end-to-end.

Usage:
    python build_cellprofiler_speed_test.py
"""

import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
CPNF_DIR = Path(r"C:\Users\jl200\source\repos\AndersenLab\cellprofiler-nf")
RAW_PIPE = CPNF_DIR / "input_data" / "CP_pipelines" / "toxin-nf.cppipe"
WORM_MODEL_DIR = CPNF_DIR / "input_data" / "worm_models"
WELLMASK = CPNF_DIR / "input_data" / "well_masks" / "wellmask_98.png"

IMAGES_DIR = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Speed_test"
    r"\WormBodyDetection.v10i.yolo26\train\raw_images"
)

OUT_DIR = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Speed_test"
    r"\CellProfiler_speed_test"
)

# toxin pipeline uses 4 worm models (see main.nf worm_model1..4)
MODEL1 = "L4_N2_HB101_100w.xml"
MODEL2 = "L2L3_N2_HB101_100w.xml"
MODEL3 = "L1_N2_HB101_100w.xml"
MODEL4 = "MDHD.xml"

METADATA_CSV_NAME = "metadata.csv"
IMAGES_SUBDIR = "raw_images"  # images live here instead of the work dir root


def build(n_images: int | None = None) -> Path:
    """Assemble OUT_DIR with pipeline.cppipe, metadata.csv, models, images.

    n_images: if set, only copy the first N images (sorted) -- useful for a
    quick smoke test before running the full set.
    """
    work = OUT_DIR
    (work / "output").mkdir(parents=True, exist_ok=True)
    (work / IMAGES_SUBDIR).mkdir(parents=True, exist_ok=True)

    # -- worm models + well mask --------------------------------------------
    for model in (MODEL1, MODEL2, MODEL3, MODEL4):
        shutil.copy2(WORM_MODEL_DIR / model, work / model)
    shutil.copy2(WELLMASK, work / WELLMASK.name)

    # -- pipeline.cppipe: substitute placeholders ---------------------------
    raw_text = RAW_PIPE.read_text()
    pipeline_text = (
        raw_text.replace("METADATA_DIR", ".")
        .replace("METADATA_CSV_FILE", METADATA_CSV_NAME)
        .replace("WORM_MODEL_DIR", ".")
        .replace("MODEL1_XML_FILE", MODEL1)
        .replace("MODEL2_XML_FILE", MODEL2)
        .replace("MODEL3_XML_FILE", MODEL3)
        .replace("MODEL4_XML_FILE", MODEL4)
    )
    (work / "pipeline.cppipe").write_text(pipeline_text)

    # -- images + metadata.csv -----------------------------------------------
    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    if n_images is not None:
        image_paths = image_paths[:n_images]

    header = (
        "Metadata_Experiment,Metadata_Date,Metadata_Plate,Metadata_Well,"
        "Metadata_Group,Metadata_Magnification,"
        "Image_FileName_RawBF,Image_PathName_RawBF,"
        f"Image_FileName_{WELLMASK.name},Image_PathName_{WELLMASK.name}"
    )
    rows = [header]
    wells = [f"{r}{c:02d}" for r in "ABCDEFGH" for c in range(1, 13)]  # 96
    for i, img_path in enumerate(image_paths):
        shutil.copy2(img_path, work / IMAGES_SUBDIR / img_path.name)
        plate = f"p{i // 96 + 1:03d}"
        well = wells[i % 96]
        group = f"{plate}_{well}"
        rows.append(
            f"speedtest,20260928,{plate},{well},{group},m2x,"
            f"{img_path.name},{IMAGES_SUBDIR},{WELLMASK.name},."
        )
    (work / METADATA_CSV_NAME).write_text("\n".join(rows) + "\n")

    print(f"Wrote {len(image_paths)} images + metadata.csv to {work}")
    return work


if __name__ == "__main__":
    build()
