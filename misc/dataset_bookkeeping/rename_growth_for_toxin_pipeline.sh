#!/usr/bin/env bash
# =============================================================================
# rename_growth_for_toxin_pipeline.sh
# -----------------------------------------------------------------------------
# Context:
#   The andersenlab/cellprofiler-nf pipeline (modes: toxin, dauer) parses image
#   filenames with a strict regex that expects:
#       <DATE>-<EXP>-p<PLATE>-m<MAG>X_<WELL>.TIF
#     e.g. 20210205-assayA-p007-m2X_H01.TIF
#
#   Our 13 "growth" images follow a different convention:
#       p<PLATE>-growth-H<HID>-2X_<WELL>.TIF
#     e.g. p01-growth-H01-2X_A02.TIF
#
#   Neither the toxin nor the dauer config script accepts that pattern, so
#   runCP receives no inputs and the pipeline fails downstream.
#
# Workaround (this script):
#   Create *symbolic links* in a SEPARATE directory using a synthesized
#   filename that satisfies the toxin regex. The originals in raw_images/
#   are not touched. The symlinks live in their own folder so the two
#   filename conventions never get mixed.
#
#   Synthesis rule:
#       p01-growth-H01-2X_A02.TIF  ->  20210101-growthH01-p01-m2X_A02.TIF
#
#       Field       Source                       Notes
#       ----------  ---------------------------  --------------------------
#       DATE        today's date (YYYYMMDD)      placeholder; growth files
#                                                have no real acquisition
#                                                date encoded in the name
#       EXP         "growth" + H-tag (e.g. H01)  preserves the H-id so we
#                                                can recover the original
#                                                file from the synthesized
#                                                name later
#       PLATE       the leading p<NN>            unchanged
#       MAG         m2X                          fixed (all growth images
#                                                are 2X)
#       WELL        trailing <[A-Z][0-9]{2}>     unchanged
#
# Caveats (READ BEFORE USING THE RESULTS):
#   * This only makes the *file-discovery* step happy. The toxin pipeline's
#     .cppipe was tuned for the toxin assay and may not be appropriate for
#     growth images. Numbers produced via this workaround are not guaranteed
#     to be comparable to numbers from whatever pipeline originally produced
#     the growth rows in our combined CSV.
#   * The synthesized DATE/EXP are placeholders, so any downstream join on
#     Metadata_Date or Metadata_Experiment must be remapped through the
#     audit log (rename_audit.csv) written by this script.
#
# Usage:
#   bash rename_growth_for_toxin_pipeline.sh <src_dir> [<dest_dir>]
#
#     <src_dir>   directory containing the original p*-growth-H*-2X_*.TIF
#                 files (typically .../CP_missing/raw_images).
#     <dest_dir>  directory in which to write the renamed symlinks.
#                 Defaults to "<parent-of-src>/renamed_images/raw_images",
#                 so the parent folder ("renamed_images") can be used
#                 directly as a nextflow --project dir (it already has a
#                 raw_images/ subdir containing only regex-compatible
#                 filenames).
#                 Created if missing. Must NOT equal <src_dir>.
#
#   Example:
#       bash rename_growth_for_toxin_pipeline.sh \
#            /vast/.../CP_missing/raw_images
#       # -> symlinks in /vast/.../CP_missing/renamed_images/raw_images/
#       # -> audit at  /vast/.../CP_missing/renamed_images/rename_audit.csv
#       # then:
#       nextflow run -latest andersenlab/cellprofiler-nf \
#            --pipeline toxin \
#            --project /vast/.../CP_missing/renamed_images
#
# Output:
#   * Symlinks (with absolute targets) inside <dest_dir>.
#   * "<parent-of-dest>/rename_audit.csv" recording every mapping.
# =============================================================================

set -euo pipefail

SRC="${1:-}"
DEST="${2:-}"

if [[ -z "$SRC" || ! -d "$SRC" ]]; then
    echo "Usage: $0 <src_dir> [<dest_dir>]" >&2
    exit 2
fi

# Resolve to absolute paths so symlink targets are unambiguous.
SRC="$(cd "$SRC" && pwd -P)"

if [[ -z "$DEST" ]]; then
    DEST="$(dirname "$SRC")/renamed_images/raw_images"
fi

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd -P)"

if [[ "$SRC" == "$DEST" ]]; then
    echo "Refusing to write symlinks into the source directory. Pick a different <dest_dir>." >&2
    exit 2
fi

AUDIT="$(dirname "$DEST")/rename_audit.csv"
echo "original,original_path,symlink,symlink_path,plate,hid,well,synthesized_date,synthesized_exp,magnification" > "$AUDIT"

shopt -s nullglob
cd "$SRC"
matches=(p*-growth-H*-2X_*.TIF)
if [[ ${#matches[@]} -eq 0 ]]; then
    echo "No growth-pattern files found in: $SRC" >&2
    exit 0
fi

created=0
for f in "${matches[@]}"; do
    if [[ ! "$f" =~ ^(p[0-9]+)-growth-(H[0-9]+)-2X_([A-Z][0-9]{2})\.TIF$ ]]; then
        echo "Skipping (unexpected name): $f" >&2
        continue
    fi
    plate="${BASH_REMATCH[1]}"
    hid="${BASH_REMATCH[2]}"
    well="${BASH_REMATCH[3]}"

    date_tag="$(date +%Y%m%d)"
    exp_tag="growth${hid}"
    mag_tag="m2X"

    new="${date_tag}-${exp_tag}-${plate}-${mag_tag}_${well}.TIF"
    target="${SRC}/${f}"
    linkpath="${DEST}/${new}"

    if [[ -e "$linkpath" || -L "$linkpath" ]]; then
        echo "Skipping (target already exists): $linkpath"
    else
        ln -sv "$target" "$linkpath"
        created=$((created + 1))
    fi

    echo "${f},${target},${new},${linkpath},${plate},${hid},${well},${date_tag},${exp_tag},${mag_tag}" >> "$AUDIT"
done

echo
echo "Source dir : $SRC"
echo "Dest dir   : $DEST"
echo "Created    : $created symlink(s)"
echo "Audit log  : $AUDIT"
