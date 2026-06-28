#!/usr/bin/env bash
# =============================================================================
# copy_missing_tifs.sh
# -----------------------------------------------------------------------------
# Copy the TIFs listed in missing_tifs_to_copy.txt from the shared raw_images
# directory into the per-user CP_missing/raw_images directory.
#
# Usage:
#   bash copy_missing_tifs.sh             # actually copy files
#   bash copy_missing_tifs.sh --dry-run   # just print what would happen
# =============================================================================
set -euo pipefail

SRC_DIR="/vast/eande106/projects/Mike/Data/WormSegmentation/raw_images"
DST_DIR="/vast/eande106/projects/John/NemaSeg/Datasets/John/CP_missing/raw_images"

# Resolve list file path relative to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_FILE="${SCRIPT_DIR}/../missing_tifs_to_copy.txt"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

if [[ ! -f "$LIST_FILE" ]]; then
    echo "ERROR: list file not found: $LIST_FILE" >&2
    exit 1
fi

mkdir -p "$DST_DIR"

n_ok=0
n_missing=0
n_skipped=0

while IFS= read -r fname || [[ -n "$fname" ]]; do
    # skip blanks / comments
    [[ -z "${fname// }" ]] && continue
    [[ "$fname" =~ ^# ]]   && continue

    src="${SRC_DIR}/${fname}"
    dst="${DST_DIR}/${fname}"

    if [[ ! -f "$src" ]]; then
        echo "MISSING: $src" >&2
        n_missing=$((n_missing + 1))
        continue
    fi

    if [[ -e "$dst" ]]; then
        echo "SKIP (already exists): $dst"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "DRY-RUN: cp $src -> $dst"
    else
        cp -v -p "$src" "$dst"
    fi
    n_ok=$((n_ok + 1))
done < "$LIST_FILE"

echo
echo "Done. copied=$n_ok  missing=$n_missing  skipped=$n_skipped"
