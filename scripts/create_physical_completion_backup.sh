#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /durable/independent/destination/fracture-completion-YYYYMMDD" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=$1
DEST_PARENT=$(dirname "$DEST")
COMPANION=/Volumes/Data/working-papers/Fracture/fem-czm-current-pf-companion
PF_FAMILY=/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/1447653d199f0b43cb475951092d69444c9b785f6fdf518c723792abb3b1f5e5
HISTORICAL_FEM=/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v10_0_5_17_paper7_self_contained/runs/v10_0_5_18_4_0_theta0_full_field_preflight/v913_paper_peak01_0242980_persistent_sites/T1000K

[[ -d "$DEST_PARENT" ]] || {
  echo "ERROR: destination parent does not exist: $DEST_PARENT" >&2
  exit 1
}
[[ ! -e "$DEST" ]] || {
  echo "ERROR: destination already exists; choose a new empty path: $DEST" >&2
  exit 1
}

SOURCE_DEVICE=$(df -P "$ROOT" | awk 'NR==2 {print $1}')
DEST_DEVICE=$(df -P "$DEST_PARENT" | awk 'NR==2 {print $1}')
[[ "$SOURCE_DEVICE" != "$DEST_DEVICE" ]] || {
  echo "ERROR: backup destination is on the same device as the source ($SOURCE_DEVICE)" >&2
  exit 1
}
FREE_KIB=$(df -Pk "$DEST_PARENT" | awk 'NR==2 {print $4}')
[[ "$FREE_KIB" -ge 524288 ]] || {
  echo "ERROR: backup destination has less than 512 MiB free" >&2
  exit 1
}

mkdir "$DEST"
mkdir "$DEST/source" "$DEST/evidence" "$DEST/dependencies" "$DEST/historical_fem"

git -C "$ROOT" archive --format=tar.gz --output="$DEST/source/shared-fracture-fatigue-options-v1.tar.gz" HEAD
git -C "$COMPANION" archive --format=tar.gz --output="$DEST/source/fem-czm-current-pf-companion.tar.gz" HEAD
rsync -a "$ROOT/integration_evidence/current_science_pilots_6abbf10/" "$DEST/evidence/integration_evidence/"
rsync -a "$ROOT/review_packet/current_science_pilots_6abbf10/" "$DEST/evidence/review_packet/"
rsync -a "$PF_FAMILY/" "$DEST/dependencies/current_pf_family_1447653/"
rsync -a "$HISTORICAL_FEM/" "$DEST/historical_fem/one_step_full_order_preflight/"

(
  cd "$DEST"
  find . -type f ! -name SHA256SUMS -exec shasum -a 256 {} + > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)

printf '%s\n' \
  '{' \
  '  "schema": "fracture-physical-completion-backup/1",' \
  "  \"source_device\": \"$SOURCE_DEVICE\"," \
  "  \"destination_device\": \"$DEST_DEVICE\"," \
  "  \"main_head\": \"$(git -C "$ROOT" rev-parse HEAD)\"," \
  "  \"companion_head\": \"$(git -C "$COMPANION" rev-parse HEAD)\"," \
  "  \"current_pf_family_sha256\": \"$(shasum -a 256 "$PF_FAMILY/family.json" | awk '{print $1}')\"," \
  '  "sha256_manifest_verified": true,' \
  '  "data_deleted": false' \
  '}' > "$DEST/BACKUP_COMPLETE.json"
shasum -a 256 "$DEST/BACKUP_COMPLETE.json"
echo "Verified independent backup complete: $DEST"
