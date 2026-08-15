#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
: "${PARAMETER_OPTION:?}" "${DELTA_K_MPA_SQRT_M:?}" "${OUTROOT:?}"
PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python}
CYCLES_MAX=${CYCLES_MAX:-100}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
STEPS=${STEPS:-20000}
N_PHASE=${N_PHASE:-48}
FAMILY_JSON=${FAMILY_JSON:-/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json}
REGISTRY=${REGISTRY:-arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv}
EXPECTED_HEAD=${EXPECTED_HEAD:-}
NORMALIZED_F=${NORMALIZED_F:-}
CLASS_LABEL=${CLASS_LABEL:-}
CURRENT_HEAD=$(git rev-parse HEAD)
CURRENT_BRANCH=$(git branch --show-current)
if [[ -n "$EXPECTED_HEAD" && "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "HEAD mismatch: expected $EXPECTED_HEAD, found $CURRENT_HEAD" >&2
  exit 2
fi
if [[ -n "$EXPECTED_HEAD" && -n "$(git status --porcelain)" ]]; then
  echo "authoritative launch requires a clean worktree" >&2
  exit 2
fi
[[ ! -e "$OUTROOT" ]] || { echo "output exists: $OUTROOT" >&2; exit 2; }
mkdir -p "$OUTROOT"
export CURRENT_HEAD CURRENT_BRANCH NORMALIZED_F CLASS_LABEL PARAMETER_OPTION
export DELTA_K_MPA_SQRT_M REGISTRY FAMILY_JSON HAZARD_SEED TARGET_EXT_UM CYCLES_MAX N_PHASE
"$PYTHON_BIN" - "$OUTROOT/hybrid_launch_contract.json" <<'PY'
import hashlib, json, os, pathlib, sys

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

payload = {
    "schema": "v10.2.32_explicit_2d_hybrid_contract_v1",
    "repository": str(pathlib.Path.cwd()),
    "repository_branch": os.environ["CURRENT_BRANCH"],
    "repository_head": os.environ["CURRENT_HEAD"],
    "repository_clean": True,
    "class": os.environ.get("CLASS_LABEL", ""),
    "parameter_option": os.environ["PARAMETER_OPTION"],
    "deltaK_MPa_sqrt_m": float(os.environ["DELTA_K_MPA_SQRT_M"]),
    "normalized_f": float(os.environ["NORMALIZED_F"]) if os.environ.get("NORMALIZED_F") else None,
    "temperature_K": 300.0,
    "R": 0.1,
    "frequency_Hz": 1000.0,
    "seed": int(os.environ.get("HAZARD_SEED", "1720")),
    "target_extension_um": float(os.environ.get("TARGET_EXT_UM", "100")),
    "maximum_cycles": float(os.environ.get("CYCLES_MAX", "100")),
    "phase_steps": int(os.environ.get("N_PHASE", "48")),
    "cycle_integration_mode": "explicit",
    "registry": str(pathlib.Path(os.environ["REGISTRY"]).resolve()),
    "registry_sha256": digest(os.environ["REGISTRY"]),
    "family_json": str(pathlib.Path(os.environ["FAMILY_JSON"]).resolve()),
    "family_sha256": digest(os.environ["FAMILY_JSON"]),
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
export PYTHONPATH="$ROOT"
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_HAZARD_SEED=${HAZARD_SEED:-1720}
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export KERNEL_STRICT_FAMILY_OVERRIDE=1
export V10230_ENERGY_GATE_ENABLED=1
export V10230_SAVE_ACTIVE_STATE_SNAPSHOT=1
export V10230_HIGH_CYCLE_CHECKPOINT_DIR="$OUTROOT"
export V10230_FORWARD_OUTER_PROPOSAL_CYCLES=1
"$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_32_explicit_lcf \
  --cycle-integration-mode explicit \
  --signed-kernel-family "$FAMILY_JSON" \
  --parameter-registry "$REGISTRY" \
  --parameter-option "$PARAMETER_OPTION" \
  --mode 2d --temperatures 300 --nx 36 --ny 72 --dt 8.4 --n-stagger 2 \
  --tip-h-fine 1e-6 --tip-ratio 1.20 --da-phys 5e-6 \
  --target-crack-extension-um "$TARGET_EXT_UM" \
  --front-state-model moving_pz --tip-source-model continuum \
  --tip-kinetics-mode moving_velocity --bulk-plasticity-mode tip_only \
  --directional-j-mode root_signed --tip-plasticity --active-shielding \
  --signed-active-shielding --mobile-shield-fraction 0 --no-wake-shielding \
  --crystal-aniso --crystal-compete --crystal-theta-deg 30 \
  --crystal-material w --j-decomposition cluster --max-fronts 1 \
  --crack-backend sharp_wake --dU 2e-7 --fatigue-cycles --fatigue-hold-load \
  --R 0.1 --frequency-Hz 1000 --cycle-block-mode hazard_limited \
  --min-block-cycles 1e-6 --target-dB .10 --n-phase "$N_PHASE" \
  --max-da-per-block-um 5 --print-every 1 --save-snapshots 0 --no-plots \
  --target-deltaK-MPa-sqrt-m "$DELTA_K_MPA_SQRT_M" \
  --steps "$STEPS" --cycles-max "$CYCLES_MAX" --block-cycles 1 \
  --max-block-cycles 1 --out "$OUTROOT" 2>&1 | tee "$OUTROOT/run.log"
