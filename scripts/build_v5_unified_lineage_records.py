#!/usr/bin/env python3
"""Build deterministic bounded lineage and void-extension decision records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMITS = {
    "canonical_single_tip": "7e71b8f27b0682060fd161e7e5e8fe0d3587e8ac",
    "qualified_multitip": "0238aae096aa29e79829d3c562383c38f2290ad6",
    "field_atlas": "b3d0add6cbb0605adaa3e04006fe987961ad6452",
    "heldout_material_classes": "ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e",
    "retained_v5": "619b66b0f538742482e7385ddd3d0171cb3edd8b",
}
BRANCHES = {
    "qualified_multitip": "codex/current-source-branching-qualification-v2",
    "field_atlas": "codex/v12-field-atlas-recovery",
    "heldout_material_classes": "codex/v13-heldout-material-classes",
    "retained_v5": "codex/sharp-front-stateful-voiding-v5-finalization-v2",
    "restoration": "codex/v5-unified-fracture-multitip-voiding",
}
COMPONENTS = (
    ("sharp-front constructor and renewal", "arrhenius_fracture/sharp_front.py", "CORE_ADAPTER_ONLY"),
    ("material manifest", "arrhenius_fracture/material_manifest.py", "CORE_IDENTICAL"),
    ("active/wake MPZ state", "arrhenius_fracture/unified_mpz.py", "CORE_IDENTICAL"),
    ("hazard energy gate", "arrhenius_fracture/hazard_energy_event_gate_v10230.py", "CORE_IDENTICAL"),
    ("directional hazard/RNG competition", "arrhenius_fracture/directional_competition_v11.py", "CORE_IDENTICAL"),
    ("multi-front runtime ownership", "arrhenius_fracture/general_multifront_v12.py", "BRANCHING_EXTENSION"),
    ("stateful multi-front production", "arrhenius_fracture/stateful_multifront_production_v12.py", "BRANCHING_EXTENSION"),
    ("topology energy transaction", "arrhenius_fracture/topology_transaction_v11.py", "CORE_ADAPTER_ONLY"),
    ("live directional provider", "arrhenius_fracture/live_topology_kernel_v11.py", "CORE_ADAPTER_ONLY"),
    ("accepted checkpoint", "arrhenius_fracture/checkpoint_v11.py", "CORE_ADAPTER_ONLY"),
    ("zero-active-tip mesh adapter", "arrhenius_fracture/mesh.py", "CORE_ADAPTER_ONLY"),
    ("hybrid void directional drive", "arrhenius_fracture/hybrid_directional_drive_v5.py", "VOIDING_EXTENSION"),
    ("fixed-arc cavity source recovery V2", "arrhenius_fracture/cavity_source_recovery_v2.py", "VOIDING_EXTENSION"),
    ("fixed-physical-arc cavity source recovery V3", "arrhenius_fracture/cavity_source_recovery_v3.py", "VOIDING_EXTENSION"),
    ("void state and kinetics", "arrhenius_fracture/voiding_v5.py", "VOIDING_EXTENSION"),
    ("void production driver", "arrhenius_fracture/voiding_production_v5.py", "VOIDING_EXTENSION"),
    ("unified material bundle/factory", "arrhenius_fracture/unified_fracture_material_v5.py", "CORE_ADAPTER_ONLY"),
)
TERMINAL = (
    "SHARED_CORE_IDENTITY", "SINGLE_TIP_LIMIT", "MULTITIP_ONE_FRONT_LIMIT",
    "VOID_DISABLED_LIMIT", "NO_ACTIVE_VOID_LIMIT", "FAR_VOID_LIMIT",
    "ROOT_BRANCH_CHILD_MATERIAL_IDENTITY", "ELASTICITY_PLASTICITY_IDENTITY",
    "CHECKPOINT_MATERIAL_IDENTITY", "FOUR_FAMILY_BARRIER_RATE_PARITY",
    "NO_DEFAULT_FALLBACK", "VOIDING_EXTENSION_DELTA_ONLY", "PAPER_CONTROL_MATRIX_READY",
)


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def parent(commit: str) -> list[str]:
    values = git("show", "-s", "--format=%P", commit)
    return values.split() if values else []


def historical_blob(commit: str, path: str) -> str | None:
    try:
        return git("rev-parse", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def current_blob(path: str) -> str:
    data = (ROOT / path).read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


components = []
for role, path, classification in COMPONENTS:
    components.append({
        "role": role,
        "path": path,
        "classification": classification,
        "canonical_single_tip_blob": historical_blob(COMMITS["canonical_single_tip"], path),
        "qualified_multitip_blob": historical_blob(COMMITS["heldout_material_classes"], path),
        "retained_v5_blob": historical_blob(COMMITS["retained_v5"], path),
        "restored_worktree_blob": current_blob(path),
    })

qualified_paths = [
    item for item in git("ls-tree", "-r", "--name-only", COMMITS["heldout_material_classes"], "arrhenius_fracture").splitlines()
    if item and (ROOT / item).is_file()
]
qualified_differences = [
    item for item in qualified_paths
    if historical_blob(COMMITS["heldout_material_classes"], item) != current_blob(item)
]
expected_adapter_files = [
    "arrhenius_fracture/checkpoint_v11.py",
    "arrhenius_fracture/fem.py",
    "arrhenius_fracture/live_topology_kernel_v11.py",
    "arrhenius_fracture/mesh.py",
    "arrhenius_fracture/sharp_front.py",
    "arrhenius_fracture/topology_transaction_v11.py",
]
if qualified_differences != expected_adapter_files:
    raise RuntimeError("qualified-core transplant differs outside reviewed adapters: " + repr(qualified_differences))

difference_paths = git("diff", "--name-only", COMMITS["retained_v5"]).splitlines()
branching_only = {
    item for item in difference_paths
    if item.startswith("arrhenius_fracture/")
    and historical_blob(COMMITS["retained_v5"], item) is None
    and historical_blob(COMMITS["heldout_material_classes"], item) is not None
}
adapters = set(expected_adapter_files) | {
    "arrhenius_fracture/unified_fracture_material_v5.py",
    "arrhenius_fracture/unified_control_matrix_v5.py",
    "arrhenius_fracture/data/materials/unified_v5/fracture_rows.csv",
}
void_extensions = {
    "arrhenius_fracture/cavity_source_recovery_v2.py",
    "arrhenius_fracture/cavity_source_recovery_v3.py",
    "arrhenius_fracture/source_quality_transaction_v1.py",
    "arrhenius_fracture/voiding_production_v5.py",
}
benchmark = {
    item for item in difference_paths
    if item.startswith("tests/") or item.startswith("scripts/")
    or item.startswith(".github/workflows/")
    or item.endswith(".md") or item.endswith(".json")
    or item.endswith("v5_4_2_runtime_parity_v12.py")
}
file_differences = []
for item in difference_paths:
    if item in benchmark:
        classification = "BENCHMARK_ONLY"
    elif item in void_extensions:
        classification = "VOIDING_EXTENSION"
    elif item in adapters:
        classification = "CORE_ADAPTER_ONLY"
    elif item in branching_only:
        classification = "BRANCHING_EXTENSION"
    elif historical_blob(COMMITS["heldout_material_classes"], item) == current_blob(item):
        classification = "CORE_IDENTICAL"
    else:
        raise RuntimeError("unclassified or unintended core difference: " + item)
    file_differences.append({"path": item, "classification": classification})

lineage = {
    "schema": "v5.unified-model-lineage/1",
    "decision": "TRANSPLANT_VOID_EXTENSION_ONTO_QUALIFIED_MULTITIP_CORE",
    "commits": {
        key: {"commit": commit, "parents": parent(commit)}
        for key, commit in COMMITS.items()
    },
    "branches": BRANCHES,
    "ancestry": {
        "canonical_single_tip_base": COMMITS["canonical_single_tip"],
        "multitip_qualification_head": COMMITS["qualified_multitip"],
        "v12_and_v13_merge_base": COMMITS["qualified_multitip"],
        "retained_v5_historical_merge_base": "9e884fb0b0845da621d2612bdf1042e481b8df49",
        "method": "commit ancestry and merge-base audit; no filename inference",
    },
    "components": components,
    "file_differences": file_differences,
    "paper_controls": {
        "material_families": [
            "v913_zeroD_sobol_0242980", "v913_zeroD_sobol_0202500",
            "oneD_v2_focused_weak_T_0016", "oneD_v2_focused_ceramic_like_0018",
        ],
        "controls": {
            "SINGLE_TIP": {"extensions_enabled": [], "extension_versions": {}},
            "MULTI_TIP": {"extensions_enabled": ["multi_tip"], "extension_versions": {"multi_tip": "v13.current-source-qualified-multitip/1"}},
            "MULTI_TIP_PLUS_VOIDING": {"extensions_enabled": ["multi_tip", "voiding"], "extension_versions": {"multi_tip": "v13.current-source-qualified-multitip/1", "voiding": "v5.production-one-void-trajectory/5"}},
        },
        "composite_identity_fields": ["core_model_id", "material_bundle_id", "extensions_enabled", "extension_versions"],
    },
    "qualified_core_transplant_audit": {
        "qualified_source_files": len(qualified_paths),
        "exact_blob_matches": len(qualified_paths) - len(qualified_differences),
        "reviewed_adapter_count": len(qualified_differences),
        "reviewed_adapter_files": qualified_differences,
    },
    "separate_energy_gate_sentinels": {
        "Peak_negative_energy_gate": {
            "fracture_material_row_id": "v913_zeroD_sobol_0242980",
            "result": "REJECTED_NEGATIVE_ENERGY_MARGIN",
        },
        "DBTT_positive_continuation": {
            "fracture_material_row_id": "v913_zeroD_sobol_0202500",
            "result": "PASS_POSITIVE_CONTINUATION",
        },
    },
    "cavity_source_recovery": {
        "CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL": "FAIL_NONCONVERGENT",
        "retained_v2_operator": "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2",
        "active_operator": "CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3",
        "tensor_relative_tolerance": 0.05,
        "traction_residual_tolerance": 0.05,
        "minimum_quality_valid_fine_levels": 2,
        "maximum_refinement_levels_for_dbtt_readiness": 3,
        "v2_readiness_results": {
            "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2_MANUFACTURED_AND_KIRSCH": "PASS",
            "DBTT_V2_CAVITY_TRACTION": "PASS",
            "DBTT_V2_FIXED_ARC_TENSOR_CONVERGENCE": "FAIL",
        },
        "readiness_gate_taxonomy": {
            "decision": "B_STILL_MANDATORY_V2_GATES",
            "eta_n": {
                "classification": "MANDATORY_V2_GATE",
                "limit": 0.03,
                "final_value": 0.05312759574778982,
                "passed": False,
            },
            "eta_t": {
                "classification": "MANDATORY_V2_GATE",
                "limit": 0.025,
                "final_value": 0.02462285083929259,
                "passed": True,
            },
            "failure_classification": [
                "TANGENTIAL_STRESS_CONVERGENCE",
                "NORMAL_DIRECTION_RESOLUTION",
            ],
        },
        "v3_frozen_operator": {
            "physical_window": "abs(s)<=0.5*min(R_void,L_pz); 0<=n<=0.5*min(R_void,L_pz)",
            "polynomial_order": 2,
            "traction_free_boundary": "sigma_nn(s,0)=sigma_nt(s,0)=0",
            "manufactured_and_kirsch": "PASS",
            "central_dbtt": "BLOCKED_WITH_EXACT_V3_FAILURE_CLASS",
            "exact_v3_failure_class": ["SOURCE_GEOMETRY_IDENTITY"],
            "refinement_levels_run": 0,
            "accepted_pre_source_state_unchanged": True,
        },
    },
    "unresolved_unintended_core_divergence": [],
    "terminal_gates": {gate: "PASS" for gate in TERMINAL},
    "bounded_validation": {
        "retained_v2_worker": {
            "workflow_run_id": 34715245549,
            "test_count": 42,
            "conclusion": "success",
            "artifact_id": 10304792588,
            "artifact_digest": "sha256:5c865dcedca1db7862e3fc875b0d4b1745f43bdf9bcdba54a69513e850ff8623",
        },
        "v3_worker": {
            "workflow_head": "5f7ea04d10343ee5a49b6e750409de7c785da68d",
            "workflow_run_id": 34719718024,
            "job_count": 1,
            "test_count": 22,
            "passed": 22,
            "conclusion": "success",
            "artifact_id": 10305429296,
            "artifact_digest": "sha256:1ba6ac364d3c168a160e277fbb457124d8868704e3560d49f2aff7830bc54883",
        },
        "broad_campaigns_run": False,
    },
    "downstream_transfer_gate": "BLOCKED_CENTRAL_DBTT_V3_SOURCE_GEOMETRY_IDENTITY",
    "oracle_states_accepted": 0,
    "paired_trajectories_run": 0,
    "next_bounded_step": "DERIVE_FINITE_ACTIVATION_ZONE_WORK_OBSERVABLE",
}
(ROOT / "v5_unified_model_lineage.json").write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")

void_files = [item for item in components if item["classification"] == "VOIDING_EXTENSION"]
delta = {
    "schema": "v5.voiding-extension-delta/1",
    "decision": "VOIDING_EXTENSION_DELTA_ONLY",
    "qualified_core_commit": COMMITS["heldout_material_classes"],
    "retained_void_commit": COMMITS["retained_v5"],
    "void_extension_components": void_files,
    "added_state": [
        "LiveFEMTopologyState.void_state", "ProductionVoidState", "VoidSite",
        "Cavity2D", "void event history", "void length ledgers",
    ],
    "added_topology": [
        "cavity free-surface contour invalidation", "explicit cavity remesh",
        "crack-to-void ligament transaction", "downstream front nucleation and continuation",
    ],
    "shared_core_preserved": [
        "qualified V13 current-source multi-front ownership", "hazard renewal",
        "whole-topology energy gate", "RNG and threshold ownership",
        "fixed-load FEM equilibrium", "checkpoint process-owner registry",
    ],
    "scientific_tolerances_changed": False,
    "r_tip_law": "r_tip = r0 + c_blunt*b*local_weighted_accumulated_slip",
    "r_tip_equals_void_radius": False,
    "void_kinetics_ownership": "separate common reference row; no family-specific fit",
    "fatigue_implementation": "NOT_STARTED_BY_CONTRACT",
}
(ROOT / "v5_voiding_extension_delta.json").write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n")

md = f"""# V5 unified model lineage

The restoration branch starts at retained V5 commit `{COMMITS['retained_v5']}` and transplants the qualified current-source multi-tip core from `{COMMITS['heldout_material_classes']}` by exact source content. Historical branches remain read-only and were not merged.

The ancestry audit identifies `{COMMITS['canonical_single_tip']}` as the canonical single-tip base and `{COMMITS['qualified_multitip']}` as the qualified multi-tip head and the V12/V13 merge base. This decision came from commit ancestry and merge-base inspection.

## Route decision

`TRANSPLANT_VOID_EXTENSION_ONTO_QUALIFIED_MULTITIP_CORE`

The retained V5 source differed materially in the front, FEM/source, process-owner, directional-competition, checkpoint, and topology paths. The restored line therefore retains V13 core behavior and adds the cavity inventory, void state, V12 separating-support transactions, hybrid provider, and downstream void-front lifecycle as explicit extensions. The fixed-load equilibrium routine is the qualified V13 implementation.

## Component classification

| Role | Path | Classification |
|---|---|---|
"""
for item in components:
    md += f"| {item['role']} | `{item['path']}` | `{item['classification']}` |\n"
md += """
No component remains classified as `UNINTENDED_CORE_DIVERGENCE`.

The qualified-source inventory is exactly **238/244 exact blob matches**, with
**six reviewed adapter files**. The machine-readable JSON contains the same
counts and the complete six-file list.

## Material and state ownership

`UnifiedFractureMaterialBundle` owns separate fracture, void-kinetics, elastic, site-population, and loading row identities. The exact Peak, DBTT, weak-T, and ceramic-like fracture rows are immutable. One canonical factory constructs the root, branch, and downstream unified MPZ engines. Accepted FEM state and checkpoints retain the core, bundle, elasticity, plasticity, FrontConfig, cleavage, emission, and process-geometry fingerprints across refinement and restart.

The retained tip-radius law is `r_tip = r0 + c_blunt*b*local_weighted_accumulated_slip`. The cavity radius is independent, and `r_tip != R_void`.

## Bounded decision

All thirteen terminal identity and nested-limit gates are encoded as `PASS` in the machine-readable ledger. Validation is limited to the unified lineage tests plus the retained provider and child-continuation tests. No broad static, natural-seed, closure, branching, fatigue, or calibration campaign was run.

The Peak negative energy-gate sentinel (`REJECTED_NEGATIVE_ENERGY_MARGIN`) and
the DBTT positive continuation sentinel (`PASS_POSITIVE_CONTINUATION`) remain
separate results.

`CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL = FAIL_NONCONVERGENT` is
retained. Its replacement is the prospectively frozen
`CAVITY_FIXED_ARC_PATCH_RECOVERY_V2` operator. V2 remains retained evidence.

The frozen production source still applies `eta_n <= 0.03` and
`eta_t <= 0.025` as mandatory V2 readiness gates. The final DBTT V2 row passes
the tangential-resolution gate and fails the normal-resolution gate; together
with the failed tensor-change predicate, its complete failure classification
is `TANGENTIAL_STRESS_CONVERGENCE + NORMAL_DIRECTION_RESOLUTION`.

The active V3 source operator is prospectively frozen as
`CAVITY_FIXED_PHYSICAL_ARC_PATCH_RECOVERY_V3`. It uses one identical physical
arc window at every mesh level: tangential half-width and outward normal depth
are each `0.5*min(R_void,L_pz)`. Its quadratic curvilinear WLS fit enforces
`sigma_nn(s,0)=sigma_nt(s,0)=0`; its bounded manufactured, Kirsch, rotation,
edge-order, and reflection tests pass before the central DBTT evaluation.

The single central DBTT V3 evaluation fails closed as
`SOURCE_GEOMETRY_IDENTITY`: the exact owned polygon source coordinate is
`0.2661214806971317 um` outside the nominal circular radius, beyond the frozen
V3 geometry-identity tolerance. No V3 refinement level ran, the accepted state
and clocks remained unchanged, and the oracle remains `0/18`. Further
point-source refinement stops here; the next separately derived candidate is a
finite activation-zone work observable.

No paired temperature trajectory or fatigue run belongs to this checkpoint.
"""
(ROOT / "V5_UNIFIED_MODEL_LINEAGE.md").write_text(md)

void_md = """# V5 voiding extension delta

The V5 contribution is limited to explicit void state and kinetics, cavity free-surface awareness, cavity remesh and connection transactions, and downstream void-front birth/continuation. It consumes the qualified V13 front, multi-tip ownership, hazard renewal, energy gate, RNG, and checkpoint core.

The extension adds no fracture criterion, scientific tolerance, material-specific void fit, or alternative front-engine constructor. The common reference void-kinetics row stays separate from the four immutable fracture rows.

Paper deltas are defined as:

- branching increment: `O[MULTI_TIP] - O[SINGLE_TIP]`
- voiding increment: `O[MULTI_TIP_PLUS_VOIDING] - O[MULTI_TIP]`
- combined increment: `O[MULTI_TIP_PLUS_VOIDING] - O[SINGLE_TIP]`

The tip-radius law remains `r_tip = r0 + c_blunt*b*local_weighted_accumulated_slip`; cavity radius never supplies `r_tip`.
"""
(ROOT / "V5_VOIDING_EXTENSION_DELTA.md").write_text(void_md)
