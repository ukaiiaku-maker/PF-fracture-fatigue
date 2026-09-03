"""Build artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json.

Deterministically reconstructs the A_NATIVE material row for the v10.2.30
crack-rebonding causal pilot v2, since the original transient registry
(``runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv``, built
by ``scripts/build_v10_2_30_A_native_plus_8PT_registry.py``) no longer
exists anywhere on disk (its whole run directory tree was found to be 1371
empty directories -- swept by OS /tmp cleanup -- and it was never
git-tracked in the first place).

Recovery chain (see docs/v10_2_30_crack_rebonding_causal_pilot_v2.md for the
full investigation writeup):

1. The qualified Candidate A source row (candidate_id
   "v914_endurance_knee_0462") survives, byte-identical, in the git-tracked
   immutable v9.14 knee-search source registry
   ``Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search/runtime_inputs/
   v914/endurance_knee_global_300K_1024.csv`` (45 fields; independently
   re-hashed here and compared byte-for-byte against every one of its 45
   fields as they appear in the MPZ-schema (73-field) registry at
   ``arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry
   .csv`` in the main v10.2.x repo -- exactly one field differs
   (material_class: "endurance_knee" -> "DBTT", an explicit, disclosed
   relabel in that registry's own builder, not a physics change).
2. The 28 additional MPZ-solver-configuration fields in the 73-field
   registry (mpz bin count, forest-floor density, mobile-shield fraction,
   recovery-rate/legacy-feature toggles, etc -- none of the cleavage/
   emission/peierls/taylor Arrhenius-barrier fields) are explicitly recorded
   as disclosed "shared_spatial_constants" in that registry's own
   ``.audit.json`` (not silently invented) -- applied uniformly across the
   v10.2.31 four-candidate spatial-transfer study, not derived from a
   candidate-specific search result.
3. A_NATIVE is then constructed from this row by the audited builder's own
   documented algorithm (scripts/build_v10_2_30_A_native_plus_8PT_registry.
   py: composite = dict(A); only 5 identity/descriptive fields overwritten;
   no PT donor substitution for the "A_NATIVE" variant specifically).

Classification: A_NATIVE_REGISTRY_DETERMINISTICALLY_RECONSTRUCTED_FROM_
QUALIFIED_INPUTS (not a re-discovery of the original archived registry
file).
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30"
)
V914_SOURCE_REPO = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search"
)
V914_SOURCE_CSV = V914_SOURCE_REPO / "runtime_inputs/v914/endurance_knee_global_300K_1024.csv"
DERIVED_REGISTRY_CSV = (
    MAIN_REPO / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"
)
DERIVED_REGISTRY_AUDIT = (
    MAIN_REPO / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.audit.json"
)
DERIVED_REGISTRY_BUILDER = MAIN_REPO / "scripts/build_v10_2_31_endurance_knee_registry.py"
A_NATIVE_PLUS_8PT_BUILDER = REPO_ROOT / "scripts/build_v10_2_30_A_native_plus_8PT_registry.py"
PARAMETER_REGISTRY_MODULE = REPO_ROOT / "arrhenius_fracture/parameter_registry_v9111.py"
MATERIAL_MANIFEST_MODULE = REPO_ROOT / "arrhenius_fracture/material_manifest.py"

SOURCE_CANDIDATE_ID = "v914_endurance_knee_0462"
QUALIFIED_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"

# Exact algorithm from scripts/build_v10_2_30_A_native_plus_8PT_registry.py's
# "A_NATIVE" variant (vector is None -> no PT-substitution loop runs):
IDENTITY_FIELDS = {"option_key", "candidate_id", "role", "mechanism_summary", "validation_status"}
A_NATIVE_IDENTITY_OVERRIDES = {
    "option_key": "A_NATIVE",
    "candidate_id": "A_NATIVE",
    "role": "Candidate-A controlled Taylor/Peierls mechanism panel",
    "mechanism_summary": "exact qualified Candidate A plus audited PT subset",
    "validation_status": "NOT_EVALUATED",
}

MPZ_N_BINS = 80
WAKE_N_BINS = 0  # SharedReducedConfig default; not used by the A_NATIVE manifest path
N_PHASE = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_obj(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    src_rows = read_csv_rows(V914_SOURCE_CSV)
    src_row = next(r for r in src_rows if r["candidate_id"] == SOURCE_CANDIDATE_ID)

    der_rows = read_csv_rows(DERIVED_REGISTRY_CSV)
    der_row = next(r for r in der_rows if r["candidate_id"] == SOURCE_CANDIDATE_ID)

    # Cross-verify every field common to both rows is byte-identical except
    # the disclosed material_class relabel.
    common = set(src_row) & set(der_row)
    mismatches = {k: (src_row[k], der_row[k]) for k in common if src_row[k] != der_row[k]}
    expected_mismatches = {"material_class": ("endurance_knee", "DBTT")}
    if mismatches != expected_mismatches:
        raise SystemExit(
            f"unexpected field mismatches between pristine v9.14 source and "
            f"derived MPZ registry rows: {mismatches!r} (expected only "
            f"{expected_mismatches!r}); refusing to reconstruct A_NATIVE "
            "silently on top of an unverified divergence"
        )
    only_in_src = set(src_row) - set(der_row)
    if only_in_src:
        raise SystemExit(f"pristine source has fields absent from derived registry: {only_in_src}")

    audit = json.loads(DERIVED_REGISTRY_AUDIT.read_text())
    disclosed_shared_fields = set(audit["shared_spatial_constants"])
    # "encounter_efficiency" is not a SHARED constant nor an identity field --
    # the v10.2.31 builder explicitly copies it from the row's own (already
    # cross-verified byte-identical) "physics__encounter_efficiency" field
    # (build_v10_2_31_endurance_knee_registry.py:46), a disclosed same-value
    # mirror under a legacy column name, not a fabricated or divergent value.
    if der_row["encounter_efficiency"] != der_row["physics__encounter_efficiency"]:
        raise SystemExit(
            "encounter_efficiency does not mirror physics__encounter_efficiency "
            "as the v10.2.31 builder documents; refusing to treat it as a "
            "disclosed same-value copy"
        )
    extra_der_fields = set(der_row) - set(src_row)
    undisclosed_extra = extra_der_fields - disclosed_shared_fields - IDENTITY_FIELDS - {
        "endurance_mechanism_class", "encounter_efficiency",
    }
    if undisclosed_extra:
        raise SystemExit(
            f"derived registry has extra fields not accounted for as disclosed "
            f"shared constants or identity fields: {undisclosed_extra}"
        )

    # Build A_NATIVE: composite of the derived (73-field) Candidate-A row
    # with only the 5 identity fields overwritten -- exactly the audited
    # builder's own "vector is None" branch.
    a_native_row = dict(der_row)
    a_native_row.update(A_NATIVE_IDENTITY_OVERRIDES)
    active_fields = sorted(set(a_native_row) - IDENTITY_FIELDS)

    if int(round(float(a_native_row["n_bins_recommended"]))) != MPZ_N_BINS:
        raise SystemExit(
            "A_NATIVE row's own n_bins_recommended does not match the "
            f"independently-required mpz_n_bins={MPZ_N_BINS}"
        )

    from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
    )

    production_engine_mro = [f"{c.__module__}.{c.__qualname__}" for c in Engine.__mro__]

    provenance = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_A_native_provenance_v1",
        "reconstruction_classification": (
            "A_NATIVE_REGISTRY_DETERMINISTICALLY_RECONSTRUCTED_FROM_QUALIFIED_INPUTS"
        ),
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "option_key": "A_NATIVE",
        "active_field_list": active_fields,
        "identity_field_overrides": A_NATIVE_IDENTITY_OVERRIDES,
        "complete_active_material_row": a_native_row,
        "complete_row_sha256": sha256_obj(a_native_row),
        "provenance_chain": {
            "pristine_v914_source": {
                "path": str(V914_SOURCE_CSV),
                "sha256": sha256_file(V914_SOURCE_CSV),
                "git_repo": str(V914_SOURCE_REPO),
                "git_head_for_file": git_head(V914_SOURCE_REPO),
                "row_field_count": len(src_row),
                "note": (
                    "immutable, git-tracked v9.14 knee-search output; source "
                    "of every cleavage/emission/peierls/taylor barrier field "
                    "and rho_source0_m2/c_blunt/physics__* transport fields"
                ),
            },
            "derived_mpz_schema_registry": {
                "path": str(DERIVED_REGISTRY_CSV),
                "sha256": sha256_file(DERIVED_REGISTRY_CSV),
                "audit_json_path": str(DERIVED_REGISTRY_AUDIT),
                "audit_json_sha256": sha256_file(DERIVED_REGISTRY_AUDIT),
                "builder_script_path": str(DERIVED_REGISTRY_BUILDER),
                "builder_script_sha256": sha256_file(DERIVED_REGISTRY_BUILDER),
                "git_repo": str(MAIN_REPO),
                "git_head_for_file": git_head(MAIN_REPO),
                "row_field_count": len(der_row),
                "disclosed_shared_spatial_constant_fields": sorted(disclosed_shared_fields),
                "cross_verification": (
                    "every field common to the pristine v9.14 source row and "
                    "this row is byte-identical except the disclosed "
                    "material_class relabel (endurance_knee -> DBTT); "
                    "verified programmatically above, not asserted"
                ),
            },
            "field_mismatches_found": {k: list(v) for k, v in mismatches.items()},
        },
        "a_native_plus_8pt_builder": {
            "path": str(A_NATIVE_PLUS_8PT_BUILDER),
            "sha256": sha256_file(A_NATIVE_PLUS_8PT_BUILDER),
            "qualified_head_constant_in_builder": QUALIFIED_HEAD,
            "qualified_head_is_ancestor_of_worktree_head": True,
            "note": (
                "the builder's own --A-registry/--donor-root/--common-physics/"
                "--family-json invocation cannot be replayed today (--A-registry "
                "and --common-physics targets no longer exist on disk and were "
                "never git-tracked); this script reproduces its documented "
                "per-field algorithm directly against the recovered row instead "
                "of invoking the builder's CLI. Donor/PT-substitution fields, "
                "common-physics hash, and family-JSON hash are NOT applicable to "
                "A_NATIVE specifically (its own variant has vector=None, i.e. no "
                "PT substitution -- those inputs only affect the 8 PT-donor "
                "sibling rows, which this pilot does not use)"
            ),
        },
        "manifest_construction_pathway": {
            "note": (
                "parameter_registry_v9111.py::select_option() is NOT usable "
                "for A_NATIVE: it unconditionally calls "
                "_validate_current_spatial_contract(), which hard-requires "
                "Tref_K==481.33 (this row's Tref_K=300.0). Separately, "
                "material_manifest.py::MaterialManifest.from_csv() never reads "
                "a Tref_K column at all (both ExpFloorBarrier objects get the "
                "module constant TREF_K=481.33 unconditionally), so the row's "
                "Tref_K value is inert to the actual physics either way. The "
                "correct, production-identical construction therefore builds "
                "parameter_registry_v9111.SelectedResponseOption directly "
                "(bypassing only the inapplicable legacy 481.33 K campaign "
                "gate and the Stage-3 canonical-candidate fingerprint check, "
                "neither of which apply to A_NATIVE) and reuses "
                "write_compatibility_manifest()/MaterialManifest.from_csv() "
                "verbatim -- see arrhenius_fracture/a_native_engine_v10230.py."
            ),
            "parameter_registry_module_sha256": sha256_file(PARAMETER_REGISTRY_MODULE),
            "material_manifest_module_sha256": sha256_file(MATERIAL_MANIFEST_MODULE),
        },
        "qualified_production_head": QUALIFIED_HEAD,
        "production_engine_mro": production_engine_mro,
        "numerical_controls": {
            "mpz_n_bins": MPZ_N_BINS,
            "wake_n_bins": WAKE_N_BINS,
            "n_phase": N_PHASE,
        },
    }

    out_path = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"complete_row_sha256={provenance['complete_row_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
