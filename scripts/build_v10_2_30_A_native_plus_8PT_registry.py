#!/usr/bin/env python3
"""Build and independently audit Candidate A plus eight PR-67 PT substitutions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


QUALIFIED_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"
PHYSICS_FILES = (
    "arrhenius_fracture/persistent_site_reversible_transport_v10230.py",
    "arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py",
    "arrhenius_fracture/persistent_site_forward_coupled_hazard_v10230.py",
    "arrhenius_fracture/persistent_site_high_cycle_state_v10230.py",
    "arrhenius_fracture/persistent_site_high_cycle_engine_v10230_v2.py",
    "arrhenius_fracture/persistent_site_high_cycle_dmd_v10230_v5.py",
    "arrhenius_fracture/sharp_front_v10_2_30_energy_gated_fatigue.py",
)
DONOR_FILES = (
    "taylor_peierls_microstructure_option_manifest.json",
    "taylor_peierls_microstructure_option_bank.csv",
    "taylor_peierls_microstructure_option_bank.parquet",
    "taylor_peierls_microstructure_option_parameters_long.csv",
    "taylor_peierls_microstructure_option_features.csv",
)
IDENTITY_FIELDS = {"option_key", "candidate_id", "role", "mechanism_summary", "validation_status"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    try:
        return format(float(value), ".17g")
    except (TypeError, ValueError):
        return str(value)


def subset_hash(row: dict, fields) -> str:
    payload = {key: canonical(row[key]) for key in sorted(fields)}
    return sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    fields = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def git_blob(repo: Path, head: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{head}:{relative}"], cwd=repo, check=True,
        stdout=subprocess.PIPE,
    ).stdout


def solver_audit(repo: Path, head: str, common: Path, family: Path) -> dict:
    files = []
    for relative in PHYSICS_FILES:
        qualified = sha(git_blob(repo, head, relative))
        study = sha((repo / relative).read_bytes())
        files.append({
            "path": relative, "qualified_sha256": qualified,
            "study_sha256": study, "comparison": "exact_equal" if qualified == study else "DIFFERS",
        })
    result = {
        "qualified_head": head,
        "files": files,
        "qualified_common_physics_sha256": sha(common.read_bytes()),
        "study_common_physics_sha256": sha(common.read_bytes()),
        "qualified_family_sha256": sha(family.read_bytes()),
        "study_family_sha256": sha(family.read_bytes()),
    }
    result["solver_physics_preserved"] = all(x["comparison"] == "exact_equal" for x in files)
    result["common_physics_preserved"] = (
        result["qualified_common_physics_sha256"] == result["study_common_physics_sha256"]
        and result["qualified_family_sha256"] == result["study_family_sha256"]
    )
    return result


def verify_composite(A: dict, composite: dict, donor: dict, pt: list[str], active: list[str]) -> dict:
    non_pt = sorted(set(active) - set(pt))
    pt_equal = all(canonical(composite[key]) == canonical(donor[key]) for key in pt)
    non_pt_equal = all(canonical(composite[key]) == canonical(A[key]) for key in non_pt)
    effective = sorted(key for key in active if canonical(composite[key]) != canonical(A[key]))
    if not pt_equal:
        raise ValueError("composite does not contain every exact donor PT value")
    if not non_pt_equal or set(effective) - set(pt):
        raise ValueError("composite changed a non-PT Candidate-A coordinate")
    return {"pt_equal": pt_equal, "non_pt_equal": non_pt_equal, "effective": effective}


def build(args) -> dict:
    before = {name: sha((args.donor_root / name).read_bytes()) for name in DONOR_FILES}
    donor_manifest = json.loads((args.donor_root / DONOR_FILES[0]).read_text())
    bank = read_csv(args.donor_root / DONOR_FILES[1])
    long_rows = read_csv(args.donor_root / DONOR_FILES[3])
    donors = donor_manifest["complete_mechanism_set_candidate_ids"]
    if len(donors) != 8 or {row["candidate_id"] for row in bank} != set(donors):
        raise ValueError("audited complete mechanism set is not exactly eight rows")
    vectors = {row["candidate_id"]: json.loads(row["full_material_vector_json"]) for row in bank}
    flags: dict[str, set[tuple[str, str]]] = {}
    for row in long_rows:
        flags.setdefault(row["parameter"], set()).add(
            (row["is_taylor_peierls_coordinate"], row["coordinate_type"])
        )
    ambiguous = {key: values for key, values in flags.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"ambiguous active donor coordinate metadata: {ambiguous}")
    unsupported = [key for key, values in flags.items() if next(iter(values))[1] != "MATERIAL_COORDINATE"]
    if unsupported:
        raise ValueError(f"unsupported active donor coordinates: {unsupported}")
    pt = sorted(key for key, values in flags.items() if next(iter(values))[0] == "True")
    non_pt_donor = sorted(set(flags) - set(pt))
    if any(set(flags) != set(vectors[key]) for key in donors):
        raise ValueError("long-form active coordinate schema does not exactly match donor vectors")

    A_matches = [row for row in read_csv(args.A_registry) if row["candidate_id"] == "v914_endurance_knee_0462"]
    if len(A_matches) != 1:
        raise ValueError("source registry must contain exactly one qualified Candidate A")
    A = A_matches[0]; fields = list(A)
    if set(pt) - set(fields):
        raise ValueError(f"fatigue registry lacks audited PT fields: {sorted(set(pt)-set(fields))}")
    active = sorted(set(fields) - IDENTITY_FIELDS)
    non_pt = sorted(set(active) - set(pt))
    cleavage = [key for key in active if key.startswith("cleave_")]
    emission = [key for key in active if key.startswith("emit_")]
    source_blunting = [key for key in non_pt if key.startswith("source_") or key in {
        "rho_source0_m2", "c_blunt", "encounter_efficiency"
    }]

    rows, evidence, diff_audit, import_audit = [], [], [], []
    variants = [("A_NATIVE", "NATIVE_A", None)] + [
        (f"A_PT_{index:02d}_{donor}", donor, vectors[donor])
        for index, donor in enumerate(donors, 1)
    ]
    for composite_id, donor_id, vector in variants:
        composite = dict(A)
        copied = []
        if vector is not None:
            for key in pt:
                composite[key] = canonical(vector[key]); copied.append(key)
        composite.update({
            "option_key": composite_id, "candidate_id": composite_id,
            "role": "Candidate-A controlled Taylor/Peierls mechanism panel",
            "mechanism_summary": "exact qualified Candidate A plus audited PT subset",
            "validation_status": "NOT_EVALUATED",
        })
        donor_pt = {key: composite[key] if vector is None else canonical(vector[key]) for key in pt}
        verification = verify_composite(A, composite, donor_pt, pt, active)
        pt_equal = verification["pt_equal"]
        non_pt_equal = verification["non_pt_equal"]
        effective = verification["effective"]
        if vector is not None and copied != pt:
            raise ValueError(f"not every PT field was actively copied for {composite_id}")
        ev = {
            "composite_candidate_id": composite_id, "parent_A_candidate_id": A["candidate_id"],
            "donor_PT_option_id": donor_id, "actively_copied_PT_fields": copied,
            "effective_changed_fields": effective,
            "PT_subset_hash": subset_hash(composite, pt),
            "donor_PT_subset_hash": subset_hash(donor_pt, pt),
            "non_PT_candidate_hash": subset_hash(composite, non_pt),
            "A_native_non_PT_candidate_hash": subset_hash(A, non_pt),
            "cleavage_hash": subset_hash(composite, cleavage),
            "emission_hash": subset_hash(composite, emission),
            "source_and_blunting_hash": subset_hash(composite, source_blunting),
            "complete_composite_material_hash": subset_hash(composite, active),
            "PT_copy_exact": pt_equal, "non_PT_preserved": non_pt_equal,
        }
        if ev["PT_subset_hash"] != ev["donor_PT_subset_hash"]:
            raise ValueError(f"donor PT subset hash mismatch for {composite_id}")
        evidence.append(ev); rows.append(composite)
        diff_audit.append({
            "composite_candidate_id": composite_id,
            "actively_copied_PT_fields_json": json.dumps(copied),
            "effective_changed_fields_json": json.dumps(effective),
            "outside_PT_changed_fields_json": json.dumps(sorted(set(effective)-set(pt))),
            "PT_copy_exact": pt_equal, "non_PT_preserved": non_pt_equal,
        })
        if vector is not None:
            for key in sorted(vectors[donor_id]):
                imported = key in pt
                import_audit.append({
                    "donor_PT_option_id": donor_id, "source_field": key,
                    "classification": "PT_TRANSFERRED_MATERIAL_COORDINATE" if imported else "NON_PT_DONOR_MATERIAL_COORDINATE_NOT_TRANSFERRED",
                    "donor_value": canonical(vector[key]), "imported": imported,
                    "destination_field": key if imported else "",
                    "reason": "audited is_taylor_peierls_coordinate flag" if imported else "Candidate A remains authoritative outside PT subset",
                })
            for key in sorted(set(bank[0]) - {"full_material_vector_json"}):
                import_audit.append({
                    "donor_PT_option_id": donor_id, "source_field": key,
                    "classification": "DONOR_METADATA", "donor_value": next(x for x in bank if x["candidate_id"] == donor_id)[key],
                    "imported": False, "destination_field": "", "reason": "provenance/diagnostic metadata only",
                })

    if len({x["non_PT_candidate_hash"] for x in evidence}) != 1:
        raise ValueError("non-PT hashes differ")
    for subset in ("cleavage_hash", "emission_hash", "source_and_blunting_hash"):
        if len({x[subset] for x in evidence}) != 1:
            raise ValueError(f"{subset} differs")
    if any(x["imported"] is True and x["classification"] != "PT_TRANSFERRED_MATERIAL_COORDINATE" for x in import_audit):
        raise ValueError("prohibited donor field imported")
    after = {name: sha((args.donor_root / name).read_bytes()) for name in DONOR_FILES}
    if before != after:
        raise ValueError("audited PR-67 source artifacts changed during composition")
    solver = solver_audit(args.repo, args.qualified_head, args.common_physics, args.family_json)
    if not solver["solver_physics_preserved"] or not solver["common_physics_preserved"]:
        raise ValueError("qualified solver/common physics hash preflight failed")

    args.out.mkdir(parents=True, exist_ok=True)
    registry = args.out / "A_native_plus_8PT_registry.csv"
    write_csv(registry, rows, fields)
    import pandas as pd
    pd.DataFrame(rows).to_parquet(args.out / "A_native_plus_8PT_registry.parquet", index=False)
    write_csv(args.out / "A_native_plus_8PT_parameter_diff_audit.csv", diff_audit)
    write_csv(args.out / "donor_field_import_audit.csv", import_audit)
    contract = []
    for key in sorted(flags):
        transferred = key in pt
        contract.append({
            "source_field": key,
            "classification": "PT_TRANSFERRED_MATERIAL_COORDINATE" if transferred else "NON_PT_DONOR_MATERIAL_COORDINATE_NOT_TRANSFERRED",
            "imported_to_fatigue": transferred,
            "reason": "audited machine-readable flag" if transferred else "outside audited PT subset",
        })
    (args.out / "pt_substitution_field_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True)+"\n")
    selection = {"schema": "v10.2.30_A_native_plus_8PT_selection_v1", "canonical_option_order": [x["option_key"] for x in rows], "installed_registry_sha256": sha(registry.read_bytes()), "numerical_bins": int(A["n_bins_recommended"])}
    (args.out / "A_native_plus_8PT_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True)+"\n")
    result = {
        "schema": "v10.2.30_A_native_plus_8PT_provenance_v2",
        "variant_count": 9, "PT_SUBSTITUTION_FIELDS": pt,
        "qualified_A_registry": str(args.A_registry.resolve()),
        "qualified_A_registry_sha256": sha(args.A_registry.read_bytes()),
        "audited_fracture_HEAD": "2df158bf8d1484f40898c64a11fc76fdd327178c",
        "donor_artifact_before_sha256": before, "donor_artifact_after_sha256": after,
        "donor_artifacts_preserved": before == after,
        "solver_preflight": solver, "variants": evidence,
        "material_promotion_status": "NOT_PROMOTED_DIAGNOSTIC_ONLY",
    }
    (args.out / "A_native_plus_8PT_provenance_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A-registry", required=True, type=Path)
    parser.add_argument("--donor-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--qualified-head", default=QUALIFIED_HEAD)
    parser.add_argument("--common-physics", required=True, type=Path)
    parser.add_argument("--family-json", required=True, type=Path)
    args = parser.parse_args(argv)
    for name in DONOR_FILES:
        if not (args.donor_root / name).is_file():
            raise SystemExit(f"missing audited donor artifact: {name}")
    try:
        result = build(args)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
