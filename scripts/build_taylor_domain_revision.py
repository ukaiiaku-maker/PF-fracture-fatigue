#!/usr/bin/env python3
"""Build versioned Taylor-amplitude candidates and continuous-domain evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K, MaterialManifest
from arrhenius_fracture.parameter_registry_v9111 import (
    select_option, write_compatibility_manifest,
)
from arrhenius_fracture.taylor_domain_revision import (
    DEFAULT_MARGINS_EV, T_MAX_K, T_MIN_K, assert_amplitude_only,
    construct_amplitude_rows, minimal_intercept_stress_candidate, read_rows,
    sha256_file,
)
from arrhenius_fracture.taylor_factorization import evaluate_same_state, raw_scales

PARENT_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
PARENT_SELECTION = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_selection.json"
APPROVED_ALTERNATIVE_REGISTRY = ROOT / "arrhenius_fracture/data/materials/MPZ_v9_11_1_parameter_registry.csv"
APPROVED_ALTERNATIVE_KEY = "dbtt_primary"
TEMPERATURES_K = (300.0, 800.0, 1000.0, 1100.0, 1300.0)
LOCAL_STRESSES_PA = (0.0, 0.5e9, 2.0e9, 6.0e9)
DENSITY_RATIOS = (0.01, 1.0, 100.0)
RATE_CONTOURS_S = (1.0, 1.0e3, 1.0e6, 1.0e9)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _inverse_elementary_contour(surface, T_K: float, rate_s: float) -> dict[str, object]:
    diagnostic = surface.diagnostic(0.0, T_K)
    G0 = float(diagnostic["executed_G0_eV"])
    floor = float(diagnostic["floor_eV"])
    sigc = float(diagnostic["executed_sigc_Pa"])
    target = -KB_EV_PER_K * T_K * math.log(rate_s / surface.attempt_frequency_s)
    if not floor < target < G0 or surface.alpha <= 0.0 or surface.exponent <= 0.0:
        return {"rate_s": rate_s, "target_barrier_eV": target, "status": "NO_FINITE_CONTOUR"}
    ratio = (target - floor) / (G0 - floor)
    stress = sigc * ((-math.log(ratio) / surface.alpha) ** (1.0 / surface.exponent))
    return {
        "rate_s": rate_s, "target_barrier_eV": target,
        "activated_resolved_stress_Pa": stress, "status": "FINITE_CONTOUR",
        "stress_definition": "Taylor-surface argument after projection/amplification, not applied specimen stress",
    }


def _native_dossier(registry: Path, records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {str(item["variant_option_key"]): item for item in records}
    result: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="taylor-domain-native-") as scratch:
        for key, construction in by_key.items():
            selected = select_option(key, registry)
            manifest_path = Path(scratch) / f"{hashlib.sha256(key.encode()).hexdigest()}.csv"
            write_compatibility_manifest(selected, manifest_path)
            manifest = MaterialManifest.from_csv(manifest_path)
            row = selected.row
            surface = manifest.taylor.as_surface(manifest.emission)
            temperatures = []
            for T_K in TEMPERATURES_K:
                raw = raw_scales(row, T_K)
                valid = all(value > 0.0 for value in raw.values())
                item: dict[str, object] = {
                    "temperature_K": T_K, "raw_scales": raw,
                    "status": "ADMISSIBLE" if valid else "UNSUPPORTED_RAW_SCALE",
                }
                if valid:
                    item["zero_stress_native_diagnostic"] = surface.diagnostic(0.0, T_K)
                    item["inverse_elementary_rate_contours"] = [
                        _inverse_elementary_contour(surface, T_K, rate)
                        for rate in RATE_CONTOURS_S
                    ]
                    states = []
                    for stress in LOCAL_STRESSES_PA:
                        for ratio in DENSITY_RATIOS:
                            states.append(evaluate_same_state(
                                row, temperature_K=T_K,
                                signed_local_stress_Pa=stress,
                                raw_density_m2=ratio * float(row["taylor_corr_rho_c_m2"]),
                            ))
                    item["native_finite_state_surface"] = states
                temperatures.append(item)
            result.append({
                "construction": construction,
                "constructor_receipt": {
                    "selected_option_key": selected.option_key,
                    "registry_sha256": selected.registry_sha256,
                    "compatibility_manifest_sha256": sha256_file(manifest_path),
                    "material_candidate_id": manifest.candidate_id,
                    "taylor_surface": surface.__dict__,
                },
                "temperatures": temperatures,
            })
    return result


def build(output: Path, margins: tuple[float, ...]) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite Taylor domain revision: {output}")
    output.mkdir(parents=True)
    fields, parents = read_rows(PARENT_REGISTRY)
    selection = json.loads(PARENT_SELECTION.read_text())
    expected = selection["canonical_option_order"]
    if [row["option_key"] for row in parents] != expected:
        raise RuntimeError("parent registry order does not match selected four-family order")
    variants, records = construct_amplitude_rows(parents, margins)
    for parent in parents:
        matches = [row for row in variants if row["option_key"].startswith(parent["option_key"] + "__")]
        for variant in matches:
            assert_amplitude_only(parent, variant)

    registry = output / "v10_2_31_taylor_domain_registry.csv"
    _write_csv(registry, fields, variants)
    variant_selection = {
        "schema": "v10.2.31-taylor-domain-selection/1",
        "status": "PROSPECTIVE_CONSTRUCTED_NOT_MATERIAL_CALIBRATED",
        "source_head": _git_head(),
        "declared_continuous_temperature_domain_K": [T_MIN_K, T_MAX_K],
        "parent_registry": str(PARENT_REGISTRY.relative_to(ROOT)),
        "parent_registry_sha256": sha256_file(PARENT_REGISTRY),
        "parent_selection": str(PARENT_SELECTION.relative_to(ROOT)),
        "parent_selection_sha256": sha256_file(PARENT_SELECTION),
        "registry": registry.name,
        "registry_sha256": sha256_file(registry),
        "margins_eV": list(margins),
        "primary_readiness_margin_eV": 0.10,
        "variant_option_order": [row["option_key"] for row in variants],
        "production_selection": None,
        "checkpoint_policy": "old checkpoints require an explicitly labeled state-transfer diagnostic; they are not resumable as current-parameter physics",
    }
    selection_path = output / "v10_2_31_taylor_domain_selection.json"
    selection_path.write_text(json.dumps(variant_selection, indent=2, sort_keys=True) + "\n")

    invalid_parents = [row for row in parents if min(
        float(row["emit_sigc0_GPa"]) + float(row["emit_sT_GPa_per_K"]) * (T - float(row["Tref_K"]))
        for T in (T_MIN_K, T_MAX_K)
    ) <= 0.0]
    minimal = [minimal_intercept_stress_candidate(row) for row in invalid_parents]
    approved = select_option(APPROVED_ALTERNATIVE_KEY, APPROVED_ALTERNATIVE_REGISTRY)
    stress_payload = {
        "schema": "v10.2.31-shared-stress-scale-candidates/1",
        "status": "EXPLICIT_PROSPECTIVE_ALTERNATIVES_NOT_SELECTED",
        "domain_K": [T_MIN_K, T_MAX_K],
        "invalid_selected_parent_count": len(invalid_parents),
        "minimal_intercept_candidates": minimal,
        "existing_approved_alternative": {
            "option_key": approved.option_key,
            "candidate_id": approved.candidate_id,
            "source_registry": str(APPROVED_ALTERNATIVE_REGISTRY.relative_to(ROOT)),
            "source_registry_sha256": approved.registry_sha256,
            "approval_record": approved.validation_status,
            "row": approved.row,
            "substitution_status": "not substituted; distinct material parameterization",
        },
        "production_selection": None,
    }
    stress_path = output / "v10_2_31_shared_stress_scale_candidates.json"
    stress_path.write_text(json.dumps(stress_payload, indent=2, sort_keys=True) + "\n")

    dossier = {
        "schema": "v10.2.31-taylor-domain-dossier/1",
        "status": "PASS_CONSTRUCTED_DOMAIN_AND_NATIVE_LAW",
        "scope": "analytical continuous-domain construction plus native constructor and same-state law evaluation; no calibration or specimen outcome",
        "derivative_conventions": [
            "fixed physical stress and frozen external state",
            "fixed normalized stress reported separately by native diagnostic",
        ],
        "records": _native_dossier(registry, records),
    }
    dossier_path = output / "v10_2_31_taylor_domain_dossier.json"
    dossier_path.write_text(json.dumps(dossier, indent=2, sort_keys=True) + "\n")
    manifest = {path.name: sha256_file(path) for path in sorted(output.iterdir())}
    (output / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": dossier["status"], "variants": len(variants),
        "invalid_shared_stress_parents": len(invalid_parents),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins-eV", nargs="+", type=float, default=DEFAULT_MARGINS_EV)
    args = parser.parse_args()
    build(args.output.resolve(), tuple(args.margins_eV))


if __name__ == "__main__":
    main()
