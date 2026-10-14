#!/usr/bin/env python3
"""Build a frozen law-to-result map without rewriting historical outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review_packet/publication_acceptance_0c72ca2b/METHOD_TO_RESULT_MAP.json"
OUT = ROOT / "common_framework_readiness/LAW_TO_RESULT_MAP_V3.json"
HISTORICAL_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"


FIGURES = {
    "baseline_and_reused_anchor": ["Figure_2_PF_baseline_96"],
    "focused_seeds": ["Figure_3_PF_focused_gaps"],
    "rate": ["Figure_4_sensitivities"],
    "orientation": ["Figure_4_sensitivities"],
    "bulk_uniform": ["Figure_4_sensitivities", "Figure_6_PF_FEM"],
    "tip_plus_bulk": ["Figure_4_sensitivities", "Figure_6_PF_FEM"],
    "developed_growth": ["Figure_5_developed_physical_history", "Figure_7_native_fields"],
    "cavity_current_source": ["Figure_8_cavity"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arg(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


def build() -> dict:
    source = json.loads(SOURCE.read_text())
    with HISTORICAL_REGISTRY.open(newline="") as stream:
        registry = {row["option_key"]: row for row in csv.DictReader(stream)}
    families = []
    for item in source["families"]:
        argv = item.get("native_argv", [])
        option = _arg(argv, "--parameter-option")
        material = registry.get(option)
        cavity = item["family"] == "cavity_current_source"
        families.append({
            "family": item["family"], "figures": FIGURES[item["family"]],
            "result_root": item["case"], "producer_commit": item["producer_commit"],
            "frozen_status": "HISTORICAL_RETAINED_NOT_CURRENT_LAW_PREDICTION",
            "completion_mapping": (
                "current-source V5 cavity operator; Taylor completion mapping not shared with persistent campaign"
                if cavity else
                "producer-bound historical persistent-site operator; predates the qualified current mean-net-rate-over-density-order repair"
            ),
            "density_order": (
                None if material is None else {
                    "correlation_density_m2": float(material["taylor_corr_rho_c_m2"]),
                    "scale": float(material["taylor_corr_scale"]),
                    "current_formula_not_back_applied": "1 + scale*sqrt(rho_f/rho_c)",
                }
            ),
            "phi_policy": (
                "V5 current-source local tensor ownership"
                if cavity else {
                    "peierls_stress_fraction": float(material["peierls_stress_fraction"]),
                    "taylor_stress_fraction": float(material["taylor_stress_fraction"]),
                    "mobile_shield_fraction_cli": _arg(argv, "--mobile-shield-fraction"),
                }
            ),
            "raw_domain_and_clamps": (
                "historical producer-bound parameter domain; current v10.2.32 Taylor-domain revisions are not retroactively applied"
            ),
            "source_lifecycle": item["source_lifecycle"],
            "source_location": item["source_location"],
            "operator": {
                "integration": item["integration"],
                "lifecycle_module": item["lifecycle_module"],
                "lifecycle_attribute": item["lifecycle_attribute"],
            },
            "mechanical_backend": (
                "original spatial PF/sharp-front FEM" if not cavity
                else "original spatial PF plus V5 current-source cavity/topology executor"
            ),
            "input_identity": {
                "parameter_registry": _arg(argv, "--parameter-registry"),
                "parameter_option": option,
                "temperature_K": _arg(argv, "--temperatures"),
                "registry_sha256": sha256(HISTORICAL_REGISTRY) if material else None,
                "native_argv": argv,
            },
        })
    return {
        "schema": "release-law-to-result-map/3",
        "frozen_status": (
            "All listed publication results are historical retained calculations. "
            "They are not predictions of the v10.2.32 Taylor amplitudes, corrected "
            "completion operator, fitted bilinear laws, or unresolved shared stress choice."
        ),
        "prohibited_inference": (
            "The retained 156%, 164%, and 32-35 MPa sqrt(m) differences must not be "
            "used as quantitative predictions of the prospective model."
        ),
        "source_map": str(SOURCE.relative_to(ROOT)),
        "source_map_sha256": sha256(SOURCE),
        "historical_registry": str(HISTORICAL_REGISTRY.relative_to(ROOT)),
        "historical_registry_sha256": sha256(HISTORICAL_REGISTRY),
        "families": families,
        "current_readiness_outputs": [
            {
                "identity": "v10.2.32_taylor_domain",
                "result": "arrhenius_fracture/data/materials/v10_2_32_taylor_domain/v10_2_32_taylor_domain_dossier.json",
                "scope": "parameter construction, exact forward/inverse contours and native net-completion diagnostics; not specimen results",
            },
            {
                "identity": "v10.2.32_bilinear_opening",
                "result": "arrhenius_fracture/data/materials/v10_2_32_bilinear_opening/v10_2_32_bilinear_opening_fit_registry.json",
                "scope": "finite-domain fit; absolute cohesive physical support remains unassigned",
            },
            {
                "identity": "current_R11_analysis",
                "result": "common_framework_readiness/LONG_GROWTH_ANALYSIS_QUALIFICATION.json",
                "scope": "analysis regression over historical native histories; does not relabel them as current-law trajectories",
            },
        ],
    }


def main() -> int:
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
