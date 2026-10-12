"""Machine-readable design for the prospective common-framework campaign.

The design is intentionally declarative.  Building it never executes a solver.
Every case carries a capability gate so a production launcher cannot silently
run a scientifically unresolved configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TEMPERATURES_K = (300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300)
RATE_FACTORS = (0.01, 1.0, 100.0)
BASE_DU_M = 2.0e-7
BASE_DT_S = 8.4
REGISTRY = "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"


@dataclass(frozen=True)
class MaterialRow:
    family: str
    option_key: str
    candidate_id: str
    Tref_K: float
    cleave_G00_eV: float
    cleave_gT_eV_per_K: float
    cleave_sigc0_GPa: float
    cleave_sT_GPa_per_K: float
    emit_G00_eV: float
    emit_gT_eV_per_K: float
    emit_sigc0_GPa: float
    emit_sT_GPa_per_K: float

    def domain(self, temperature_K: float) -> dict[str, Any]:
        dT = float(temperature_K) - self.Tref_K
        values = {
            "cleave_G0_eV_raw": self.cleave_G00_eV + self.cleave_gT_eV_per_K * dT,
            "cleave_sigc_GPa_raw": self.cleave_sigc0_GPa + self.cleave_sT_GPa_per_K * dT,
            "emit_G0_eV_raw": self.emit_G00_eV + self.emit_gT_eV_per_K * dT,
            "emit_sigc_GPa_raw": self.emit_sigc0_GPa + self.emit_sT_GPa_per_K * dT,
        }
        positive = all(value > 0.0 for value in values.values())
        min_fraction = min(
            values["cleave_G0_eV_raw"] / self.cleave_G00_eV,
            values["cleave_sigc_GPa_raw"] / self.cleave_sigc0_GPa,
            values["emit_G0_eV_raw"] / self.emit_G00_eV,
            values["emit_sigc_GPa_raw"] / self.emit_sigc0_GPa,
        )
        if not positive:
            status = "BLOCKED_RAW_CONSTITUTIVE_SCALE_NONPOSITIVE"
        elif min_fraction < 0.05:
            status = "QUALIFICATION_REQUIRED_NEAR_SCALE_ZERO"
        else:
            status = "ELIGIBLE"
        return {**values, "minimum_raw_scale_fraction": min_fraction, "domain_status": status}


@dataclass(frozen=True)
class CampaignCase:
    case_id: str
    block: str
    backend: str
    study: str
    family: str | None
    parameter_option: str | None
    temperature_K: float | None
    rate_factor: float | None
    arm: str
    seed: int
    orientation_deg: float
    initial_crack_length_m: float
    target_extension_um: float
    endpoint: str
    source_location: str
    source_lifecycle: str
    plasticity_scope: str
    crack_path: str
    cavity_mode: str
    loading_mode: str
    output_profile: str
    eligibility: str
    gate: str
    native_entrypoint: str
    command_status: str
    native_args: tuple[str, ...]
    estimand: str
    storage_estimate_GiB: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["native_args"] = list(self.native_args)
        return payload


def load_material_rows(root: str | Path) -> dict[str, MaterialRow]:
    path = Path(root) / REGISTRY
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item = MaterialRow(
                family=row["material_class"], option_key=row["option_key"],
                candidate_id=row["candidate_id"], Tref_K=float(row["Tref_K"]),
                cleave_G00_eV=float(row["cleave_G00_eV"]),
                cleave_gT_eV_per_K=float(row["cleave_gT_eV_per_K"]),
                cleave_sigc0_GPa=float(row["cleave_sigc0_GPa"]),
                cleave_sT_GPa_per_K=float(row["cleave_sT_GPa_per_K"]),
                emit_G00_eV=float(row["emit_G00_eV"]),
                emit_gT_eV_per_K=float(row["emit_gT_eV_per_K"]),
                emit_sigc0_GPa=float(row["emit_sigc0_GPa"]),
                emit_sT_GPa_per_K=float(row["emit_sT_GPa_per_K"]),
            )
            result[item.family] = item
    if set(result) != {"peak", "DBTT", "weakT", "ceramic"}:
        raise RuntimeError(f"unexpected four-family registry: {sorted(result)}")
    return result


def _case_id(block: str, *parts: object) -> str:
    raw = "_".join(str(part).replace(".", "p").replace("-", "m") for part in parts)
    return f"{block}_{raw}"


def _base_args(row: MaterialRow, T: float, rate: float, arm: str, *,
               target_um: float, theta: float = 0.0, a0_m: float = 0.5e-3) -> tuple[str, ...]:
    args = [
        "--mode", "2d", "--parameter-registry", REGISTRY,
        "--parameter-option", row.option_key, "--temperatures", f"{T:g}",
        "--steps", "200000" if target_um >= 1000 else "1000",
        "--nx", "36", "--ny", "72", "--dU", f"{BASE_DU_M:.17g}",
        "--dt", f"{BASE_DT_S / rate:.17g}", "--n-stagger", "2",
        "--tip-h-fine", "1e-6", "--tip-ratio", "1.20", "--da-phys", "5e-6",
        "--target-crack-extension-um", f"{target_um:g}",
        "--initial-crack-length-um", f"{a0_m * 1e6:g}",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding", "--crystal-aniso",
        "--crystal-compete", "--crystal-theta-deg", f"{theta:g}",
        "--crystal-material", "w", "--j-decomposition", "cluster", "--max-fronts", "1",
        "--crack-backend", "sharp_wake", "--adaptive-events",
        "--adaptive-event-target", "0.15", "--print-every", "200", "--no-mpz-state-output",
    ]
    args += ["--tip-plasticity", "--active-shielding"] if arm == "accommodation" else ["--no-tip-plasticity", "--no-active-shielding"]
    return tuple(args)


def _pf_case(block: str, study: str, row: MaterialRow, T: float, rate: float,
             arm: str, seed: int = 1733, theta: float = 0.0, a0_m: float = 0.5e-3,
             target_um: float = 10.0, *, gate: str = "framework_release",
             source_location: str = "tip_surface", crack_path: str = "single_front",
             cavity: str = "disabled", estimand: str = "first-event initiation") -> CampaignCase:
    domain = row.domain(T)
    eligibility = domain["domain_status"]
    requested_gate = gate
    command_status = (
        "RESOLVED" if requested_gate == "framework_release" and eligibility == "ELIGIBLE"
        else "RESOLVED_BLOCKED_PHYSICAL_DOMAIN" if requested_gate == "framework_release"
        else "UNRESOLVED_ADAPTER_GATE"
    )
    return CampaignCase(
        case_id=_case_id(block, study, row.family, f"T{T:g}", f"r{rate:g}", arm,
                         f"s{seed}", f"th{theta:g}", f"a{a0_m*1e6:g}",
                         source_location, crack_path, cavity),
        block=block, backend="spatial_pf", study=study, family=row.family,
        parameter_option=row.option_key, temperature_K=float(T), rate_factor=float(rate),
        arm=arm, seed=seed, orientation_deg=float(theta), initial_crack_length_m=a0_m,
        target_extension_um=target_um, endpoint=("developed_growth_or_physical_censor" if target_um >= 1000 else "first_event_or_opening_censor"),
        source_location=source_location, source_lifecycle="persistent_reusable",
        plasticity_scope=("tip_process_only" if source_location == "tip_surface" else ("full_field_evolving_rho" if source_location == "bulk_uniform" else "tip_plus_full_field")),
        crack_path=crack_path, cavity_mode=cavity, loading_mode="monotonic",
        output_profile="publication_restart" if target_um >= 1000 or crack_path != "single_front" or cavity != "disabled" else "selected_fields",
        eligibility=eligibility, gate=gate if eligibility == "ELIGIBLE" else eligibility,
        native_entrypoint="arrhenius_fracture.sharp_front_v10_2_27_audited",
        command_status=command_status,
        native_args=(_base_args(row, T, rate, arm, target_um=target_um, theta=theta, a0_m=a0_m)
                     if requested_gate == "framework_release" else ()),
        estimand=estimand, storage_estimate_GiB=0.35 if target_um >= 1000 else 0.025,
    )


def build_cases(root: str | Path) -> list[CampaignCase]:
    rows = load_material_rows(root)
    cases: list[CampaignCase] = []

    # P1: exact 288 nominal entries; invalid-domain cells remain explicit and blocked.
    for row in rows.values():
        for T in TEMPERATURES_K:
            for rate in RATE_FACTORS:
                for arm in ("accommodation", "opening_only"):
                    cases.append(_pf_case("P1", "temperature_rate", row, T, rate, arm))

    # P2a: 24 nominal long-growth paths.
    anchors = {"peak": (600, 900, 1100), "DBTT": (600, 950, 1000),
               "weakT": (300, 800, 1200), "ceramic": (300, 950, 1300)}
    for family, temps in anchors.items():
        for T in temps:
            for arm in ("accommodation", "opening_only"):
                cases.append(_pf_case("P2", "developed_resistance", rows[family], T, 1.0, arm, target_um=1000.0,
                                      estimand="event-conditioned resistance versus continuously evolved extension"))
    # P2b: rate extremes in two contrasting regimes (8 paths).
    for family, T in (("DBTT", 950), ("weakT", 800)):
        for rate in (0.01, 100.0):
            for arm in ("accommodation", "opening_only"):
                cases.append(_pf_case("P2", "propagation_rate", rows[family], T, rate, arm, target_um=1000.0,
                                      estimand="propagation-rate interaction"))
    # P2c: exact 36 initial-size initiation cases.
    for family, temps in (("DBTT", (600, 950, 1000)), ("weakT", (300, 800, 1200))):
        for T in temps:
            for a0_ratio in (0.125, 0.25, 0.375):
                for arm in ("accommodation", "opening_only"):
                    cases.append(_pf_case("P2", "initial_crack_size", rows[family], T, 1.0, arm,
                                          a0_m=2e-3*a0_ratio, estimand="initiation versus independently prepared initial crack length"))

    # P3 diagnostic interventions are frozen as gated full-model cases; their
    # executor must be qualified without adding feedback channels together.
    for family, T in (("DBTT", 950), ("weakT", 800), ("peak", 900)):
        for intervention in ("native", "retained_shielding_off", "source_backstress_off",
                             "blunting_fixed", "transport_retention_off", "history_reset_reequilibrated"):
            case = _pf_case("P3", intervention, rows[family], T, 1.0, "accommodation",
                            gate="diagnostic_intervention_adapter_qualification",
                            estimand=f"full-model controlled intervention: {intervention}")
            cases.append(case)

    # P4 orientation and source geography.
    for family, temps in (("DBTT", (600, 950, 1000)), ("weakT", (300, 800, 1200))):
        for T in temps:
            for theta in (0.0, 30.0, 45.0):
                for arm in ("accommodation", "opening_only"):
                    cases.append(_pf_case("P4", "orientation", rows[family], T, 1.0, arm, theta=theta,
                                          gate="orientation_kernel_qualified"))
    for family, T in (("DBTT", 950), ("weakT", 800)):
        for location in ("tip_surface", "bulk_uniform", "tip_plus_bulk"):
            cases.append(_pf_case("P4", "source_support", rows[family], T, 1.0, "accommodation",
                                  source_location=location, gate="source_support_executor_qualified",
                                  estimand="source-support intervention with declared opportunity"))
        cases.append(_pf_case("P4", "source_support_control", rows[family], T, 1.0, "opening_only"))

    # P5: six anchors, eight paired seeds, two arms = 96 entries.
    for family, T in (("DBTT", 950), ("DBTT", 1000), ("peak", 800),
                      ("peak", 900), ("weakT", 800), ("ceramic", 950)):
        for seed in (31415, 27182, 16180, 14142, 17320, 22360, 24495, 31622):
            for arm in ("accommodation", "opening_only"):
                cases.append(_pf_case("P5", "paired_seed", rows[family], T, 1.0, arm, seed=seed,
                                      estimand="paired arm difference; no tail-confidence claim"))

    # P6: physical branch birth versus no-bifurcation controls.
    for T in (800, 900):
        for rate in (1.0, 10.0):
            for theta in (0.0, 45.0):
                for seed in (31415, 27182, 16180, 14142):
                    for branch in (False, True):
                        cases.append(_pf_case("P6", "branch_birth" if branch else "no_bifurcation_control",
                                              rows["peak"], T, rate, "accommodation", seed=seed, theta=theta,
                                              target_um=250.0, crack_path="branching_multifront" if branch else "single_front",
                                              gate="recursive_branch_birth_and_daughter_qualification" if branch else "framework_release",
                                              estimand="branch birth probability/path and daughter continuation"))

    # P7: paired nucleation/growth plus conditional pre-existing-cavity geometry.
    for T in (800, 1000):
        for seed in (31415, 27182, 16180, 14142):
            for cavity in ("disabled", "nucleation_growth"):
                cases.append(_pf_case("P7", "native_void_nucleation", rows["DBTT"], T, 1.0, "accommodation", seed=seed,
                                      target_um=250.0, cavity=cavity,
                                      gate="native_void_nucleation_common_ancestry_qualified",
                                      estimand="paired vacancy nucleation/growth from common pre-divergence state"))
    for T in (800, 1000):
        for separation_pz in (0.5, 1.0, 2.0):
            for radius_pz in (0.10, 0.25):
                for cavity in ("disabled", "preexisting_equilibrated"):
                    case = _pf_case("P7", f"conditional_void_d{separation_pz:g}_r{radius_pz:g}", rows["DBTT"], T, 1.0,
                                    "accommodation", target_um=500.0, cavity=cavity,
                                    gate="candidate_rate_map_author_decision",
                                    estimand="conditional crack-void interaction using common physical candidate map")
                    cases.append(case)
    for rate in (0.1, 10.0):
        for cavity in ("disabled", "preexisting_equilibrated"):
            cases.append(_pf_case("P7", "conditional_void_rate", rows["DBTT"], 800, rate, "accommodation",
                                  target_um=500.0, cavity=cavity, gate="candidate_rate_map_author_decision",
                                  estimand="rate sensitivity of conditional interaction"))

    # P8: independent FEM/CZM set. Native arguments are resolved later by its
    # pinned adapter; these entries cannot silently fall back to PF.
    for family, temps in (("DBTT", (600, 950, 1000)), ("peak", (600, 900, 1100))):
        for T in temps:
            for arm in ("accommodation", "opening_only"):
                base = _pf_case("P8", "fem_czm_common", rows[family], T, 1.0, arm,
                                gate="production_mesh_support_convergence",
                                estimand="common-load PF/FEM-CZM mechanics and initiation")
                cases.append(CampaignCase(**{**asdict(base), "backend": "full_fem_czm",
                                             "native_entrypoint": "arrhenius_fracture.mode_i_first_passage_v10_0_5_18_four_class_stochastic_emission",
                                             "storage_estimate_GiB": 0.06}))
    for family, T in (("DBTT", 950), ("peak", 900)):
        for rate in (0.01, 100.0):
            for arm in ("accommodation", "opening_only"):
                base = _pf_case("P8", "fem_czm_rate", rows[family], T, rate, arm,
                                gate="production_mesh_support_convergence",
                                estimand="FEM/CZM rate comparison")
                cases.append(CampaignCase(**{**asdict(base), "backend": "full_fem_czm",
                                             "native_entrypoint": "arrhenius_fracture.mode_i_first_passage_v10_0_5_18_four_class_stochastic_emission",
                                             "storage_estimate_GiB": 0.06}))
    for family, T in (("DBTT", 950), ("peak", 900)):
        base = _pf_case("P8", "fem_czm_developed", rows[family], T, 1.0, "accommodation", target_um=1000.0,
                        gate="production_mesh_support_convergence",
                        estimand="developed native cohesive growth")
        cases.append(CampaignCase(**{**asdict(base), "backend": "full_fem_czm",
                                     "native_entrypoint": "arrhenius_fracture.mode_i_first_passage_v10_0_5_18_four_class_stochastic_emission",
                                     "storage_estimate_GiB": 0.6}))

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise RuntimeError(f"duplicate campaign case ids: {duplicates[:5]}")
    return cases


def campaign_summary(cases: Iterable[CampaignCase]) -> dict[str, Any]:
    rows = list(cases)
    by_block: dict[str, int] = {}
    by_gate: dict[str, int] = {}
    for case in rows:
        by_block[case.block] = by_block.get(case.block, 0) + 1
        by_gate[case.gate] = by_gate.get(case.gate, 0) + 1
    serialized = json.dumps([case.to_dict() for case in rows], sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "prospective-common-fracture-campaign/1",
        "case_count": len(rows), "by_block": by_block, "by_gate": by_gate,
        "estimated_storage_GiB": sum(case.storage_estimate_GiB for case in rows),
        "matrix_sha256": hashlib.sha256(serialized).hexdigest(),
        "production_execution_authorized": False,
    }


__all__ = ["CampaignCase", "MaterialRow", "TEMPERATURES_K", "RATE_FACTORS",
           "build_cases", "campaign_summary", "load_material_rows"]
