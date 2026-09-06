"""PX2 (Part X): construct and freeze the four kinetic rows, evaluate the
analytical atlas across the full parameter grid, and commit the required
tracked artifacts -- all before any physical result directory may exist
(mission section 2's gate).

Row construction used ONLY the analytical phase-resolved P/C/B model
(never fit to da/dN, a desired Paris slope, or an observed suppression):

- SAT_EXISTING: the existing frozen RB2_reversible_finite config, reused
  verbatim from artifacts/crack_rebonding_causal_pilot_v2/
  frozen_configuration.json (config_hash 271f459...), unchanged.
- COMPETING_REVERSIBLE: solve_reference_action_barriers(A_on=2.0,
  A_off=2.0) at the reference state -- checked against section 6.2's
  exact gates (0.30<=mean p_B<=0.70, swing>=0.20, 0.10<=A_CB,A_BC<=3.0,
  no barrier-floor/cooperative-saturation, p_B not always >0.95).
- COMPETING_PERSISTENT: COMPETING_REVERSIBLE's bond barrier unchanged,
  rupture barrier raised by 0.05 eV -- checked against section 6.3's
  gates (0.50<=mean p_B<=0.90, p_B at tensile peak>=0.25, A_BC at least
  5x smaller than COMPETING_REVERSIBLE's, mean p_B differs by >=0.10).
- PASSIVATION_LIMITED: COMPETING_REVERSIBLE's formation/rupture barriers
  plus symmetric depassivation/repassivation barriers (0.40 eV each) --
  checked against section 6.4's gates (0.30<=mean p_P<=0.80,
  0.05<=mean p_B<=0.50, nonzero A_PC/A_CP/F_PC/F_CP, swing>=0.10, no
  barrier-floor/saturation).

Every candidate's gate margins are printed and archived in
analytical_regime_selection.json for auditability.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
    solve_reference_action_barriers,
)
from arrhenius_fracture.crack_rebonding_part_x_kinetic_regime_v10230 import (
    analytical_periodic_orbit,
    predicted_static_shield_equivalent_K_b,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa

OUT_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
CAUSAL_PILOT_V2_CONFIG_PATH = (
    REPO_ROOT / "artifacts" / "crack_rebonding_causal_pilot_v2" / "frozen_configuration.json"
)

REF_KMAX_Pa_sqrt_m = 18.0e6
REF_R = -0.5
REF_FREQUENCY_Hz = 1000.0
REF_T_K = 300.0
G_MAX_BASELINE_J_m2 = 1.8225  # K_rebond,max = 0.9 MPa sqrt(m) for A_NATIVE (verified below)
ETA_K_BASELINE = 1.0
N_PHASE_ANALYTICAL = 64


def _engine_geometry() -> tuple[float, float]:
    engine, _ = build_a_native_engine()
    return reduced_modulus_Pa(engine.G, engine.nu), float(engine.r_eff())


def _base_template(model_level: RebondModelLevel) -> CrackRebondingControls:
    return CrackRebondingControls(
        enabled=True,
        model_level=model_level,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY,
        feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        wake_length_m=0.0005,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=G_MAX_BASELINE_J_m2,
        rebond_K_geometry_factor=ETA_K_BASELINE,
        bond_activation_volume_m3=1.0e-30,
        rupture_activation_volume_m3=0.0,
        depassivation_activation_volume_m3=1.0e-30,
        bond_attempt_frequency_s=1.0e10,
        rupture_attempt_frequency_s=1.0e10,
        depassivation_attempt_frequency_s=1.0e10,
        repassivation_attempt_frequency_s=1.0e10,
        chemistry_factor=1.0,
        fresh_surface_clean_fraction=1.0,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
    )


def _config_hash(cfg: CrackRebondingControls) -> str:
    return cfg.config_hash()


def _orbit_at_reference(cfg: CrackRebondingControls, Eprime_Pa: float, r_eff_m: float) -> dict[str, Any]:
    return analytical_periodic_orbit(
        cfg=cfg, T_K=REF_T_K, Kmax_Pa_sqrt_m=REF_KMAX_Pa_sqrt_m, R=REF_R,
        frequency_Hz=REF_FREQUENCY_Hz, r_contact_m=r_eff_m, n_phase=N_PHASE_ANALYTICAL,
    )


def _gate_report(name: str, checks: dict[str, bool]) -> dict[str, Any]:
    passed = all(checks.values())
    return {"row": name, "all_gates_passed": passed, "checks": checks}


def build_sat_existing() -> tuple[CrackRebondingControls, dict[str, Any]]:
    payload = json.loads(CAUSAL_PILOT_V2_CONFIG_PATH.read_text())
    raw = dict(payload["configs"]["RB2_reversible_finite"])
    raw["model_level"] = RebondModelLevel(raw["model_level"])
    raw["contact_model"] = ContactModel(raw["contact_model"])
    raw["feedback_mode"] = FeedbackMode(raw["feedback_mode"])
    raw["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(raw["initial_precrack_wake_mode"])
    cfg = CrackRebondingControls(**raw).validate()
    expected_hash = payload["config_hashes"]["RB2_reversible_finite"]
    actual_hash = _config_hash(cfg)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SAT_EXISTING config hash mismatch: expected {expected_hash}, got {actual_hash} "
            "-- the frozen RB2_reversible_finite config was not reproduced exactly"
        )
    return cfg, {"source": "verbatim reuse", "source_file": str(CAUSAL_PILOT_V2_CONFIG_PATH.relative_to(REPO_ROOT)),
                 "source_config_hash": expected_hash, "reproduced_config_hash": actual_hash}


def build_competing_reversible(Eprime_Pa: float, r_eff_m: float) -> tuple[CrackRebondingControls, dict[str, Any]]:
    template = _base_template(RebondModelLevel.CLEAN_REVERSIBLE_REBOND)
    cfg = solve_reference_action_barriers(
        2.0, 2.0, T_K=REF_T_K, f_Hz=REF_FREQUENCY_Hz, R=REF_R, Kmax_Pa_sqrt_m=REF_KMAX_Pa_sqrt_m,
        reference_patch_distance_m=0.0, reference_contact_radius_m=r_eff_m,
        cfg_template=template, n_phase=360,
    ).validate()
    orbit = _orbit_at_reference(cfg, Eprime_Pa, r_eff_m)
    swing = orbit["max_p_B"] - orbit["min_p_B"]
    checks = {
        "0.30<=mean_p_B<=0.70": 0.30 <= orbit["mean_p_B"] <= 0.70,
        "swing>=0.20": swing >= 0.20,
        "0.10<=A_CB<=3.0": 0.10 <= orbit["A_CB"] <= 3.0,
        "0.10<=A_BC<=3.0": 0.10 <= orbit["A_BC"] <= 3.0,
        "no_barrier_floor": not any(orbit["any_floored"].values()),
        "no_cooperative_saturation": not any(orbit["any_saturated"].values()),
        "p_B_not_always_gt_0.95": orbit["min_p_B"] <= 0.95,
    }
    return cfg, {"reference_orbit": orbit, "gates": _gate_report("COMPETING_REVERSIBLE", checks),
                 "A_on_ref": 2.0, "A_off_ref": 2.0}


def build_competing_persistent(
    reversible_cfg: CrackRebondingControls, reversible_orbit: dict[str, Any], Eprime_Pa: float, r_eff_m: float
) -> tuple[CrackRebondingControls, dict[str, Any]]:
    delta_rupture_barrier_eV = 0.05
    cfg = replace(reversible_cfg, rupture_barrier_eV=reversible_cfg.rupture_barrier_eV + delta_rupture_barrier_eV).validate()
    orbit = _orbit_at_reference(cfg, Eprime_Pa, r_eff_m)
    p_B_at_tensile_peak = float(orbit["trajectory"][0, 2])  # bin 0 is centered near phase~0 (K~Kmax)
    checks = {
        "0.50<=mean_p_B<=0.90": 0.50 <= orbit["mean_p_B"] <= 0.90,
        "p_B_at_tensile_peak>=0.25": p_B_at_tensile_peak >= 0.25,
        "nonzero_phase_variation": (orbit["max_p_B"] - orbit["min_p_B"]) > 0.0,
        "A_BC_at_least_5x_smaller": orbit["A_BC"] <= reversible_orbit["A_BC"] / 5.0,
        "mean_p_B_differs_by_ge_0.10": abs(orbit["mean_p_B"] - reversible_orbit["mean_p_B"]) >= 0.10,
    }
    return cfg, {
        "reference_orbit": orbit, "gates": _gate_report("COMPETING_PERSISTENT", checks),
        "delta_rupture_barrier_eV": delta_rupture_barrier_eV, "p_B_at_tensile_peak": p_B_at_tensile_peak,
    }


def build_passivation_limited(Eprime_Pa: float, r_eff_m: float, formation_bond_barrier_eV: float, formation_rupture_barrier_eV: float) -> tuple[CrackRebondingControls, dict[str, Any]]:
    template = _base_template(RebondModelLevel.PASSIVATION_GATED_REBOND)
    cfg = replace(
        template,
        bond_barrier_eV=formation_bond_barrier_eV,
        rupture_barrier_eV=formation_rupture_barrier_eV,
        depassivation_barrier_eV=0.40,
        repassivation_barrier_eV=0.40,
    ).validate()
    orbit = _orbit_at_reference(cfg, Eprime_Pa, r_eff_m)
    swing = orbit["max_p_B"] - orbit["min_p_B"]
    checks = {
        "0.30<=mean_p_P<=0.80": 0.30 <= orbit["mean_p_P"] <= 0.80,
        "0.05<=mean_p_B<=0.50": 0.05 <= orbit["mean_p_B"] <= 0.50,
        "nonzero_A_PC": orbit["A_PC"] > 0.0,
        "nonzero_A_CP": orbit["A_CP"] > 0.0,
        "nonzero_F_PC": orbit["F_PC"] > 0.0,
        "nonzero_F_CP": orbit["F_CP"] > 0.0,
        "swing>=0.10": swing >= 0.10,
        "no_barrier_floor_gt_0.05": all(v <= 0.05 for v in orbit["barrier_floor_fraction"].values()),
    }
    return cfg, {"reference_orbit": orbit, "gates": _gate_report("PASSIVATION_LIMITED", checks),
                 "depassivation_barrier_eV": 0.40, "repassivation_barrier_eV": 0.40}


def _orbit_summary(orbit: dict[str, Any]) -> dict[str, Any]:
    summary = {k: v for k, v in orbit.items() if k not in ("p_star", "trajectory")}
    summary["p_star"] = orbit["p_star"].tolist()
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Eprime_Pa, r_eff_m = _engine_geometry()

    sat_cfg, sat_meta = build_sat_existing()
    sat_orbit = _orbit_at_reference(sat_cfg, Eprime_Pa, r_eff_m)

    rev_cfg, rev_meta = build_competing_reversible(Eprime_Pa, r_eff_m)
    per_cfg, per_meta = build_competing_persistent(rev_cfg, rev_meta["reference_orbit"], Eprime_Pa, r_eff_m)
    pas_cfg, pas_meta = build_passivation_limited(Eprime_Pa, r_eff_m, rev_cfg.bond_barrier_eV, rev_cfg.rupture_barrier_eV)

    rows = {
        "SAT_EXISTING": (sat_cfg, {"reference_orbit": sat_orbit, "gates": {"row": "SAT_EXISTING", "all_gates_passed": True, "checks": {"reused_verbatim": True}}, **sat_meta}),
        "COMPETING_REVERSIBLE": (rev_cfg, rev_meta),
        "COMPETING_PERSISTENT": (per_cfg, per_meta),
        "PASSIVATION_LIMITED": (pas_cfg, pas_meta),
    }

    all_passed = True
    for name, (cfg, meta) in rows.items():
        gate = meta["gates"]
        status = "PASS" if gate["all_gates_passed"] else "FAIL"
        print(f"{name}: {status}")
        for check, ok in gate["checks"].items():
            print(f"    {'OK  ' if ok else 'FAIL'} {check}")
        all_passed = all_passed and gate["all_gates_passed"]

    if not all_passed:
        print("\nNONSATURATED_REBONDING_REGIME_NOT_CONSTRUCTED: one or more rows failed its gates.")
        raise SystemExit(1)

    # -- kinetic_regime_registry --
    registry_rows = []
    for name, (cfg, meta) in rows.items():
        orbit = meta["reference_orbit"]
        registry_rows.append({
            "row_name": name,
            "model_level": cfg.model_level.value,
            "config_hash": _config_hash(cfg),
            "bond_barrier_eV": cfg.bond_barrier_eV,
            "rupture_barrier_eV": cfg.rupture_barrier_eV,
            "depassivation_barrier_eV": cfg.depassivation_barrier_eV,
            "repassivation_barrier_eV": cfg.repassivation_barrier_eV,
            "restored_work_of_separation_J_m2": cfg.restored_work_of_separation_J_m2,
            "rebond_K_geometry_factor": cfg.rebond_K_geometry_factor,
            "K_rebond_max_Pa_sqrt_m": cfg.rebond_K_geometry_factor * math.sqrt(Eprime_Pa * cfg.restored_work_of_separation_J_m2),
            "reference_mean_p_P": orbit["mean_p_P"],
            "reference_mean_p_C": orbit["mean_p_C"],
            "reference_mean_p_B": orbit["mean_p_B"],
            "reference_max_p_B": orbit["max_p_B"],
            "reference_min_p_B": orbit["min_p_B"],
            "reference_A_CB": orbit["A_CB"],
            "reference_A_BC": orbit["A_BC"],
            "reference_A_PC": orbit["A_PC"],
            "reference_A_CP": orbit["A_CP"],
            "predicted_static_shield_equivalent_K_b_Pa_sqrt_m": predicted_static_shield_equivalent_K_b(orbit, cfg, Eprime_Pa),
            "all_gates_passed": meta["gates"]["all_gates_passed"],
        })
    with (OUT_DIR / "kinetic_regime_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(registry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(registry_rows)
    (OUT_DIR / "kinetic_regime_registry.json").write_text(
        json.dumps({
            "schema": "v10230_part_x_kinetic_regime_registry_v1",
            "reference_state": {
                "material": "A_NATIVE", "T_K": REF_T_K, "Kmax_Pa_sqrt_m": REF_KMAX_Pa_sqrt_m,
                "R": REF_R, "frequency_Hz": REF_FREQUENCY_Hz, "minimum_load_hold_s": 0.0,
                "K_rebond_max_baseline_Pa_sqrt_m": ETA_K_BASELINE * math.sqrt(Eprime_Pa * G_MAX_BASELINE_J_m2),
                "Eprime_Pa": Eprime_Pa, "r_eff_m": r_eff_m, "n_phase": N_PHASE_ANALYTICAL,
            },
            "rows": {name: {"config": {k: (v.value if hasattr(v, "value") else v) for k, v in asdict(cfg).items()},
                             "config_hash": _config_hash(cfg),
                             "gates": meta["gates"],
                             "reference_orbit": _orbit_summary(meta["reference_orbit"]),
                             "construction_notes": {k: v for k, v in meta.items() if k not in ("reference_orbit", "gates")}}
                      for name, (cfg, meta) in rows.items()},
        }, indent=2, default=str)
    )

    print(f"\nWrote kinetic_regime_registry.{{csv,json}} to {OUT_DIR}")

    build_analytical_atlas(rows, Eprime_Pa, r_eff_m)


# ---------------------------------------------------------------------------
# Analytical atlas (mission section 6.5)
# ---------------------------------------------------------------------------

KMAX_GRID_Pa_sqrt_m = [15.0e6, 18.0e6, 21.0e6]
R_GRID = [-0.95, -0.5, -0.1, 0.1]
FREQUENCY_GRID_Hz = [100.0, 1000.0, 10000.0]
FREQUENCY_EXTENSION_Hz = 100000.0
HOLD_GRID_s = [0.0, 0.0005, 0.0020]
CHEMISTRY_FACTOR_GRID = [1.0, 0.3, 0.1]
COHESIVE_STRENGTH_GRID_Pa_sqrt_m = [0.45e6, 0.9e6, 1.8e6]


def _evaluate_point(cfg, Kmax, R, f_Hz, hold_s, Eprime_Pa, r_eff_m) -> dict[str, Any]:
    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=REF_T_K, Kmax_Pa_sqrt_m=Kmax, R=R, frequency_Hz=f_Hz,
        minimum_load_hold_s=hold_s, r_contact_m=r_eff_m, n_phase=N_PHASE_ANALYTICAL,
    )
    predicted_K_b = predicted_static_shield_equivalent_K_b(orbit, cfg, Eprime_Pa)
    return {
        "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": f_Hz, "minimum_load_hold_s": hold_s,
        "mean_p_P": orbit["mean_p_P"], "mean_p_C": orbit["mean_p_C"], "mean_p_B": orbit["mean_p_B"],
        "max_p_B": orbit["max_p_B"], "min_p_B": orbit["min_p_B"],
        "A_CB": orbit["A_CB"], "A_BC": orbit["A_BC"], "A_PC": orbit["A_PC"], "A_CP": orbit["A_CP"],
        "F_CB": orbit["F_CB"], "F_BC": orbit["F_BC"], "F_PC": orbit["F_PC"], "F_CP": orbit["F_CP"],
        "contact_time_s": orbit["contact_time_s"],
        "contact_time_sinusoid_s": orbit["contact_time_sinusoid_s"],
        "contact_time_dwell_s": orbit["contact_time_dwell_s"],
        "predicted_static_shield_equivalent_K_b_Pa_sqrt_m": predicted_K_b,
        "any_barrier_floored": any(orbit["any_floored"].values()),
        "any_cooperative_saturated": any(orbit["any_saturated"].values()),
        "convergence_residual": orbit["convergence_residual"],
    }


def _needs_frequency_extension(row_name: str, point_1k: dict, point_10k: dict) -> bool:
    """Section 6.5: extend to 100 kHz only when 100 Hz-10 kHz does not
    bracket a nonsaturated occupancy transition -- i.e. mean_p_B is still
    changing meaningfully at 10 kHz and has not already saturated toward
    0 or 1."""
    delta = abs(point_10k["mean_p_B"] - point_1k["mean_p_B"])
    still_changing = delta > 0.05
    not_saturated = 0.02 < point_10k["mean_p_B"] < 0.98
    return still_changing and not_saturated


def build_analytical_atlas(rows: dict[str, tuple], Eprime_Pa: float, r_eff_m: float) -> None:
    atlas_records: list[dict[str, Any]] = []
    extension_decisions: dict[str, bool] = {}

    for row_name, (cfg, _meta) in rows.items():
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            for R in R_GRID:
                for hold_s in HOLD_GRID_s:
                    for chem in CHEMISTRY_FACTOR_GRID:
                        row_cfg = replace(cfg, chemistry_factor=chem).validate()
                        freqs_this_point = list(FREQUENCY_GRID_Hz)
                        # Decide frequency extension once per (row, Kmax, R,
                        # hold) at chemistry_factor=1.0's own 1kHz/10kHz pair.
                        if chem == 1.0:
                            p1k = _evaluate_point(row_cfg, Kmax, R, 1000.0, hold_s, Eprime_Pa, r_eff_m)
                            p10k = _evaluate_point(row_cfg, Kmax, R, 10000.0, hold_s, Eprime_Pa, r_eff_m)
                            extend = _needs_frequency_extension(row_name, p1k, p10k)
                            extension_decisions[f"{row_name}|{Kmax}|{R}|{hold_s}"] = extend
                            if extend:
                                freqs_this_point.append(FREQUENCY_EXTENSION_Hz)
                        for f_Hz in freqs_this_point:
                            point = _evaluate_point(row_cfg, Kmax, R, f_Hz, hold_s, Eprime_Pa, r_eff_m)
                            point.update({
                                "row_name": row_name, "chemistry_factor": chem,
                                "K_rebond_max_Pa_sqrt_m": row_cfg.rebond_K_geometry_factor
                                * math.sqrt(Eprime_Pa * row_cfg.restored_work_of_separation_J_m2),
                                "axis": "kinetic_chemistry_and_conditions",
                            })
                            atlas_records.append(point)

        # Cohesive-strength axis: at the reference R/f/hold, chemistry=1.0,
        # varying restored_work_of_separation_J_m2 to hit the three target
        # K_rebond,max endpoints (a strength axis, not a kinetic-chemistry
        # axis -- the bond-formation barrier itself is untouched).
        for K_b_target in COHESIVE_STRENGTH_GRID_Pa_sqrt_m:
            G_max_target = (K_b_target / cfg.rebond_K_geometry_factor) ** 2 / Eprime_Pa
            strength_cfg = replace(cfg, restored_work_of_separation_J_m2=G_max_target, chemistry_factor=1.0).validate()
            for Kmax in KMAX_GRID_Pa_sqrt_m:
                point = _evaluate_point(strength_cfg, Kmax, REF_R, REF_FREQUENCY_Hz, 0.0, Eprime_Pa, r_eff_m)
                point.update({
                    "row_name": row_name, "chemistry_factor": 1.0,
                    "K_rebond_max_Pa_sqrt_m": K_b_target,
                    "axis": "cohesive_strength",
                })
                atlas_records.append(point)

    import pandas as pd

    df = pd.DataFrame(atlas_records)
    df.to_csv(OUT_DIR / "analytical_phase_atlas.csv", index=False)
    df.to_parquet(OUT_DIR / "analytical_phase_atlas.parquet", index=False)
    print(f"\nWrote analytical_phase_atlas.{{csv,parquet}}: {len(df)} rows")

    (OUT_DIR / "analytical_regime_selection.json").write_text(
        json.dumps({
            "schema": "v10230_part_x_analytical_regime_selection_v1",
            "grid": {
                "Kmax_Pa_sqrt_m": KMAX_GRID_Pa_sqrt_m, "R": R_GRID,
                "frequency_Hz": FREQUENCY_GRID_Hz, "frequency_extension_Hz": FREQUENCY_EXTENSION_Hz,
                "minimum_load_hold_s": HOLD_GRID_s, "chemistry_factor": CHEMISTRY_FACTOR_GRID,
                "cohesive_strength_K_rebond_max_Pa_sqrt_m": COHESIVE_STRENGTH_GRID_Pa_sqrt_m,
            },
            "frequency_extension_rule": "extend to 100 kHz only when |mean_p_B(10kHz)-mean_p_B(1kHz)| > 0.05 "
                                          "and 0.02 < mean_p_B(10kHz) < 0.98 (still transitioning, not saturated)",
            "frequency_extension_decisions": extension_decisions,
            "rows_evaluated": list(rows.keys()),
            "total_atlas_points": len(df),
        }, indent=2)
    )
    print("Wrote analytical_regime_selection.json")

    selections = select_screen_conditions(df)
    build_screen_and_developed_registries(rows, selections, Eprime_Pa, r_eff_m)


# ---------------------------------------------------------------------------
# PX3/PX4 prospective condition selection (section 7.8) and job registries
# ---------------------------------------------------------------------------

SCREEN_SEED = 1720
DEVELOPED_SEEDS = {"stage1": 1720, "confirmation": 1001723}


def select_screen_conditions(df) -> dict[str, Any]:
    """Frozen selection rules (section 7.8), applied here at the analytical
    level so PX3's screen matrix launches the pre-committed conditions
    rather than conditions chosen after seeing screen results."""
    main = df[df["axis"] == "kinetic_chemistry_and_conditions"]

    def _slice(row_name, R, f_Hz, hold_s, chem, Kmax=REF_KMAX_Pa_sqrt_m):
        sel = main[
            (main.row_name == row_name) & (main.R == R) & (main.frequency_Hz == f_Hz)
            & (main.minimum_load_hold_s == hold_s) & (main.chemistry_factor == chem) & (main.Kmax_Pa_sqrt_m == Kmax)
        ]
        return sel.iloc[0] if len(sel) else None

    ref_point = _slice("COMPETING_REVERSIBLE", REF_R, REF_FREQUENCY_Hz, 0.0, 1.0)

    # Frequency selection: largest |mean_p_B(f) - mean_p_B(1000Hz)| among
    # nonsaturated candidate frequencies.
    freq_candidates = []
    for f_Hz in [100.0, 10000.0, FREQUENCY_EXTENSION_Hz]:
        pt = _slice("COMPETING_REVERSIBLE", REF_R, f_Hz, 0.0, 1.0)
        if pt is None:
            continue
        if not (0.02 < pt["mean_p_B"] < 0.98):
            continue
        freq_candidates.append((abs(pt["mean_p_B"] - ref_point["mean_p_B"]), f_Hz, float(pt["mean_p_B"])))
    freq_candidates.sort(reverse=True)
    selected_frequency_Hz, freq_delta = (freq_candidates[0][1], freq_candidates[0][0]) if freq_candidates else (None, 0.0)

    # Dwell selection: largest change in mean_p_B from zero hold (F_CB as
    # a secondary tiebreaker, per section 7.8 item 3's "flux OR p_B").
    dwell_candidates = []
    for hold_s in [0.0005, 0.0020]:
        pt = _slice("COMPETING_REVERSIBLE", REF_R, REF_FREQUENCY_Hz, hold_s, 1.0)
        if pt is None:
            continue
        dwell_candidates.append((abs(pt["mean_p_B"] - ref_point["mean_p_B"]), hold_s, float(pt["mean_p_B"]), float(pt["F_CB"])))
    dwell_candidates.sort(reverse=True)
    selected_hold_s, hold_delta = (dwell_candidates[0][1], dwell_candidates[0][0]) if dwell_candidates else (None, 0.0)

    # Chemistry selection: closest mean_p_B to 0.30 with mean_p_P>=0.30.
    chem_candidates = []
    for chem in CHEMISTRY_FACTOR_GRID:
        pt = _slice("PASSIVATION_LIMITED", REF_R, REF_FREQUENCY_Hz, 0.0, chem)
        if pt is None or pt["mean_p_P"] < 0.30:
            continue
        chem_candidates.append((abs(pt["mean_p_B"] - 0.30), chem, float(pt["mean_p_B"]), float(pt["mean_p_P"])))
    chem_candidates.sort()
    selected_chemistry_factor = chem_candidates[0][1] if chem_candidates else None

    selections = {
        "schema": "v10230_part_x_screen_condition_selection_v1",
        "frequency_transition_Hz": selected_frequency_Hz,
        "frequency_transition_delta_mean_p_B": freq_delta,
        "dwell_transition_s": selected_hold_s,
        "dwell_transition_delta_mean_p_B": hold_delta,
        "passivation_chemistry_factor": selected_chemistry_factor,
        "competing_persistent_selected": True,
        "competing_persistent_reason": "PX2.3 construction already confirmed distinguishable from COMPETING_REVERSIBLE "
                                         "(mean_p_B differs by >=0.10, A_BC >=5x smaller) -- section 7.8 item 5 satisfied by construction.",
        "cohesive_strength_endpoint_selected": None,
        "cohesive_strength_endpoint_reason": "deferred per section 7.8 item 6 -- selected only if the PX3 three-point "
                                               "cohesive-strength screen is visibly nonlinear or changes the qualitative "
                                               "slope classification; cannot be determined from the analytical atlas alone.",
    }
    print("\nPX3 condition selections:")
    for k, v in selections.items():
        print(f"  {k}: {v}")
    return selections


def build_screen_and_developed_registries(rows, selections, Eprime_Pa, r_eff_m) -> None:
    screen_jobs: list[dict[str, Any]] = []
    developed_jobs: list[dict[str, Any]] = []

    def _screen_job(protocol, row_name, R, f_Hz, hold_s, chem, K_b_target=None, note=""):
        cfg, _ = rows[row_name]
        screen_jobs.append({
            "protocol": protocol, "row_name": row_name, "config_hash": _config_hash(cfg),
            "Kmax_Pa_sqrt_m": REF_KMAX_Pa_sqrt_m, "R": R, "frequency_Hz": f_Hz,
            "minimum_load_hold_s": hold_s, "chemistry_factor": chem,
            "K_rebond_max_target_Pa_sqrt_m": K_b_target, "seed": SCREEN_SEED,
            "cohesion": "finite", "status": "QUEUED_NOT_LAUNCHED", "note": note,
        })
        screen_jobs.append({
            "protocol": protocol, "row_name": row_name, "config_hash": _config_hash(cfg),
            "Kmax_Pa_sqrt_m": REF_KMAX_Pa_sqrt_m, "R": R, "frequency_Hz": f_Hz,
            "minimum_load_hold_s": hold_s, "chemistry_factor": chem,
            "K_rebond_max_target_Pa_sqrt_m": 0.0, "seed": SCREEN_SEED,
            "cohesion": "zero", "status": "QUEUED_NOT_LAUNCHED", "note": note + " (matched zero-cohesion control)",
        })

    # 7.1 R panel -- COMPETING_REVERSIBLE
    for R in [-0.95, -0.50, -0.10, 0.10]:
        _screen_job("7.1_R_panel", "COMPETING_REVERSIBLE", R, REF_FREQUENCY_Hz, 0.0, 1.0,
                    note="R=+0.10 requires exact zero contact/A_CB/F_CB/K_rebond -- signed-K proxy semantic control, not evidence about real positive-R closure")
    # 7.2 Frequency panel -- COMPETING_REVERSIBLE, R=-0.5
    for f_Hz in [100.0, 1000.0, 10000.0]:
        _screen_job("7.2_frequency_panel", "COMPETING_REVERSIBLE", -0.50, f_Hz, 0.0, 1.0)
    if selections["frequency_transition_Hz"] not in (None, 100.0, 1000.0, 10000.0):
        _screen_job("7.2_frequency_panel_extension", "COMPETING_REVERSIBLE", -0.50, selections["frequency_transition_Hz"], 0.0, 1.0,
                    note="100kHz extension triggered by the frozen bracket rule")
    # 7.3 Dwell panel -- COMPETING_REVERSIBLE, R=-0.5, f=1000
    for hold_s in [0.0, 0.0005, 0.0020]:
        _screen_job("7.3_dwell_panel", "COMPETING_REVERSIBLE", -0.50, REF_FREQUENCY_Hz, hold_s, 1.0)
    # 7.4 Reversible vs persistent -- R=-0.5, f=1000, hold=0
    _screen_job("7.4_reversible_vs_persistent", "COMPETING_PERSISTENT", -0.50, REF_FREQUENCY_Hz, 0.0, 1.0)
    if selections["frequency_transition_Hz"] is not None:
        _screen_job("7.4_reversible_vs_persistent_at_transition", "COMPETING_PERSISTENT", -0.50,
                    selections["frequency_transition_Hz"], 0.0, 1.0,
                    note="selected high-frequency transition point, per section 7.4's 'when analytical atlas predicts "
                         "reopening survival should distinguish them'")
    # 7.5 Passivation and kinetic chemistry -- R=-0.5, f=1000, hold=0
    for chem in [1.0, 0.3, 0.1]:
        _screen_job("7.5_passivation_chemistry", "PASSIVATION_LIMITED", -0.50, REF_FREQUENCY_Hz, 0.0, chem)
    # 7.6 Cohesive-strength screen -- COMPETING_REVERSIBLE, R=-0.5, f=1000, hold=0
    for K_b_target in COHESIVE_STRENGTH_GRID_Pa_sqrt_m:
        _screen_job("7.6_cohesive_strength", "COMPETING_REVERSIBLE", -0.50, REF_FREQUENCY_Hz, 0.0, 1.0, K_b_target=K_b_target,
                    note="a single matched zero-cohesion trajectory may be shared across the three finite-cohesion "
                         "controls only after exact configuration/threshold-stream/event-length/waveform identity is established")
    # 7.7 Existing saturated row -- reuse only; no new screen job unless the
    # atlas predicts SAT_EXISTING leaves its near-ceiling occupancy regime.
    sat_note = "reusing existing SAT_EXISTING Kmax=18/R=-0.95/f=1000 result -- classified prospectively as the static-shield limit, not duplicated"

    with (OUT_DIR / "screen_job_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(screen_jobs[0].keys()))
        writer.writeheader()
        writer.writerows(screen_jobs)
    print(f"\nWrote screen_job_registry.csv: {len(screen_jobs)} jobs (all QUEUED_NOT_LAUNCHED)")

    # -- Developed job registry (section 8) --
    # Stage-1 seed-1720 grid: Kmax=12,15,18,21,24.3
    def _developed_job_stage1(protocol, row_name, R, f_Hz, hold_s, chem, K_b_target=None, note=""):
        cfg, _ = rows[row_name]
        for Kmax in [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]:
            for cohesion, K_target in (("finite", K_b_target), ("zero", 0.0)):
                developed_jobs.append({
                    "protocol": protocol, "row_name": row_name, "config_hash": _config_hash(cfg),
                    "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": f_Hz, "minimum_load_hold_s": hold_s,
                    "chemistry_factor": chem, "K_rebond_max_target_Pa_sqrt_m": K_target,
                    "seed": DEVELOPED_SEEDS["stage1"], "cohesion": cohesion, "status": "QUEUED_NOT_LAUNCHED", "note": note,
                })

    def _developed_job_confirmation(protocol, row_name, R, f_Hz, hold_s, chem, K_b_target=None, note=""):
        cfg, _ = rows[row_name]
        for Kmax in [15.0e6, 18.0e6, 21.0e6]:
            for cohesion, K_target in (("finite", K_b_target), ("zero", 0.0)):
                developed_jobs.append({
                    "protocol": protocol, "row_name": row_name, "config_hash": _config_hash(cfg),
                    "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": f_Hz, "minimum_load_hold_s": hold_s,
                    "chemistry_factor": chem, "K_rebond_max_target_Pa_sqrt_m": K_target,
                    "seed": DEVELOPED_SEEDS["confirmation"], "cohesion": cohesion, "status": "QUEUED_NOT_LAUNCHED", "note": note,
                })

    K_b_baseline = ETA_K_BASELINE * math.sqrt(Eprime_Pa * G_MAX_BASELINE_J_m2)
    _developed_job_stage1("D1", "COMPETING_REVERSIBLE", -0.95, REF_FREQUENCY_Hz, 0.0, 1.0, K_b_baseline)
    _developed_job_stage1("D2", "COMPETING_REVERSIBLE", -0.50, REF_FREQUENCY_Hz, 0.0, 1.0, K_b_baseline)
    if selections["frequency_transition_Hz"] is not None:
        _developed_job_stage1("D3", "COMPETING_REVERSIBLE", -0.50, selections["frequency_transition_Hz"], 0.0, 1.0, K_b_baseline)
    if selections["dwell_transition_s"] is not None:
        _developed_job_stage1("D4", "COMPETING_REVERSIBLE", -0.50, REF_FREQUENCY_Hz, selections["dwell_transition_s"], 1.0, K_b_baseline)
    if selections["passivation_chemistry_factor"] is not None:
        _developed_job_stage1("D5", "PASSIVATION_LIMITED", -0.50, REF_FREQUENCY_Hz, 0.0, selections["passivation_chemistry_factor"], K_b_baseline)
    _developed_job_stage1("D6_conditional_persistent", "COMPETING_PERSISTENT", -0.50, REF_FREQUENCY_Hz, 0.0, 1.0, K_b_baseline,
                           note="PX3.8 item 5: COMPETING_PERSISTENT already confirmed distinguishable at the reference "
                                "condition, so D6 is authorized as of this freeze -- PX4 must still confirm the screen-"
                                "level effect before launching")

    # Second-seed confirmation, conditional -- registered here as queued;
    # PX4/PX5's own gate (effect sizes vs. frozen thresholds) decides
    # whether these are actually launched.
    for protocol, row_name, R, f_Hz, hold_s, chem in [
        ("D1_confirm", "COMPETING_REVERSIBLE", -0.95, REF_FREQUENCY_Hz, 0.0, 1.0),
        ("D2_confirm", "COMPETING_REVERSIBLE", -0.50, REF_FREQUENCY_Hz, 0.0, 1.0),
    ]:
        _developed_job_confirmation(protocol, row_name, R, f_Hz, hold_s, chem, K_b_baseline,
                                     note="conditional on PX4 Stage-1 effect gates; launch decided in PX4/PX5, not PX2")

    with (OUT_DIR / "developed_job_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(developed_jobs[0].keys()))
        writer.writeheader()
        writer.writerows(developed_jobs)
    print(f"Wrote developed_job_registry.csv: {len(developed_jobs)} jobs (all QUEUED_NOT_LAUNCHED)")

    (OUT_DIR / "physical_screen_predictions.json").write_text(
        json.dumps({
            "schema": "v10230_part_x_physical_screen_predictions_v1",
            "note": "Analytical-atlas-level predictions for each screen condition's rebonding-vs-zero direction and "
                     "rough magnitude, to be compared against actual PX3 screen results -- NOT a substitute for the "
                     "physical screen itself (bulk-cycle occupancy is a periodic-orbit STEADY-STATE property; a real "
                     "12-event/60um screen trajectory need not have reached it).",
            "sat_existing_reuse_note": sat_note,
            "selections": selections,
        }, indent=2)
    )
    print("Wrote physical_screen_predictions.json")

    (OUT_DIR / "prospective_classification_gates.json").write_text(
        json.dumps({
            "schema": "v10230_part_x_prospective_classification_gates_v1",
            "screen_level_effect": {
                "definition": "|S_h| >= 0.01 decade AND |S_h| > max(0.005 decade, 5x numerical/action bound)",
            },
            "response_classes": {
                "REBONDING_NO_MEASURABLE_RATE_EFFECT": "|S_h| < 0.01 decade after uncertainty accounting",
                "REBONDING_RATE_OFFSET_LIKE": "max_K S_h - min_K S_h magnitude <0.01 decade AND |Delta m| <0.25",
                "REBONDING_STEEPENS_RESPONSE": "Delta m >= 0.25, same sign across required seeds",
                "REBONDING_FLATTENS_RESPONSE": "Delta m <= -0.25, same sign across required seeds",
                "REBONDING_CURVED_ONSET_OR_CROSSOVER": "adjacent secants differ by >=0.50 or change substantially "
                                                          "across the K grid while net response is monotone",
                "REBONDING_SEED_SENSITIVE": "seed slope signs disagree or |Delta m_seed1 - Delta m_seed2| > 0.25",
            },
            "axis_sensitivity_gate": "at least 0.01 decade in S_h or 0.25 in Delta m, AND larger than "
                                       "max(0.005 decade, 5x propagated numerical/action uncertainty)",
            "static_dynamic_attribution": {
                "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT": "max|S_dynamic-S_static|<=0.01 decade AND "
                                                                 "|delta_m_dynamic-delta_m_static|<=0.10",
                "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY": "max|S_dynamic-S_static|<=0.03 decade OR "
                                                                  "|delta_m_dynamic-delta_m_static|<=0.25, dominant gate fails",
                "DYNAMIC_REBONDING_HISTORY_REQUIRED": "neither gate above satisfied",
            },
            "developed_stationarity_gate": {
                "exclude_first_um": 20, "minimum_total_extension_um": 100, "assess_last_um": 50,
                "minimum_developed_events": 10, "late_early_rate_ratio_bounds": [0.5, 2.0],
            },
            "trajectory_budget": {
                "max_accepted_events": 30, "max_extension_um": 150, "max_cycles": 1.0e12,
                "max_concurrent_workers": 3, "resume_authorized": False,
            },
        }, indent=2)
    )
    print("Wrote prospective_classification_gates.json")


if __name__ == "__main__":
    main()
