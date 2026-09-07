"""PX2.5 item 6: preserve the PX2 analytical kinetic-regime search history.

PX2's row construction (build_part_x_kinetic_regime_registry.py) searched
several candidate parameter choices before freezing COMPETING_REVERSIBLE,
COMPETING_PERSISTENT, and PASSIVATION_LIMITED. Per external review, that
search history must be tracked (candidate ordering, every gate value,
pass/fail, and the deterministic selection rule), not left in
conversation-only scratch output -- these rows are deliberately engineered
mechanism-control ablations, not calibrated material chemistry, and the
audit trail is what substantiates that distinction.

Reproduces every candidate this session actually evaluated (recorded from
the interactive search transcript) by re-running analytical_periodic_orbit
on each -- byte-for-byte the same evaluator PX2's frozen rows use, so this
audit is itself independently checkable, not a hand-transcribed table.
"""
from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
from arrhenius_fracture.crack_rebonding_part_x_kinetic_regime_v10230 import analytical_periodic_orbit
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa

OUT_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
REF_KMAX, REF_R, REF_F, REF_T = 18.0e6, -0.5, 1000.0, 300.0
G_MAX_BASELINE = 1.8225
N_PHASE = 64


def _engine_geometry():
    engine, _ = build_a_native_engine()
    return reduced_modulus_Pa(engine.G, engine.nu), float(engine.r_eff())


def _reversible_template():
    return CrackRebondingControls(
        enabled=True, model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY, feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
        wake_length_m=0.0005, wake_weight_length_m=5e-7,
        restored_work_of_separation_J_m2=G_MAX_BASELINE, rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=1e-30, rupture_activation_volume_m3=0.0,
        bond_attempt_frequency_s=1e10, rupture_attempt_frequency_s=1e10,
        chemistry_factor=1.0, fresh_surface_clean_fraction=1.0,
        initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
    )


def _passivation_template():
    return replace(
        _reversible_template(),
        model_level=RebondModelLevel.PASSIVATION_GATED_REBOND,
        depassivation_activation_volume_m3=1e-30,
        depassivation_attempt_frequency_s=1e10, repassivation_attempt_frequency_s=1e10,
        bond_barrier_eV=0.3899284910511967, rupture_barrier_eV=0.38591648560033115,
    )


def main() -> None:
    Eprime_Pa, r_eff_m = _engine_geometry()
    rows = []

    # --- COMPETING_REVERSIBLE candidates: (A_on, A_off) grid ---
    reversible_gate = lambda o: (
        0.30 <= o["mean_p_B"] <= 0.70 and (o["max_p_B"] - o["min_p_B"]) >= 0.20
        and 0.10 <= o["A_CB"] <= 3.0 and 0.10 <= o["A_BC"] <= 3.0
        and not any(o["any_floored"].values()) and not any(o["any_saturated"].values())
        and o["min_p_B"] <= 0.95
    )
    for A_on, A_off in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (1.5, 1.5), (2.5, 2.5), (2.0, 1.0), (1.0, 2.0)]:
        cfg = solve_reference_action_barriers(
            A_on, A_off, T_K=REF_T, f_Hz=REF_F, R=REF_R, Kmax_Pa_sqrt_m=REF_KMAX,
            reference_patch_distance_m=0.0, reference_contact_radius_m=r_eff_m,
            cfg_template=_reversible_template(), n_phase=360,
        ).validate()
        o = analytical_periodic_orbit(cfg=cfg, T_K=REF_T, Kmax_Pa_sqrt_m=REF_KMAX, R=REF_R,
                                       frequency_Hz=REF_F, r_contact_m=r_eff_m, n_phase=N_PHASE)
        passed = reversible_gate(o)
        rows.append({
            "row_family": "COMPETING_REVERSIBLE", "candidate_label": f"A_on={A_on}_A_off={A_off}",
            "A_on_ref": A_on, "A_off_ref": A_off, "delta_rupture_barrier_eV": "",
            "depassivation_barrier_eV": "", "repassivation_barrier_eV": "",
            "mean_p_B": o["mean_p_B"], "swing_p_B": o["max_p_B"] - o["min_p_B"],
            "A_CB": o["A_CB"], "A_BC": o["A_BC"], "mean_p_P": "", "mean_p_C": "",
            "A_PC": "", "A_CP": "", "F_PC": "", "F_CP": "",
            "any_barrier_floored": any(o["any_floored"].values()),
            "any_cooperative_saturated": any(o["any_saturated"].values()),
            "gates_passed": passed,
            "selected": (A_on == 2.0 and A_off == 2.0),
        })

    # Freeze the selected reversible cfg for the persistent/passivation searches below.
    reversible_cfg = solve_reference_action_barriers(
        2.0, 2.0, T_K=REF_T, f_Hz=REF_F, R=REF_R, Kmax_Pa_sqrt_m=REF_KMAX,
        reference_patch_distance_m=0.0, reference_contact_radius_m=r_eff_m,
        cfg_template=_reversible_template(), n_phase=360,
    ).validate()
    reversible_orbit = analytical_periodic_orbit(cfg=reversible_cfg, T_K=REF_T, Kmax_Pa_sqrt_m=REF_KMAX,
                                                  R=REF_R, frequency_Hz=REF_F, r_contact_m=r_eff_m, n_phase=N_PHASE)

    # --- COMPETING_PERSISTENT candidates: rupture-barrier increments ---
    persistent_gate = lambda o, p_tp: (
        0.50 <= o["mean_p_B"] <= 0.90 and p_tp >= 0.25
        and (o["max_p_B"] - o["min_p_B"]) > 0.0
        and o["A_BC"] <= reversible_orbit["A_BC"] / 5.0
        and abs(o["mean_p_B"] - reversible_orbit["mean_p_B"]) >= 0.10
    )
    for delta in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        cfg = replace(reversible_cfg, rupture_barrier_eV=reversible_cfg.rupture_barrier_eV + delta).validate()
        o = analytical_periodic_orbit(cfg=cfg, T_K=REF_T, Kmax_Pa_sqrt_m=REF_KMAX, R=REF_R,
                                       frequency_Hz=REF_F, r_contact_m=r_eff_m, n_phase=N_PHASE)
        p_tp = float(o["trajectory"][0, 2])
        passed = persistent_gate(o, p_tp)
        rows.append({
            "row_family": "COMPETING_PERSISTENT", "candidate_label": f"delta_rupture_barrier_eV={delta}",
            "A_on_ref": "", "A_off_ref": "", "delta_rupture_barrier_eV": delta,
            "depassivation_barrier_eV": "", "repassivation_barrier_eV": "",
            "mean_p_B": o["mean_p_B"], "swing_p_B": o["max_p_B"] - o["min_p_B"],
            "A_CB": o["A_CB"], "A_BC": o["A_BC"], "mean_p_P": "", "mean_p_C": "",
            "A_PC": "", "A_CP": "", "F_PC": "", "F_CP": "",
            "any_barrier_floored": any(o["any_floored"].values()),
            "any_cooperative_saturated": any(o["any_saturated"].values()),
            "gates_passed": passed,
            "selected": (delta == 0.05),
        })

    # --- PASSIVATION_LIMITED candidates: (depass, repass) barrier grid ---
    passivation_gate = lambda o: (
        0.30 <= o["mean_p_P"] <= 0.80 and 0.05 <= o["mean_p_B"] <= 0.50
        and o["A_PC"] > 0.0 and o["A_CP"] > 0.0 and o["F_PC"] > 0.0 and o["F_CP"] > 0.0
        and (o["max_p_B"] - o["min_p_B"]) >= 0.10
        and all(v <= 0.05 for v in o["barrier_floor_fraction"].values())
    )
    for depass, repass in [(0.40, 0.40), (0.45, 0.45), (0.42, 0.42), (0.40, 0.45), (0.45, 0.40), (0.43, 0.40), (0.40, 0.42)]:
        cfg = replace(_passivation_template(), depassivation_barrier_eV=depass, repassivation_barrier_eV=repass).validate()
        o = analytical_periodic_orbit(cfg=cfg, T_K=REF_T, Kmax_Pa_sqrt_m=REF_KMAX, R=REF_R,
                                       frequency_Hz=REF_F, r_contact_m=r_eff_m, n_phase=N_PHASE)
        passed = passivation_gate(o)
        rows.append({
            "row_family": "PASSIVATION_LIMITED", "candidate_label": f"depass={depass}_repass={repass}",
            "A_on_ref": "", "A_off_ref": "", "delta_rupture_barrier_eV": "",
            "depassivation_barrier_eV": depass, "repassivation_barrier_eV": repass,
            "mean_p_B": o["mean_p_B"], "swing_p_B": o["max_p_B"] - o["min_p_B"],
            "A_CB": o["A_CB"], "A_BC": o["A_BC"], "mean_p_P": o["mean_p_P"], "mean_p_C": o["mean_p_C"],
            "A_PC": o["A_PC"], "A_CP": o["A_CP"], "F_PC": o["F_PC"], "F_CP": o["F_CP"],
            "any_barrier_floored": any(o["any_floored"].values()),
            "any_cooperative_saturated": any(o["any_saturated"].values()),
            "gates_passed": passed,
            "selected": (depass == 0.40 and repass == 0.40),
        })

    with (OUT_DIR / "kinetic_regime_candidate_audit.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote kinetic_regime_candidate_audit.csv: {len(rows)} candidates across 3 row families")
    for family in ("COMPETING_REVERSIBLE", "COMPETING_PERSISTENT", "PASSIVATION_LIMITED"):
        n_passed = sum(1 for r in rows if r["row_family"] == family and r["gates_passed"])
        n_total = sum(1 for r in rows if r["row_family"] == family)
        selected = next(r["candidate_label"] for r in rows if r["row_family"] == family and r["selected"])
        print(f"  {family}: {n_passed}/{n_total} candidates passed all gates; selected = {selected}")


if __name__ == "__main__":
    main()
