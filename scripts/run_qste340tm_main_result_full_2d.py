#!/usr/bin/env python3
"""Run selected QSTE340TM main-result points through the native 2-D solver.

This runner does not call the analytical/state-reduced QSTE evaluators.  It
binds the selected opening surface and EXP-floor plasticity to the existing
spatial FEM/J-integral, cyclic-mechanics, moving-MPZ, shielding and blunting
owners in :mod:`arrhenius_fracture.sharp_front`.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.canonical_v2_registry_v10230 import ThermodynamicBarrierAdapter
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from arrhenius_fracture.qste340tm_restored_binding import (
    PlasticCoordinates,
    load_parent_binding,
    material_binding,
)
from arrhenius_fracture import sharp_front
from scripts.calibrate_qste340tm_opening_reference import (
    ATTEMPT_FREQUENCY_S,
    COOPERATIVE_HITS,
    COOPERATIVE_TAU_S,
    EVENT_LENGTH_M,
    NOMINAL_FREQUENCY_HZ,
    SMOOTHING_RATIO,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import KineticSurface, SmoothBroadBasis


DATA = ROOT / "literature_data/fatigue_recovered_review/model_conditional"
MAIN = DATA / "emission_owned_blunting/MAIN_RESULT_MANIFEST.json"
RELEASED_SLOPING_FIT = DATA / (
    "emission_owned_blunting/resolved_increment_1d/"
    "released_sloping_barrier_search/fit.json"
)
REFERENCE = DATA / "conditional_opening_reference.json"
ALT_REGISTRY = ROOT / "integration_evidence/r10_alternative_gstar_joint_search/alternative_gstar_registry.json"
PARENT_REGISTRY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
    "v3-fatigue-informed-joint-barrier-search/analysis_outputs/"
    "v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json"
)
PARENT_CANDIDATE = "R10_ALT_GSTAR_02506"
TEMPERATURE_K = 298.15
STEEL_E_PA = 208.0e9
STEEL_NU = 0.30
STEEL_B_M = 2.48e-10
# Same-mesh full 2-D pilots bracketed the nonlinear boundary compliance:
# factor=1 -> K_J=0.646 and factor=5.3 -> K_J=25.466 MPa sqrt(m) for the
# 20 MPa sqrt(m) R=0.1 target.  The power interpolation is used only to select
# a boundary displacement; every accepted case reports the actual FEM/J K.
COMPLIANCE_REFERENCE_K = 20.0
COMPLIANCE_REFERENCE_OBSERVED_K = 0.646
COMPLIANCE_EXPONENT = math.log(25.466 / 0.646) / math.log(5.3)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_manifest(fit_path: Path | None = None):
    source_fit = Path(fit_path) if fit_path is not None else MAIN
    selected = json.loads(source_fit.read_text())
    p = selected["parameters"]
    reference = json.loads(REFERENCE.read_text())["nominal_fit"]["parameters"]
    parent = load_parent_binding(
        ALT_REGISTRY, PARENT_REGISTRY, candidate_id=PARENT_CANDIDATE,
    )
    plastic = PlasticCoordinates(
        emission_G0_298_eV=float(p["emission_G0_298_eV"]),
        plastic_sigc_298_Pa=float(p["plastic_sigc_298_Pa"]),
        peierls_G0_298_eV=float(p["transport_G0_298_eV"]),
        taylor_G0_298_eV=float(p["transport_G0_298_eV"]),
    )
    bound = material_binding(parent, plastic)
    kbt = KB_EV_PER_K * TEMPERATURE_K
    opening_amplitude = float(p.get(
        "opening_amplitude_over_kBT",
        reference["g_amplitude_Gstar_over_kBT"],
    ))
    opening_sigma = float(p.get("opening_sigma_b_Pa", reference["sigma_b_Pa"]))
    opening_exponent = float(p.get("opening_exponent", reference["exponent"]))
    opening_smoothing = float(p.get("opening_smoothing_ratio", SMOOTHING_RATIO))
    surface = KineticSurface(
        process="cleavage",
        attempt_frequency_s=ATTEMPT_FREQUENCY_S,
        reference_floor_eV=float(p["opening_floor_over_kBT"]) * kbt,
        reference_amplitudes_eV=(
            opening_amplitude * kbt,
        ),
        bases=(SmoothBroadBasis(
            sigma_b_Pa=opening_sigma,
            exponent=opening_exponent,
            smoothing_ratio=opening_smoothing,
        ),),
        entropy_infinity_kB=0.0,
        entropy_amplitudes_kB=(0.0,),
        heat_capacity_infinity_kB=0.0,
        heat_capacity_amplitudes_kB=(0.0,),
        hits=COOPERATIVE_HITS,
        tau_s=COOPERATIVE_TAU_S,
    )
    cleavage = ThermodynamicBarrierAdapter(
        surface=surface, parent=dict(bound.manifest.cleavage.parent),
    )
    manifest = replace(
        bound.manifest,
        name=("QSTE340TM_RESOLVED_SLOPING_BARRIER_2D" if fit_path is not None
              else "QSTE340TM_EMISSION_OWNED_BLUNTING_MAIN_2D"),
        candidate_id=("QSTE340TM_RESOLVED_SLOPING_20261027" if fit_path is not None
                      else "QSTE340TM_EOB_MAIN_20261026"),
        cleavage=cleavage,
        retained_recovery_rate_s=float(p["retained_recovery_per_s"]),
        # Native accumulated_slip is emitted-source slip.  Forward passage
        # remains an independently reported comparison discrepancy; it is not
        # folded into an empirical multiplier inside the full spatial solve.
        c_blunt=float(p["source_slip_blunting_coefficient"]),
    )
    reference = dict(reference) | {
        "g_amplitude_Gstar_over_kBT": opening_amplitude,
        "sigma_b_Pa": opening_sigma,
        "exponent": opening_exponent,
        "smoothing_ratio": opening_smoothing,
        "source_fit": str(source_fit),
    }
    return selected, reference, manifest


def displacement_for_kmax(kmax_mpa_sqrt_m: float) -> float:
    """LEFM starting estimate; the executed 2-D J field reports the actual K."""
    a0 = 0.5e-3
    height = 4.0e-3
    eprime = STEEL_E_PA / (1.0 - STEEL_NU**2)
    unit_estimate = 1.0e6 * height / (eprime * math.sqrt(math.pi * a0))
    relative_displacement = (
        float(kmax_mpa_sqrt_m) / COMPLIANCE_REFERENCE_OBSERVED_K
    ) ** (1.0 / COMPLIANCE_EXPONENT)
    return float(unit_estimate * COMPLIANCE_REFERENCE_K * relative_displacement)


def case_namespace(
    out: Path, R: float, delta_k: float, cycles: float, steps: int,
    *, adaptive: bool, block_cycles: float, compact: bool, target_dB: float,
    mpz_length_um: float, blunting_length_um: float,
    forward_blunting_coefficient: float,
    tip_only_plasticity: bool, analytical_source_slip_ledger: bool,
    state_increment_target: float, loading_control: str,
    passage_count_normalization: float,
    incremental_opening_advection: bool = False,
    opening_advection_step_um: float = 0.1,
    opening_advection_duration_s: float | None = None,
    opening_advection_capture_width_um: float | None = None,
    opening_advection_retained_fraction: float | None = None,
    opening_advection_state_owner: str = "unsigned_density",
    signed_gnd_tip_absorption: bool = False,
    tip_h_fine_m: float = 1.0e-6,
):
    parser = sharp_front._build_parser()
    kmax = float(delta_k) / (1.0 - float(R))
    argv = [
        "--mode", "2d", "--temperatures", str(TEMPERATURE_K),
        "--out", str(out), "--front-state-model", "moving_pz",
        "--fatigue-cycles", "--fatigue-hold-load",
        "--R", str(R), "--frequency-Hz", str(NOMINAL_FREQUENCY_HZ),
        "--multihit-m", str(COOPERATIVE_HITS),
        "--multihit-tau", str(COOPERATIVE_TAU_S),
        "--cycles-max", str(cycles), "--block-cycles", str(block_cycles),
        "--max-block-cycles", str(cycles), "--target-dB", str(target_dB),
        "--target-dN-store", str(state_increment_target),
        "--target-dN-emit", str(state_increment_target),
        "--target-dN-mobile", str(state_increment_target),
        "--target-dN-escape", str(state_increment_target),
        "--target-dN-peierls", "inf", "--target-dN-taylor", "inf",
        "--n-phase", "48",
        "--cyclic-mechanics-phases", "16", "--cyclic-mechanics-stagger", "2",
        "--steps", str(steps), "--dU", str(displacement_for_kmax(kmax)),
        "--nx", "90", "--ny", "150", "--tip-h-fine", str(tip_h_fine_m),
        "--tip-ratio", "1.18", "--n-stagger", "2",
        "--r-pz", "1e-6", "--L-pz", str(float(mpz_length_um) * 1.0e-6),
        "--mpz-length-um", str(mpz_length_um), "--mpz-n-bins", "200",
        "--mpz-source-bins", "2", "--mpz-blunting-length-um", str(blunting_length_um),
        "--mpz-forward-slip-blunting-coefficient", str(forward_blunting_coefficient),
        "--mpz-passage-count-normalization", str(passage_count_normalization),
        "--da-phys", str(EVENT_LENGTH_M), "--save-snapshots", "3",
        "--print-every", "1",
    ]
    argv += ["--fatigue-loading-control", str(loading_control)]
    if loading_control in {"fixed_local_kmax", "fixed_reaction",
                           "calibrated_fixed_displacement"}:
        argv += ["--fatigue-target-Kmax-MPa-sqrt-m", str(kmax)]
    if adaptive:
        argv += ["--cycle-block-mode", "hazard_limited"]
    else:
        argv += ["--no-adaptive-cycles", "--cycle-block-mode", "requested_cap"]
    if compact:
        argv += ["--no-plots", "--save-snapshots", "0"]
    if tip_only_plasticity:
        argv += ["--tip-only-plasticity"]
    if analytical_source_slip_ledger:
        argv += ["--mpz-analytical-source-slip-ledger"]
    if incremental_opening_advection:
        argv += ["--incremental-opening-advection",
                 "--opening-advection-step-um", str(opening_advection_step_um),
                 "--opening-advection-state-owner", str(opening_advection_state_owner)]
        if opening_advection_duration_s is not None:
            argv += ["--opening-advection-duration-s", str(opening_advection_duration_s)]
        if opening_advection_capture_width_um is not None:
            argv += ["--opening-advection-capture-width-um",
                     str(opening_advection_capture_width_um)]
        if opening_advection_retained_fraction is not None:
            argv += ["--opening-advection-retained-fraction",
                     str(opening_advection_retained_fraction)]
        if signed_gnd_tip_absorption:
            argv += ["--signed-gnd-tip-absorption"]
    return parser.parse_args(argv), kmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "emission_owned_blunting/full_2d_comparison")
    ap.add_argument("--delta-k", type=float, default=18.0)
    ap.add_argument("--R", type=float, nargs="+", default=(0.1, 0.5, 0.7))
    ap.add_argument("--cycles", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--block-cycles", type=float, default=None)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--events", type=int, default=0)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--target-dB", type=float, default=0.10)
    ap.add_argument("--mpz-length-um", type=float, default=100.0,
                    help="Active transport/process-zone length. Historical native control is 100; analytical QSTE fit used 1.")
    ap.add_argument("--blunting-pathway", choices=("source_only", "source_plus_forward"),
                    default="source_only")
    ap.add_argument("--plasticity-scope", choices=("tip_only", "tip_plus_bulk"),
                    default="tip_plus_bulk")
    ap.add_argument("--blunting-ledger-mode", choices=("spatial_source", "analytical_equivalent"),
                    default="spatial_source")
    ap.add_argument("--state-increment-target", type=float, default=float("inf"),
                    help="Adaptive per-block target for emitted/mobile/stored/escaped MPZ counts.")
    ap.add_argument("--loading-control",
                    choices=("fixed_displacement", "calibrated_fixed_displacement",
                             "fixed_local_kmax", "fixed_reaction"),
                    default="fixed_displacement",
                    help="Historical fixed displacement, J-controlled primary-tip Kmax, or constant top-reaction amplitude calibrated at the requested initial Kmax.")
    ap.add_argument("--normalize-passage-by-source-capacity", action="store_true",
                    help="Convert gross two-system native passage counts to the normalized analytical source coordinate before applying fitted blunting coefficients.")
    ap.add_argument("--incremental-opening-advection", action="store_true")
    ap.add_argument("--opening-advection-step-um", type=float, default=0.1)
    ap.add_argument("--opening-advection-duration-s", type=float, default=None)
    ap.add_argument("--opening-advection-capture-width-um", type=float, default=None)
    ap.add_argument("--opening-advection-retained-fraction", type=float, default=None)
    ap.add_argument("--opening-advection-state-owner",
                    choices=("unsigned_density", "signed_gnd"),
                    default="unsigned_density")
    ap.add_argument("--signed-gnd-tip-absorption", action="store_true",
                    help="Book signed crack-swept GND content at the tip and apply its opening projection through the existing c_blunt law.")
    ap.add_argument("--tip-h-fine-m", type=float, default=1.0e-6)
    ap.add_argument("--material-fit", type=Path, default=None,
                    help="Optional complete QSTE fit JSON; default preserves the historical MAIN_RESULT_MANIFEST binding.")
    ns = ap.parse_args()
    selected, reference, manifest = selected_manifest(ns.material_fit)
    ns.out.mkdir(parents=True, exist_ok=True)
    campaign = {
        "schema": "qste340tm-main-result-full-2d/1",
        "main_result_sha256": sha256(ns.material_fit or MAIN),
        "material_fit": str(ns.material_fit or MAIN),
        "source_head": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "temperature_K": TEMPERATURE_K,
        "frequency_Hz": NOMINAL_FREQUENCY_HZ,
        "deltaK_MPa_sqrt_m": float(ns.delta_k),
        "boundary_compliance_exponent": COMPLIANCE_EXPONENT,
        "boundary_compliance_source": "same-mesh full-2D nonlinear displacement bracket; actual K_J retained",
        "full_spatial_owners": [
            "finite-element displacement/stress solve", "J-integral local drive",
            "phase-resolved cyclic mechanics", "moving MPZ transport",
            "signed spatial K shielding", "emission-owned accumulated-slip blunting",
        ],
        "model_reduction_used": False,
        "forward_slip_mapping": "not fitted into native c_blunt; retained as comparison discrepancy",
        "cases": [],
    }
    forward_coefficient = (
        float(selected["parameters"]["forward_slip_blunting_coefficient"])
        if ns.blunting_pathway == "source_plus_forward" else 0.0
    )
    blunting_length_um = 1.0e6 * float(
        selected["parameters"]["blunting_length_m"]
    )
    source_capacity = float(manifest.source_sites_per_system) * 2.0
    passage_count_normalization = (
        source_capacity if ns.normalize_passage_by_source_capacity else 1.0
    )
    campaign.update({
        "blunting_pathway": ns.blunting_pathway,
        "source_slip_blunting_coefficient": float(manifest.c_blunt),
        "forward_slip_blunting_coefficient": forward_coefficient,
        "blunting_length_um": blunting_length_um,
        "mpz_length_um": float(ns.mpz_length_um),
        "driver_L_pz_m": float(ns.mpz_length_um) * 1.0e-6,
        "analytical_process_zone_length_um": 1.0,
        "plasticity_scope": ns.plasticity_scope,
        "bulk_element_plasticity_enabled": ns.plasticity_scope == "tip_plus_bulk",
        "blunting_ledger_mode": ns.blunting_ledger_mode,
        "state_increment_target": float(ns.state_increment_target),
        "loading_control": ns.loading_control,
        "target_local_Kmax_is_recomputed_after_plasticity": ns.loading_control == "fixed_local_kmax",
        "reaction_amplitude_is_held_after_initial_K_calibration": ns.loading_control == "fixed_reaction",
        "displacement_amplitude_is_held_after_initial_K_calibration": ns.loading_control == "calibrated_fixed_displacement",
        "cooperative_hits": COOPERATIVE_HITS,
        "cooperative_tau_s": COOPERATIVE_TAU_S,
        "passage_count_normalization": passage_count_normalization,
        "native_total_source_capacity": source_capacity,
        "incremental_opening_advection": bool(ns.incremental_opening_advection),
        "opening_advection_step_um": float(ns.opening_advection_step_um),
        "opening_advection_duration_s": ns.opening_advection_duration_s,
        "opening_advection_capture_width_um": ns.opening_advection_capture_width_um,
        "opening_advection_retained_fraction": ns.opening_advection_retained_fraction,
        "opening_advection_state_owner": ns.opening_advection_state_owner,
        "signed_gnd_tip_absorption": bool(ns.signed_gnd_tip_absorption),
        "tip_h_fine_m": float(ns.tip_h_fine_m),
    })
    campaign["forward_slip_mapping"] = (
        "gross mobile passage through the forward MPZ boundary; moving-tip ledger convected exponentially on crack advance"
        if forward_coefficient > 0.0 else "disabled source-only control"
    )
    for r_value in ns.R:
        label = f"R_{r_value:.1f}".replace(".", "p")
        out = ns.out / label
        block_cycles = (
            float(ns.block_cycles) if ns.block_cycles is not None
            else float(ns.cycles) / max(int(ns.steps), 1)
        )
        args, kmax = case_namespace(
            out, r_value, ns.delta_k, ns.cycles, ns.steps,
            adaptive=bool(ns.adaptive), block_cycles=block_cycles,
            compact=bool(ns.compact), target_dB=float(ns.target_dB),
            mpz_length_um=float(ns.mpz_length_um),
            blunting_length_um=blunting_length_um,
            forward_blunting_coefficient=forward_coefficient,
            tip_only_plasticity=ns.plasticity_scope == "tip_only",
            analytical_source_slip_ledger=(
                ns.blunting_ledger_mode == "analytical_equivalent"
            ),
            state_increment_target=float(ns.state_increment_target),
            loading_control=str(ns.loading_control),
            passage_count_normalization=passage_count_normalization,
            incremental_opening_advection=bool(ns.incremental_opening_advection),
            opening_advection_step_um=float(ns.opening_advection_step_um),
            opening_advection_duration_s=ns.opening_advection_duration_s,
            opening_advection_capture_width_um=ns.opening_advection_capture_width_um,
            opening_advection_retained_fraction=ns.opening_advection_retained_fraction,
            opening_advection_state_owner=ns.opening_advection_state_owner,
            signed_gnd_tip_absorption=bool(ns.signed_gnd_tip_absorption),
            tip_h_fine_m=float(ns.tip_h_fine_m),
        )
        args.material_manifest_object = manifest
        args.elastic_E_Pa = STEEL_E_PA
        args.elastic_nu = STEEL_NU
        args.burgers_m = STEEL_B_M
        args.mpz_initial_mobile_count = float(selected["parameters"]["initial_mobile_count"])
        args.mpz_initial_retained_count = float(selected["parameters"]["initial_retained_count"])
        receipt = {
            "R": float(r_value), "target_DeltaK_MPa_sqrt_m": float(ns.delta_k),
            "target_Kmax_MPa_sqrt_m": kmax, "initial_displacement_m": float(args.dU),
            "output": str(out),
        }
        campaign["cases"].append(receipt)
        (ns.out / "CAMPAIGN.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
        old_stop = os.environ.get("V10230_STOP_AFTER_COMMITTED_EVENTS")
        try:
            if ns.events > 0:
                os.environ["V10230_STOP_AFTER_COMMITTED_EVENTS"] = str(ns.events)
            else:
                os.environ.pop("V10230_STOP_AFTER_COMMITTED_EVENTS", None)
            sharp_front.run_2d(args)
        finally:
            if old_stop is None:
                os.environ.pop("V10230_STOP_AFTER_COMMITTED_EVENTS", None)
            else:
                os.environ["V10230_STOP_AFTER_COMMITTED_EVENTS"] = old_stop
        receipt["status"] = "completed"
        (ns.out / "CAMPAIGN.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
