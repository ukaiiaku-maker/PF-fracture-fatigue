#!/usr/bin/env python3
"""Storage-light original-state qualification for native Taylor--Peierls kinetics."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.persistent_runtime_contract import require_persistent_binding
from arrhenius_fracture.persistent_site_source_v10221 import (
    PersistentSiteConfig,
    install_persistent_site_source,
)
from arrhenius_fracture.signed_burgers_shared_v1025 import (
    KERNEL_SCHEMA,
    SignedShieldingKernel,
    install_signed_burgers_population,
)
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState

B_M = 2.74e-10
G_PA = 160.0e9
T_K = 950.0
STRESS_PA = 5.0e9
DT_S = 0.01
INITIAL_STEPS = 6
TOTAL_STEPS = 12
ARRAY_NAMES = (
    "mobile_positive", "mobile_negative", "retained_positive", "retained_negative",
    "accumulated_slip_positive", "accumulated_slip_negative", "wake_mobile_positive",
    "wake_mobile_negative", "wake_retained_positive", "wake_retained_negative",
    "wake_slip_positive", "wake_slip_negative",
)
SCALAR_NAMES = (
    "emitted_total", "escaped_total", "recovered_total", "net_retained_gain_total",
    "net_retained_loss_total", "gross_trapping_total", "gross_release_total",
    "mobile_distance_population_total_m", "advance_total_m",
    "wake_discarded_mobile_total", "wake_discarded_retained_total",
    "wake_discarded_slip_total", "time_s", "signed_source_activations_total",
    "signed_line_content_emitted_total",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_state(
    manifest_path: Path,
    *,
    mobile_recovery_rate_s: float = 0.0,
    retained_recovery_rate_s: float = 0.0,
) -> UnifiedMPZState:
    manifest = MaterialManifest.from_csv(manifest_path)
    state = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=8.0e-6, n_bins=16, n_systems=2, source_bin_count=2,
                  mobile_recovery_rate_s=0.0, wake_length_m=8.0e-6),
    )
    kernel = SignedShieldingKernel(
        active_kernel=np.ones((2, 16)), wake_kernel=np.zeros((2, 16)),
        active_x_m=state.x.copy(), wake_x_m=state.wake_x.copy(),
        activation_to_line_content=np.ones(2),
        source_capacity_bounds=np.array([[1.0, 1000.0], [1.0, 1000.0]]),
        metadata={"schema": KERNEL_SCHEMA},
        source_path="native-taylor-peierls-spatial-qualification",
    )
    install_signed_burgers_population(state, kernel, "validated_scalar")
    state._campaign_G_Pa = G_PA
    state._campaign_b = B_M
    state._campaign_backstress_scale = 1.0
    state._anisotropic_drive_reliable = True
    state._anisotropic_drive_factors = np.ones(2)
    state._anisotropic_tau_signed_Pa = np.ones(2)
    install_persistent_site_source(
        state,
        config=PersistentSiteConfig(rho_site0_m2=1.4115242646890916e16),
        r0_m=1.0e-6, b_m=B_M,
    )
    # Preserve the top-1 physical installation contract: its selected material
    # row has zero recovery. Nonzero values here are an explicit numerical
    # qualification overlay applied only after the native state and source law
    # are installed; they are recorded in every trace and metadata receipt.
    state.cfg.mobile_recovery_rate_s = max(float(mobile_recovery_rate_s), 0.0)
    state.manifest = replace(
        state.manifest,
        retained_recovery_rate_s=max(float(retained_recovery_rate_s), 0.0),
    )
    require_persistent_binding(state, stage="constructed")
    return state


def save_checkpoint(state: UnifiedMPZState, path: Path, step: int) -> None:
    payload: dict[str, Any] = {name: np.asarray(getattr(state, name)) for name in ARRAY_NAMES}
    payload.update({name: np.asarray(float(getattr(state, name, 0.0))) for name in SCALAR_NAMES})
    payload["step"] = np.asarray(step)
    np.savez(path, **payload)


def load_checkpoint(state: UnifiedMPZState, path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        for name in ARRAY_NAMES:
            setattr(state, name, data[name].copy())
        for name in SCALAR_NAMES:
            setattr(state, name, float(data[name]))
        step = int(data["step"])
    state.mobile = state.mobile_positive + state.mobile_negative
    state.retained = state.retained_positive + state.retained_negative
    state.accumulated_slip = (
        state.accumulated_slip_positive + state.accumulated_slip_negative
    )
    state.wake_mobile = state.wake_mobile_positive + state.wake_mobile_negative
    state.wake_retained = state.wake_retained_positive + state.wake_retained_negative
    state.wake_slip = state.wake_slip_positive + state.wake_slip_negative
    require_persistent_binding(state, stage="fresh-process-restart")
    return step


def trace(state: UnifiedMPZState, step: int, out: dict[str, Any]) -> dict[str, Any]:
    stress = state.local_stress_profile_Pa(STRESS_PA)
    rho = state.local_forest_density_m2(False)
    rates = state._transport_rates(stress, rho, T_K, B_M)
    tau_p = state.cfg.peierls_stress_fraction * stress
    tau_t = state.cfg.taylor_stress_fraction * stress * rates["jump"] / B_M
    p_surface = state.manifest.peierls.as_surface(state.manifest.emission)
    t_surface = state.manifest.taylor.as_surface(state.manifest.emission)
    mobile = state.mobile_positive + state.mobile_negative
    face_flux = mobile * (rates["velocity"] / state.dx)[None, :]
    total = np.sum(state.mobile + state.retained, axis=0)
    eq_mobile = total * np.divide(
        rates["taylor"], rates["encounter"] + rates["taylor"],
        out=np.zeros_like(total), where=(rates["encounter"] + rates["taylor"]) > 0.0,
    )
    scalar = (float(np.sum(rates["velocity"] * eq_mobile) / np.sum(eq_mobile))
              if np.sum(eq_mobile) > 0.0 else float(np.mean(rates["velocity"][:2])))
    return {
        "step": step, "time_s": step * DT_S, "temperature_K": T_K,
        "tip_opening_stress_Pa": STRESS_PA, "x_m": state.x.tolist(),
        "mobile_recovery_rate_s": float(state.cfg.mobile_recovery_rate_s),
        "retained_recovery_rate_s": float(state.manifest.retained_recovery_rate_s),
        "local_opening_stress_Pa": stress.tolist(), "forest_density_m2": rho.tolist(),
        "peierls_projected_stress_Pa": tau_p.tolist(),
        "taylor_projected_stress_Pa": tau_t.tolist(),
        "peierls_jump_length_m": rates["jump"].tolist(),
        "peierls_forward_barrier_eV": p_surface.values_eV(np.maximum(tau_p, 0.0), T_K).tolist(),
        "peierls_reverse_barrier_eV": p_surface.values_eV(np.maximum(-tau_p, 0.0), T_K).tolist(),
        "taylor_forward_barrier_eV": t_surface.values_eV(np.maximum(tau_t, 0.0), T_K).tolist(),
        "taylor_reverse_barrier_eV": t_surface.values_eV(np.maximum(-tau_t, 0.0), T_K).tolist(),
        "peierls_prefactor_s": state.manifest.peierls.attempt_frequency_s,
        "taylor_prefactor_s": state.manifest.taylor.attempt_frequency_s,
        "peierls_forward_rate_s": rates["peierls_forward"].tolist(),
        "peierls_reverse_rate_s": rates["peierls_reverse"].tolist(),
        "peierls_net_rate_s": rates["peierls"].tolist(),
        "taylor_forward_rate_s": rates["taylor_forward"].tolist(),
        "taylor_reverse_rate_s": rates["taylor_reverse"].tolist(),
        "taylor_net_elementary_rate_s": rates["taylor_single"].tolist(),
        "taylor_density_order": rates["m"].tolist(),
        "taylor_completion_rate_s": rates["taylor"].tolist(),
        "velocity_m_s_by_bin": rates["velocity"].tolist(),
        "face_flux_population_s_by_system_bin": face_flux.tolist(),
        "diagnostic_scalar_velocity_m_s": scalar,
        "emitted_this_interval": float(out["dN_emit"]),
        "emitted_total": state.emitted_total,
        "mobile_total": state.mobile_count, "retained_total": state.retained_count,
        "coefficient_residual": float(out["coupled_source_coefficient_residual"]),
        "state_fixed_point_residual": float(out["coupled_source_state_residual"]),
        "complementarity_residual": float(
            out["coupled_source_complementarity_residual"]),
        "affine_direct_state_residual": float(
            out["coupled_source_affine_direct_residual"]),
        "transport_invariant_roundoff_correction": float(
            out["transport_invariant_roundoff_correction"]),
        "transport_base_map_correction": float(
            out["transport_base_map_correction"]),
        "transport_positivity_correction": float(
            out["transport_positivity_correction"]),
        "transport_complete_numerical_correction": float(
            out["transport_complete_numerical_correction"]),
        "transport_affine_invariant_defect": float(
            out["transport_affine_invariant_defect"]),
        "transport_affine_retry_discrepancy": float(
            out["transport_affine_retry_discrepancy"]),
        "transport_error_controlled_stiff_solve": bool(
            out["transport_error_controlled_stiff_solve"]),
        "transport_stiff_solver_method": str(
            out["transport_stiff_solver_method"]),
        "escaped_total": state.escaped_total,
        "recovered_total": state.recovered_total,
        "gross_trapping_total": state.gross_trapping_total,
        "gross_release_total": state.gross_release_total,
        "mobile_distance_population_total_m": state.mobile_distance_population_total_m,
        "population_ledger_balance_error": float(
            out["population_ledger_balance_error"]),
        "taylor_release_mode": rates["taylor_release_mode"],
        "taylor_release_window_s": rates["taylor_release_window_s"],
        "taylor_single_rate_cap_s": rates["taylor_single_rate_cap_s"],
    }


def evolve(state: UnifiedMPZState, start: int, stop: int) -> list[dict[str, Any]]:
    records = []
    for step in range(start + 1, stop + 1):
        require_persistent_binding(state, stage=f"before-step-{step}")
        out = state.evolve(DT_S, T_K, STRESS_PA, B_M)
        accounted = (
            state.mobile_count + state.retained_count
            + state.wake_mobile_count + state.wake_retained_count
            + state.escaped_total + state.recovered_total
            + state.wake_discarded_mobile_total + state.wake_discarded_retained_total
        )
        balance = accounted - state.emitted_total
        if abs(balance) > 2.0e-8 * max(state.emitted_total, 1.0):
            raise RuntimeError(
                f"owned population/flux ledger imbalance at step {step}: {balance:.6e}"
            )
        out["population_ledger_balance_error"] = balance
        records.append(trace(state, step, out))
        require_persistent_binding(state, stage=f"after-step-{step}")
    return records


def run(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    phase_dir = root / args.phase
    phase_dir.mkdir(exist_ok=True)
    state = build_state(
        manifest,
        mobile_recovery_rate_s=args.mobile_recovery_rate_s,
        retained_recovery_rate_s=args.retained_recovery_rate_s,
    )
    if args.phase == "initial":
        records = evolve(state, 0, INITIAL_STEPS)
        save_checkpoint(state, root / "restart_checkpoint_step6.npz", INITIAL_STEPS)
    elif args.phase == "resume":
        start = load_checkpoint(state, root / "restart_checkpoint_step6.npz")
        records = evolve(state, start, TOTAL_STEPS)
    elif args.phase == "uninterrupted":
        records = evolve(state, 0, TOTAL_STEPS)
    else:
        raise ValueError(args.phase)
    save_checkpoint(state, phase_dir / "terminal_state.npz", TOTAL_STEPS if args.phase != "initial" else INITIAL_STEPS)
    (phase_dir / "trace.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (phase_dir / "runtime_binding.json").write_text(
        json.dumps(require_persistent_binding(state, stage=f"{args.phase}-terminal"), indent=2, sort_keys=True) + "\n"
    )
    metadata = {
        "phase": args.phase, "source_commit": _commit(), "manifest": str(manifest),
        "manifest_sha256": _sha(manifest), "temperature_K": T_K, "stress_Pa": STRESS_PA,
        "dt_s": DT_S, "initial_steps": INITIAL_STEPS, "total_steps": TOTAL_STEPS,
        "mobile_recovery_rate_s": float(state.cfg.mobile_recovery_rate_s),
        "retained_recovery_rate_s": float(state.manifest.retained_recovery_rate_s),
        "recovery_input_classification": "post-install numerical qualification overlay",
        "selected_material_recovery_unchanged": True,
        "production_execution_authorized": False,
    }
    (phase_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def compare(root: Path) -> None:
    left = root / "resume" / "terminal_state.npz"
    right = root / "uninterrupted" / "terminal_state.npz"
    differences = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys_equal = set(a.files) == set(b.files)
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                differences[key] = "missing"
            elif not np.array_equal(a[key], b[key]):
                differences[key] = float(np.max(np.abs(a[key] - b[key])))
    traces = json.loads((root / "uninterrupted" / "trace.json").read_text())
    max_record = max(traces, key=lambda item: max(item["velocity_m_s_by_bin"]))
    terminal = traces[-1]
    metadata = json.loads((root / "uninterrupted" / "metadata.json").read_text())
    (root / "maximum_velocity_trace.json").write_text(json.dumps(max_record, indent=2, sort_keys=True) + "\n")
    (root / "terminal_velocity_trace.json").write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    report = {
        "schema": "native-taylor-peierls-spatial-qualification/1",
        "status": "PASSED" if keys_equal and not differences else "FAILED",
        "fresh_process_restart": True, "array_keys_equal": keys_equal,
        "bitwise_terminal_state_equal": not differences, "differences": differences,
        "arrays_and_scalars_compared": len(np.load(left, allow_pickle=False).files),
        "maximum_velocity_m_s": max(max_record["velocity_m_s_by_bin"]),
        "terminal_velocity_m_s": max(terminal["velocity_m_s_by_bin"]),
        "emitted_total": terminal["emitted_total"],
        "max_invariant_roundoff_correction": max(
            item["transport_invariant_roundoff_correction"] for item in traces),
        "max_base_map_correction": max(
            item["transport_base_map_correction"] for item in traces),
        "max_positivity_correction": max(
            item["transport_positivity_correction"] for item in traces),
        "max_complete_numerical_correction": max(
            item["transport_complete_numerical_correction"] for item in traces),
        "max_affine_invariant_defect": max(
            item["transport_affine_invariant_defect"] for item in traces),
        "max_affine_retry_discrepancy": max(
            item["transport_affine_retry_discrepancy"] for item in traces),
        "error_controlled_stiff_solve_used": any(
            item["transport_error_controlled_stiff_solve"] for item in traces),
        "stiff_solver_methods": sorted(set(
            item["transport_stiff_solver_method"] for item in traces)),
        "max_coefficient_residual": max(item["coefficient_residual"] for item in traces),
        "max_complementarity_residual": max(item["complementarity_residual"] for item in traces),
        "max_population_ledger_balance_error": max(
            abs(item.get("population_ledger_balance_error", 0.0)) for item in traces),
        "terminal_escaped_total": terminal["escaped_total"],
        "terminal_recovered_total": terminal["recovered_total"],
        "mobile_recovery_rate_s": metadata["mobile_recovery_rate_s"],
        "retained_recovery_rate_s": metadata["retained_recovery_rate_s"],
        "terminal_gross_trapping_total": terminal["gross_trapping_total"],
        "terminal_gross_release_total": terminal["gross_release_total"],
        "terminal_mobile_distance_population_total_m": terminal[
            "mobile_distance_population_total_m"],
        "scope": (
            "prescribed-drive native subsystem integration fixture with a synthetic "
            "shielding kernel; explicit removal/recovery is distinct from Taylor "
            "retained-to-mobile release and sign-reversing source-surface return; no "
            "specimen equilibrium solve and not a production campaign"
        ),
    }
    (root / "restart_comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "PASSED":
        raise SystemExit("restart comparison failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("initial", "resume", "uninterrupted", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(
        "integration_evidence/f1_coupled_native_qualification_8982d422/initial_to_step48/selected_material_manifest_v10_2_22.csv"))
    parser.add_argument("--mobile-recovery-rate-s", type=float, default=0.0)
    parser.add_argument("--retained-recovery-rate-s", type=float, default=0.0)
    args = parser.parse_args()
    compare(args.output.resolve()) if args.phase == "compare" else run(args)


if __name__ == "__main__":
    main()
