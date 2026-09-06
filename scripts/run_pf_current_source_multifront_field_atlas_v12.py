#!/usr/bin/env python3
"""Launch one or queue all eight pinned V12 field-atlas trajectories."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
BASE_COMMIT = "21a37787968b4922b8cab30847e3493df830b707"
BASE_TREE = "48277a1ed8941c163187ab261b91c86f9ec40417"
RECORDED_PHYSICAL_ENTRY_BLOB = "6b890e6c8def2376d948cbdfeec4e93e88598787"
BRANCH = "codex/v12-field-atlas-recovery"
CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
FAMILY_SHA256 = "f30cc083e2f284bb043170ea69f5501cf2aeca43dd71f3e477750c995f4af62e"
FAMILY_PHYSICS = "9f005522688c668616f27308805812af583ff3d73b0d446eb4de6f6c57cee954"
MECHANICAL_SHA256 = "3ccc690c103f96da8eca7ee4bc67272653bd888b402bb0878c1b1f6eb9dee6f9"
MECHANICAL_FINGERPRINT = "adb7754436a66542a38c17d671bc62639939d85075168a5db721b93b791e87d0"
ROWS = {
    "Peak": ("v913_zeroD_sobol_0242980", "v913_paper_peak01_0242980_persistent_sites"),
    "DBTT": ("v913_zeroD_sobol_0202500", "v913_paper_dbtt01_0202500_persistent_sites"),
    "weakT": ("oneD_v2_focused_weak_T_0016", "v913_paper_weakT01_0129902_persistent_sites"),
    "ceramic": ("oneD_v2_focused_ceramic_like_0018", "v913_paper_ceramic01_0077080_persistent_sites"),
}
CASES = tuple(f"{material}_{temperature}K" for material in ROWS for temperature in (300, 1000))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def case_parts(case: str) -> tuple[str, int, str, str]:
    if case not in CASES:
        raise ValueError(f"unsupported case {case!r}")
    material, temperature = case.rsplit("_", 1)
    canonical, alias = ROWS[material]
    return material, int(temperature[:-1]), canonical, alias


def registry_row(canonical: str) -> tuple[dict[str, str], str]:
    path = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv"
    with path.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["candidate_id"] == canonical]
    if len(matches) != 1:
        raise RuntimeError(f"registry must contain exactly one row for {canonical}")
    encoded = json.dumps(matches[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return matches[0], hashlib.sha256(encoded).hexdigest()


def protected_source_audit() -> dict:
    physical = "arrhenius_fracture/sharp_front_current_source_multifront_physical_v12.py"
    current_blob = subprocess.check_output(
        ("git", "hash-object", physical), cwd=ROOT, text=True
    ).strip()
    if current_blob != RECORDED_PHYSICAL_ENTRY_BLOB:
        raise RuntimeError("recovered V12 physical executable differs from its archived Git blob")
    return {
        "archived_executable_source_commit": BASE_COMMIT,
        "archived_executable_source_tree": BASE_TREE,
        "physical_entry_blob": current_blob,
        "campaign_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "campaign_tree": subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=ROOT, text=True).strip(),
        "endpoint_search_correction": "convergence-derived iteration budget; tolerances unchanged",
    }


def validate_inputs(family: Path) -> dict:
    from arrhenius_fracture.kernel_registry_v10227 import validate_family
    family = family.resolve()
    mechanics = family.parent / "mechanical_configuration.json"
    if sha256(family) != FAMILY_SHA256 or sha256(mechanics) != MECHANICAL_SHA256:
        raise RuntimeError("family or mechanical-configuration SHA-256 mismatch")
    audit = validate_family(
        family, expected_configuration_fingerprint=MECHANICAL_FINGERPRINT,
        expected_physics_fingerprint=FAMILY_PHYSICS,
    )
    mechanical = json.loads(mechanics.read_text())
    if mechanical.get("temperature_dependent_mechanics") is not False or mechanical.get("temperature_K") is not None:
        raise RuntimeError("pinned family is not explicitly temperature-independent")
    if audit["maximum_extension_um"] < 1600.0:
        raise RuntimeError("owner-local family does not cover 1600 um")
    return {"family_validation": audit, "mechanical_configuration": mechanical}


def common_arguments(alias: str, temperature: int, family: Path, output: Path) -> list[str]:
    return [
        "--signed-kernel-family", str(family), "--mode", "2d",
        "--parameter-option", alias, "--temperatures", str(temperature),
        "--steps", "2000000", "--nx", "36", "--ny", "72",
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "2",
        "--tip-h-fine", "1e-6", "--tip-ratio", "1.2", "--da-phys", "5e-6",
        "--target-crack-extension-um", "1000", "--mpz-length-um", "50",
        "--mpz-n-bins", "80", "--front-state-model", "moving_pz",
        "--tip-source-model", "continuum", "--tip-kinetics-mode", "moving_velocity",
        "--bulk-plasticity-mode", "tip_only", "--directional-j-mode", "root_signed",
        "--tip-plasticity", "--active-shielding", "--signed-active-shielding",
        "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "40",
        "--crystal-material", "w", "--j-decomposition", "cluster",
        "--crack-backend", "sharp_wake", "--adaptive-events",
        "--adaptive-event-target", "0.15", "--print-every", "200",
        "--save-snapshots", "0", "--no-plots", "--out", str(output),
    ]


def campaign_environment(family: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "V12_STOP_AFTER_TOTAL_BIRTHS", "PF_QUALIFIED_DAUGHTER_STOP_UM",
        "V12_SOURCE_V11_CHECKPOINT", "V12_SOURCE_ACCEPTED_CHECKPOINT",
        "V12_TARGET_ADDED_EXTENSION_UM", "V12_TARGET_MAXIMUM_FORWARD_REACH_UM",
    ):
        environment.pop(name, None)
    environment.update({
        "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "PARAMETER_CAMPAIGN": "1", "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": "3621", "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5", "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1", "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1", "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "MECHANICAL_CONFIG": str(family.parent / "mechanical_configuration.json"),
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0", "ONED_V2_TP_STATE_DIAGNOSTICS": "events",
        "MPLCONFIGDIR": "/private/tmp/pf-current-source-v12-field-atlas-mpl",
    })
    return environment


def build_fresh_initial(case: str, alias: str, temperature: int, family: Path, output: Path) -> Path:
    """Use the pinned paper entry stack to construct a zero-time V11 state, then seal it as V12."""
    from arrhenius_fracture import sharp_front as base
    from arrhenius_fracture import sharp_front_v10_2_27 as paper
    from arrhenius_fracture import sharp_front_v10_2_28_audited as entry
    from arrhenius_fracture.branch_checkpoint_v11 import ProductionBranchCheckpoint, write_branch_checkpoint
    from arrhenius_fracture.branch_scale_identity_v11 import resolve_branch_scale_identity
    from arrhenius_fracture.crack_network_v11 import CrackNetworkState
    from arrhenius_fracture.directional_competition_v11 import DirectionalCompetitionState, tungsten_cleavage_candidates
    from arrhenius_fracture.fem import assemble_mechanics, plane_strain_D
    from arrhenius_fracture.general_multifront_v12 import ResourcePolicy
    from arrhenius_fracture.live_topology_runtime_v11 import LiveTopologyRuntime
    from arrhenius_fracture.mesh import make_boundary_data, make_tri_mesh
    from arrhenius_fracture.multifront_checkpoint_v12 import load_v11_checkpoint_as_v12, write_accepted_boundary_checkpoint_v12
    from arrhenius_fracture.multifront_field_atlas_v12 import export_snapshot
    from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine, _hash, _mesh_identity
    from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState

    initial_root = output / "initial_setup"
    v11_path = initial_root / "checkpoint" / "step0000000.json"
    v12_path = output / "checkpoint" / "latest.v12.pkl"
    if v12_path.is_file():
        return v12_path
    result = {}

    def create(args):
        cfg = base.make_emergent_config()
        cfg.mesh.nx = args.nx; cfg.mesh.ny = args.ny
        cfg.mesh.tip_h_fine = float(args.tip_h_fine or 0.0)
        cfg.mesh.tip_ratio = float(args.tip_ratio)
        cfg.loading.n_steps = args.steps; cfg.loading.dU_top = args.dU; cfg.loading.dt = args.dt
        mat = cfg.material
        mesh = make_tri_mesh(cfg.geometry, cfg.mesh, seed=42)
        boundary = make_boundary_data(mesh, cfg.geometry)
        from arrhenius_fracture.crystal import W_C11, W_C12, W_C44, cubic_plane_strain_D
        D = cubic_plane_strain_D(
            float(args.crystal_C11 or W_C11), float(args.crystal_C12 or W_C12),
            float(args.crystal_C44 or W_C44), float(args.crystal_theta_deg or 0.0),
        ) if args.crystal_aniso else plane_strain_D(mat)
        engine = base.build_engine(args, mat)
        engine.f.da = float(args.da_phys); engine.f.max_advances_per_step = 1
        scale = resolve_branch_scale_identity(args, mesh)
        candidates = tungsten_cleavage_candidates(
            theta_deg=float(args.crystal_theta_deg or 0.0),
            include_110=bool(args.crystal_include_110),
            gamma_110_rel=float(args.gamma_110_rel or 1.3),
        )
        competition = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
        network = replace(
            CrackNetworkState.one_tip(((0.0, 0.0), (float(cfg.geometry.a0), 0.0))),
            branching_enabled=True,
        )
        damage = np.zeros(mesh.nn)
        damage[(mesh.nodes[:, 0] <= cfg.geometry.a0) & (np.abs(mesh.nodes[:, 1]) <= cfg.geometry.notch_half_thickness)] = 1.0
        state = LiveFEMTopologyState(
            mesh=mesh, boundary=boundary, damage=damage, displacement=np.zeros(mesh.ndof),
            ep_gp=np.zeros((3, mesh.ne)), rho_gp=np.full(mesh.ne, engine.f.rho0),
            elasticity_D=D, material=mat, cohesive_network=None, crack_network=network,
            competition=competition, tip_process_state={"shared_engine_model": type(engine).__name__},
            junction_process_state={"crack_representation": "sharp_wake_causal_v11"},
            energy_ledgers={name: 0.0 for name in (
                "retained", "mobile", "escaped", "recovered", "stored_energy",
                "emission_work", "unconsumed_action", "topology_release_J_per_m",
                "hazard_dissipation_J_per_m",
            )},
            rng_state=np.random.default_rng(3621).bit_generator.state,
            event_counters={"topology_actions": 0, "shared_state_updates": 0, "accepted_steps": 0},
            stored_energy_J_per_m=0.0,
        )
        cache = initial_root / "initial_live_kernel_cache"
        provider = LiveTopologyRuntime(str(cache.resolve()))
        checkpoint = ProductionBranchCheckpoint(
            state=state, shared_process_state=_capture_shared_engine(engine),
            physical_time_s=0.0, accepted_load=0.0, mesh_identity=_mesh_identity(mesh),
            boundary_condition_state={"opening_m": 0.0}, provider_runtime=provider,
            provider_cache_identity=str(cache.resolve()),
            topology_fingerprint=_hash(network), front_competitions={network.active_tip_ids[0]: competition},
            branch_clusters=(), projected_extension_m=0.0, physical_extension_m=0.0,
            handoff_guard_diagnostics={"scale_identity": scale.to_dict()}, termination_reason=None,
        )
        write_branch_checkpoint(checkpoint, v11_path)
        runtime = load_v11_checkpoint_as_v12(
            checkpoint, resource_policy=ResourcePolicy(
                "mechanistic", 6, None, "forbid", "v11_correlated_proposal_compatibility"
            ),
        )
        sigma = assemble_mechanics(
            mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
            state.elasticity_D, state.material, cohesive_network=None,
        )[2]
        from arrhenius_fracture.stateful_multifront_production_v12 import (
            accepted_context_state_identity, stress_field_identity,
        )
        accepted_id = accepted_context_state_identity(
            state, runtime, physical_time_s=0.0, accepted_opening_m=0.0, step_count=0,
        )
        runtime = replace(runtime, accepted_state_id=accepted_id)
        runtime = replace(runtime, stress_field_state_id=stress_field_identity(
            state, sigma, mechanics_source_identity="current_source_internal_fem_v12",
            accepted_state_id=accepted_id,
        ))
        write_accepted_boundary_checkpoint_v12(
            runtime, v12_path, accepted_fem_state=state, accepted_stress_field=sigma,
            mechanics_source_identity="current_source_internal_fem_v12",
            physical_time_s=0.0, accepted_opening_m=0.0, step_count=0,
            context_restart={"fresh_matched_single_front_initial_state": True, "case": case},
        )
        export_snapshot(
            output / "field_snapshots", case=case, runtime=runtime,
            accepted_fem_state=state, accepted_stress_field=sigma,
            physical_time_s=0.0, accepted_opening_m=0.0, step_count=0,
            mechanics_source_identity="current_source_internal_fem_v12",
            selection_reasons=("initial_accepted_state",), accepted_checkpoint_path=v12_path,
        )
        result["checkpoint"] = v12_path
        return 0

    original_run = base.run_2d
    original_registry, original_selection, original_options = paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS
    paper.DEFAULT_REGISTRY = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv"
    paper.SELECTION_RECORD = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json"
    paper.VALID_OPTIONS = {alias_value: canonical for canonical, alias_value in ROWS.values()}
    base.run_2d = create
    try:
        args = common_arguments(alias, temperature, family, initial_root)
        try:
            entry.main(args)
        except RuntimeError as exc:
            # The legacy entry stack writes a post-run stochastic-backend audit.
            # Initial-state construction deliberately performs no stochastic
            # update, so that audit has no backend to inspect.
            if str(exc) != "no stochastic avalanche backend was constructed" or not v12_path.is_file():
                raise
    finally:
        base.run_2d = original_run
        paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS = original_registry, original_selection, original_options
    if result.get("checkpoint") != v12_path or not v12_path.is_file():
        raise RuntimeError("fresh initial V12 accepted boundary was not created")
    return v12_path


def terminal_label(reason: str) -> str:
    if reason == "maximum_network_forward_reach_target_reached":
        return "V12_FIELD_ATLAS_TARGET_1000UM_REACHED"
    if reason == "configured_front_resource_limit_reached":
        return "V12_FIELD_ATLAS_STOPPED_POLICY_BOUND_AT_SIX_ACTIVE_FRONTS"
    if reason == "owner_local_kernel_coordinate_outside_qualified_family_domain":
        return "V12_FIELD_ATLAS_STOPPED_OWNER_KERNEL_ENVELOPE"
    safe = "".join(ch if ch.isalnum() else "_" for ch in reason.upper()).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return "V12_FIELD_ATLAS_STOPPED_FAIL_CLOSED_" + (safe or "UNKNOWN_REASON")


def run_case(
    case: str, campaign_root: Path, family: Path,
    resume_root: Path | None = None,
) -> int:
    from arrhenius_fracture.multifront_checkpoint_v12 import load_accepted_boundary_checkpoint_v12
    import arrhenius_fracture.multifront_checkpoint_v12 as checkpoint_module
    from arrhenius_fracture.multifront_field_atlas_v12 import (
        MILESTONES_UM, SparseAcceptedStateObserver, _owner_signature,
        export_snapshot,
    )
    from arrhenius_fracture.network_metrics_v11 import crack_growth_metrics
    from arrhenius_fracture import sharp_front_current_source_multifront_physical_v12 as physical

    material, temperature, canonical, alias = case_parts(case)
    output = (campaign_root / case).resolve()
    output.mkdir(parents=True, exist_ok=True)
    terminal_path = output / "terminal_manifest.json"
    if terminal_path.is_file():
        return 0
    row, row_hash = registry_row(canonical)
    source = protected_source_audit(); inputs = validate_inputs(family)
    checkpoint = output / "checkpoint" / "latest.v12.pkl"
    if resume_root is not None and not checkpoint.is_file():
        source_checkpoint = resume_root / case / "checkpoint" / "latest.v12.pkl"
        source_terminal = resume_root / case / "terminal_manifest.json"
        if not source_checkpoint.is_file() or not source_terminal.is_file():
            raise RuntimeError(f"immutable continuation source is incomplete for {case}")
        source_hash = sha256(source_checkpoint)
        source_record = json.loads(source_terminal.read_text())
        if source_hash != source_record.get("final_checkpoint_sha256"):
            raise RuntimeError(f"continuation checkpoint hash mismatch for {case}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, checkpoint)
        if sha256(checkpoint) != source_hash:
            raise RuntimeError(f"copied continuation checkpoint hash mismatch for {case}")
        atomic_json(output / "continuation_source.json", {
            "schema": "v12.field-atlas-checkpoint-continuation/1",
            "case": case,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": source_hash,
            "copied_checkpoint": str(checkpoint.resolve()),
            "copied_checkpoint_sha256": sha256(checkpoint),
            "source_raw_tree_fingerprint": str(
                (resume_root / case / "raw_tree_fingerprint.json").resolve()
            ),
            "source_terminal_reason": source_record.get("exact_terminal_reason"),
            "continuation_preserves_clocks_thresholds_ordinals_and_rng": True,
        })
    elif not checkpoint.is_file():
        checkpoint = build_fresh_initial(case, alias, temperature, family, output)
    launch_args = [
        "--maximum-fronts", "6", "--v12-restart-checkpoint", str(checkpoint),
        "--v12-target-maximum-forward-reach-um", "1000",
        *common_arguments(alias, temperature, family, output),
    ]
    launch = {
        "schema": "v12.multifront-field-atlas-case-launch/1", "claim_label": CLAIM,
        "case": case, "material_class": material,
        "canonical_parameterization_id": canonical, "execution_alias": alias,
        "parameter_row": row, "parameter_row_canonical_json_sha256": row_hash,
        "temperature_K": temperature, "theta_deg": 40.0, "seed": 3621,
        "source": source, "family": str(family.resolve()), "family_sha256": FAMILY_SHA256,
        "family_physics_fingerprint": FAMILY_PHYSICS,
        "mechanical_configuration_sha256": MECHANICAL_SHA256,
        "mechanical_fingerprint": MECHANICAL_FINGERPRINT,
        "mechanics_group": "theta40_temperature_independent_mechanics",
        "family_validation": inputs["family_validation"],
        "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": sha256(checkpoint),
        "physical_entry_arguments": launch_args,
        "outer_command": [str(PYTHON), "-u", str(Path(__file__).resolve()), "--case", case,
                          "--campaign-root", str(campaign_root), "--family", str(family)],
        "maximum_active_fronts": 6, "total_branch_birth_limit": None,
        "target_maximum_forward_reach_um": 1000.0, "started_utc": now(),
    }
    atomic_json(output / "launch_manifest.json", launch)
    original_writer = checkpoint_module.write_accepted_boundary_checkpoint_v12
    initial = load_accepted_boundary_checkpoint_v12(checkpoint)
    crossed_milestones = set()
    for metadata_path in (output / "field_snapshots").glob("*/metadata.json"):
        for reason in json.loads(metadata_path.read_text()).get("selection_reasons", []):
            for milestone in MILESTONES_UM:
                if reason == f"milestone_at_or_below_{int(milestone)}um":
                    crossed_milestones.add(milestone)
    observer = SparseAcceptedStateObserver(
        case=case, snapshot_root=output / "field_snapshots", checkpoint_path=checkpoint,
        original_writer=original_writer, previous_births=initial.runtime.cumulative_branch_births,
        previous_owner_signature=_owner_signature(initial.runtime),
        crossed_milestones=crossed_milestones,
    )
    checkpoint_module.write_accepted_boundary_checkpoint_v12 = observer
    failure = None; failure_class = None; returncode = 0
    try:
        result = physical.main(launch_args)
        returncode = int(result or 0)
        if returncode:
            failure = f"physical_entry_returncode_{returncode}"
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        returncode = int(exc.code) if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1
        failure = str(exc) or type(exc).__name__
        failure_class = type(exc).__name__
        (output / "terminal_exception.txt").write_text(traceback.format_exc())
    finally:
        checkpoint_module.write_accepted_boundary_checkpoint_v12 = original_writer

    restored = load_accepted_boundary_checkpoint_v12(checkpoint)
    growth = crack_growth_metrics(restored.runtime.crack_network, initial_crack_length_m=0.5e-3)
    complete_path = output / "v12_run_complete.json"
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
    reason = failure or complete.get("termination") or restored.runtime.termination_reason or "unknown_fail_closed_stop"
    reached_um = growth.max_forward_projected_extension_m * 1e6
    final_reasons = ["final_last_accepted_state"]
    final_snapshot = export_snapshot(
        output / "field_snapshots", case=case, runtime=restored.runtime,
        accepted_fem_state=restored.accepted_fem_state,
        accepted_stress_field=restored.accepted_stress_field,
        physical_time_s=restored.physical_time_s,
        accepted_opening_m=restored.accepted_opening_m,
        step_count=restored.step_count,
        mechanics_source_identity=restored.mechanics_source_identity,
        selection_reasons=final_reasons, accepted_checkpoint_path=checkpoint,
    )
    maximum_active = max(
        [1] + [int(item.get("active_front_count", 1)) for item in (
            json.loads(line) for line in (output / "v12_progress.jsonl").read_text().splitlines()
        )] if (output / "v12_progress.jsonl").is_file() else [1]
    )
    terminal = {
        "schema": "v12.multifront-field-atlas-terminal/1", "claim_label": CLAIM,
        "case": case, "canonical_parameterization_id": canonical, "execution_alias": alias,
        "temperature_K": temperature, "theta_deg": 40.0, "seed": 3621,
        "terminal_label": terminal_label(reason), "exact_terminal_reason": reason,
        "terminal_exception_class": failure_class,
        "returncode": returncode, "target_reached": reached_um >= 1000.0 - 1e-8,
        "achieved_forward_reach_um": reached_um, "terminal_step": restored.step_count,
        "terminal_time_s": restored.physical_time_s,
        "terminal_opening_m": restored.accepted_opening_m,
        "cumulative_branch_births": restored.runtime.cumulative_branch_births,
        "maximum_concurrent_active_fronts": maximum_active,
        "final_active_front_count": len(restored.runtime.active_front_ids),
        "junction_count": len(restored.runtime.junctions),
        "retirement_count": restored.runtime.cumulative_retirements,
        "coalescence_count": restored.runtime.cumulative_coalescences,
        "final_checkpoint_path": str(checkpoint), "final_checkpoint_sha256": sha256(checkpoint),
        "final_field_package_path": final_snapshot["files"]["fields_npz"],
        "final_field_package_sha256": final_snapshot["files"]["fields_npz_sha256"],
        "finished_utc": now(), "predictive_recursive_branching_physics_validated": False,
    }
    atomic_json(terminal_path, terminal)
    launch["finished_utc"] = terminal["finished_utc"]; launch["returncode"] = returncode
    atomic_json(output / "launch_manifest.json", launch)
    return 0


def queue_cases(
    campaign_root: Path, family: Path, workers: int,
    resume_root: Path | None = None,
) -> int:
    if workers < 1 or workers > 2:
        raise ValueError("the campaign permits one or two heavy workers")
    campaign_root.mkdir(parents=True, exist_ok=True)
    pending = [case for case in CASES if not (campaign_root / case / "terminal_manifest.json").is_file()]
    running: dict[subprocess.Popen, tuple[str, object, object]] = {}
    failures = []
    while pending or running:
        while pending and len(running) < workers:
            case = pending.pop(0); out = campaign_root / case; out.mkdir(parents=True, exist_ok=True)
            stdout = (out / "worker.stdout.log").open("a")
            stderr = (out / "worker.stderr.log").open("a")
            cmd = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--case", case,
                   "--campaign-root", str(campaign_root), "--family", str(family)]
            if resume_root is not None:
                cmd.extend(("--resume-root", str(resume_root)))
            proc = subprocess.Popen(cmd, cwd=ROOT, env=campaign_environment(family), stdout=stdout, stderr=stderr)
            running[proc] = (case, stdout, stderr)
            print(json.dumps({"event": "worker_started", "case": case, "pid": proc.pid, "utc": now()}), flush=True)
        finished = []
        for proc, handles in running.items():
            code = proc.poll()
            if code is not None:
                case, stdout, stderr = handles; stdout.close(); stderr.close()
                print(json.dumps({"event": "worker_finished", "case": case, "returncode": code, "utc": now()}), flush=True)
                if code or not (campaign_root / case / "terminal_manifest.json").is_file():
                    failures.append(case)
                finished.append(proc)
        for proc in finished:
            del running[proc]
        if running and not finished:
            import time; time.sleep(5.0)
    atomic_json(campaign_root / "queue_terminal.json", {
        "schema": "v12.multifront-field-atlas-queue/1", "cases": list(CASES),
        "maximum_workers": workers, "failures": failures, "finished_utc": now(),
    })
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=ROOT, text=True).strip()
    if branch != BRANCH:
        raise RuntimeError(f"campaign branch mismatch: {branch}")
    tracked_dirty = subprocess.run(("git", "diff", "--quiet"), cwd=ROOT).returncode
    index_dirty = subprocess.run(("git", "diff", "--cached", "--quiet"), cwd=ROOT).returncode
    if tracked_dirty or index_dirty:
        raise RuntimeError("campaign execution requires clean committed tracked source")
    validate_inputs(args.family); protected_source_audit()
    for canonical, alias in ROWS.values():
        row, _ = registry_row(canonical)
        if row["option_key"] != alias:
            raise RuntimeError(f"registry alias mismatch for {canonical}")
    if args.preflight:
        return 0
    if args.case:
        return run_case(
            args.case, args.campaign_root.resolve(), args.family.resolve(),
            None if args.resume_root is None else args.resume_root.resolve(),
        )
    return queue_cases(
        args.campaign_root.resolve(), args.family.resolve(), args.workers,
        None if args.resume_root is None else args.resume_root.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
