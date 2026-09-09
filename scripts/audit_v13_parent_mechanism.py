#!/usr/bin/env python3
"""Read-only prerequisite audit for the proposed V13 parent-mechanism contract.

This is not V13 parity qualification. No V13 overlay or PF step is executed.
Full accepted checkpoints supply the clocks and physical engine fields; source
call paths determine their semantics. Scalar history rows are never substitutes
for absent emission clocks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

from arrhenius_fracture.current_source_multifront_hooks_v12 import (
    restore_complete_current_source_engine,
)
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalHazardState,
    competition_state_from_dict, competition_state_to_dict,
    tungsten_cleavage_candidates,
)
from arrhenius_fracture.general_multifront_v12 import (
    FrontRuntimeState, MultiFrontRuntimeState, ProcessEngineState,
    ResourcePolicy, canonical_hash,
)
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.multifront_competition_v12 import preview_competitions
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_accepted_boundary_checkpoint_v12,
)
from arrhenius_fracture.persistent_site_source_v10221 import (
    SOURCE_MODEL, _persistent_emit,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine


ROOT = Path(__file__).resolve().parents[1]
PARENT_COMMIT = "b3d0add6cbb0605adaa3e04006fe987961ad6452"
BOUNDARY = "BRANCHING_KINETICS_MODEL_UNCALIBRATED"


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def binding_identity(method):
    function = getattr(method, "__func__", method)
    return f"{function.__module__}.{function.__qualname__}"


def correlation_timing_counterexample():
    """Exercise the production preview with a deterministic one-clock fixture."""
    candidate = tungsten_cleavage_candidates(theta_deg=0.0)[0]
    competition = DirectionalCompetitionState(
        (candidate,),
        (DirectionalHazardState(candidate.candidate_id, previous_rate_per_s=1.0),),
        global_hazard_seed=0,
    )
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front_id = network.active_tip_ids[0]
    runtime = MultiFrontRuntimeState.one_front(
        network,
        FrontRuntimeState(front_id, competition_state_to_dict(competition),
                          (candidate.candidate_id,), {"seed": 0}),
        ProcessEngineState("engine:fixture", "source:fixture", {}, {}, {}),
        resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ),
        accepted_state_id="fixture", stress_field_state_id="fixture",
    )
    before = runtime.registry_fingerprint
    rates = {front_id: {candidate.candidate_id: 1.0}}
    endpoints = {front_id: {candidate.candidate_id: (5e-6, 0.0)}}
    def preview(duration, correlation):
        return preview_competitions(
            runtime, rates, endpoints, start_time_s=0.0,
            duration_s=duration, correlation_interval_s=correlation,
        )
    direct = preview(1.0, 0.0)
    waiting = preview(1.0, 0.25)
    ready = preview(1.25, 0.25)
    return {
        "fixture_not_a_physical_trajectory": True,
        "rate_per_s": 1.0, "threshold_action": 1.0,
        "correlation_interval_s": 0.25,
        "cleavage_clock_completion_s": direct.proposals[0].completion_time_s,
        "proposal_count_at_clock_completion_with_correlation": len(waiting.proposals),
        "one_arm_proposal_ready_s": ready.proposals[0].completion_time_s,
        "runtime_unchanged": before == runtime.registry_fingerprint,
    }


def inspect_checkpoint(case, role, path, expected_sha):
    before = sha_file(path)
    if before != expected_sha:
        raise RuntimeError(f"{case}/{role}: checkpoint source hash mismatch")
    restored = load_accepted_boundary_checkpoint_v12(path)
    runtime = restored.runtime
    runtime_before = canonical_hash(runtime.to_dict())
    fronts = []
    for front_id, front in sorted(runtime.front_runtimes.items()):
        competition = competition_state_from_dict(front.competition_state)
        fronts.append({
            "front_id": front_id,
            "owner_id": runtime.owner_by_front[front_id],
            "candidate_ids": list(front.candidate_ids),
            "mechanically_active_candidate_ids": list(front.mechanically_active_candidate_ids),
            "cleavage_clock_count": sum(
                item.candidate_id.startswith("cleave:") for item in competition.hazard_states
            ),
            "noncleavage_clock_count": sum(
                not item.candidate_id.startswith("cleave:") for item in competition.hazard_states
            ),
            "complete_competition_state": front.competition_state,
            "lineage_rng_state": front.lineage_rng_state,
            "competition_sha256": canonical_hash(front.competition_state),
        })
    owners = []
    for engine_id, saved in sorted(runtime.process_engines.items()):
        engine = restore_complete_current_source_engine(saved)
        frozen_hash = object_hash(_capture_shared_engine(engine))
        rng_before = object_hash(getattr(engine, "_hazard_rng", None))
        # Re-evaluate only the unchanged barrier law at the engine's cached
        # owner opening. This is NOT a new front tensor/kinetic-J evaluation.
        opening = float(getattr(engine, "_opening_sigma_local_Pa", 0.0))
        temperature = float(case.rsplit("_", 1)[1][:-1])
        rate = engine.lambda_cleave(opening, temperature)
        repeated = engine.lambda_cleave(opening, temperature)
        if rate != repeated:
            raise RuntimeError("frozen cleavage evaluator is not repeatable")
        actual_binding = binding_identity(engine.mpz._emit)
        expected_binding = binding_identity(_persistent_emit)
        owner = {
            "engine_id": engine_id,
            "engine_class": type(engine).__name__,
            "source_model": engine.mpz.source_model,
            "complete_engine_payload_sha256": saved.checkpoint_payload_sha256,
            "restored_serializable_state_sha256": frozen_hash,
            "cached_owner_opening_Pa": opening,
            "rate_observation_scope": "cached_owner_scalar_opening_no_new_FEM_or_tensor_probe",
            "cleavage_effective_rate_per_s": float(rate[0]),
            "cleavage_raw_rate_per_s": float(rate[1]),
            "cleavage_barrier_J": float(rate[2]),
            "tau_c_s": float(engine.f.tau_c),
            "multihit_order": float(engine.f.m_hits),
            "restored_emission_method": actual_binding,
            "persistent_installer_emission_method": expected_binding,
            "persistent_emission_binding_matches_installer": actual_binding == expected_binding,
            "serialized_engine_field_names": sorted(saved.complete_checkpoint_payload()["engine_fields"]),
            "serialized_mpz_field_names": sorted(saved.complete_checkpoint_payload()["mpz_fields"]),
            "engine_rng_sha256_before": rng_before,
            "engine_rng_sha256_after": object_hash(getattr(engine, "_hazard_rng", None)),
            "frozen_rates_repeat_exactly": True,
            "state_unchanged": frozen_hash == object_hash(_capture_shared_engine(engine)),
            "cached_emission_diagnostics_scope": "last_source_update_diagnostics_not_emission_clock_state",
        }
        for field in (
            "persistent_site_last_rate_initial_s", "persistent_site_last_rate_final_s",
            "persistent_site_last_activations", "anisotropic_last_lambda_emit_by_system_s",
        ):
            value = getattr(engine.mpz, field, None)
            owner[field] = None if value is None else value.tolist()
        if not owner["state_unchanged"] or owner["engine_rng_sha256_after"] != rng_before:
            raise RuntimeError("frozen-state inspection mutated engine state or RNG")
        owners.append(owner)
    if runtime_before != canonical_hash(runtime.to_dict()) or before != sha_file(path):
        raise RuntimeError("read-only audit changed its accepted source")
    return {
        "case": case, "checkpoint_role": role, "path": str(path),
        "checkpoint_sha256": before, "checkpoint_unchanged": True,
        "step": restored.step_count, "physical_time_s": restored.physical_time_s,
        "accepted_opening_m": restored.accepted_opening_m,
        "accepted_state_id": runtime.accepted_state_id,
        "stress_field_state_id": runtime.stress_field_state_id,
        "runtime_sha256": runtime_before, "runtime_unchanged": True,
        "fronts": fronts, "owners": owners,
    }


def source_evidence():
    anchors = {
        "arrhenius_fracture/unified_front.py": ["Cleavage and plasticity evolve concurrently"],
        "arrhenius_fracture/multifront_competition_v12.py": [
            "def preview_competitions(", "correlation_ready_s = (",
        ],
        "arrhenius_fracture/sharp_front_v11_branching.py": [
            "def _serializable_fields(", "def _restore_shared_engine(",
            "engine_trial._directional_topology_owns_cleavage = True",
        ],
        "arrhenius_fracture/stateful_multifront_production_v12.py": [
            "selected_value = select_global_topology_proposal(",
            "# Evolve each physical owner once",
        ],
        "arrhenius_fracture/persistent_site_source_v10221.py": [
            "def _persistent_emit(", "state._emit = MethodType(_persistent_emit, state)",
        ],
        "arrhenius_fracture/current_source_multifront_hooks_v12.py": [
            "def restore_complete_current_source_engine(",
        ],
        "arrhenius_fracture/unified_mpz.py": ["    def _emit("],
        "arrhenius_fracture/stochastic_hazard_tip.py": ["HAZARD_SCHEMA ="],
    }
    records = {}
    for relative, needles in anchors.items():
        path = ROOT / relative
        baseline = subprocess.check_output(
            ["git", "show", f"{PARENT_COMMIT}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(baseline).hexdigest() != sha_file(path):
            raise RuntimeError(f"parent production source changed: {relative}")
        lines = path.read_text().splitlines()
        matches = {}
        for needle in needles:
            found = [i + 1 for i, line in enumerate(lines) if needle in line]
            if not found:
                raise RuntimeError(f"source evidence moved: {relative}: {needle}")
            matches[needle] = found
        records[relative] = {"sha256": sha_file(path), "line_anchors": matches}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    atlas = args.atlas_root.resolve()
    summary = atlas / "authoritative_corrected_field_atlas_v3_wake_remap/pf_multifront_field_atlas_terminal_summary.csv"
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8 or len({row["case"] for row in rows}) != 8:
        raise RuntimeError("expected the eight completed atlas cases")
    records = []
    for row in rows:
        for role, prefix in (("pre_wake_remap_source_evidence", "source"), ("corrected_terminal", "final")):
            path = Path(row[f"{prefix}_checkpoint_path"])
            records.append(inspect_checkpoint(row["case"], role, path, row[f"{prefix}_checkpoint_sha256"]))
    fronts = [front for record in records for front in record["fronts"]]
    owners = [owner for record in records for owner in record["owners"]]
    result = {
        "schema": "v13.parent-mechanism-prerequisite-audit/1",
        "interpretation_boundary": BOUNDARY,
        "baseline_source_commit": PARENT_COMMIT,
        "producer_code_sha256": sha_file(__file__),
        "producer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "input_index_sha256": sha_file(summary), "source_evidence": source_evidence(),
        "checkpoint_count": len(records), "front_record_count": len(fronts),
        "process_engine_record_count": len(owners),
        "noncleavage_scheduler_clock_count": sum(row["noncleavage_clock_count"] for row in fronts),
        "persistent_emission_binding_mismatch_count": sum(
            row["source_model"] == SOURCE_MODEL and not row["persistent_emission_binding_matches_installer"]
            for row in owners
        ),
        "existing_correlation_timing_counterexample": correlation_timing_counterexample(),
        "precondition_status": "BLOCKED_PARENT_MECHANISM_CONTRACT_MISMATCH",
        "v13_disabled_parity": "NOT_RUN_NO_V13_OVERLAY_IMPLEMENTED",
        "v13_enabled_precleavage_parity": "NOT_RUN_NO_V13_OVERLAY_IMPLEMENTED",
        "new_PF_steps": 0, "new_FEM_solves": 0, "branch_draws": 0,
        "checkpoints": records,
    }
    out = args.output_root.resolve()
    if out == atlas or out.is_relative_to(atlas):
        raise RuntimeError("audit output must be outside the immutable atlas")
    out.mkdir(parents=True, exist_ok=True)
    (out / "parent_mechanism_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"checkpoints", "source_evidence"}}, indent=2))


if __name__ == "__main__":
    main()
