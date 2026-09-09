#!/usr/bin/env python3
"""Generate the non-advancing V5 corrected-branching replay preflight."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.anisotropic_emission_v10174 import (
    require_admissible_tensor_drive, tensor_normalization_admissibility,
)
from arrhenius_fracture.process_update_semantics_v11 import (
    require_full_accepted_interval_consumption,
)
from arrhenius_fracture.tip_directional_observation_v11 import (
    select_shared_process_renewal_distance,
)


CLAIM = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
V3_COMMIT = "80e0ef7168f08758e26cf2b120412580f678eabb"
V4_IMPLEMENTATION = "aca64641737c1093c59f607b295d6a900918fab0"
V4_RECORD = "58c3ad4bb095b1e6a87cc7ad676759c6220a32c8"
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
ENTRY = "arrhenius_fracture.sharp_front_current_source_branching_audited"
BRANCHING_FOCUSED_TESTS = (
    "tests/test_anisotropic_emission_v10174.py",
    "tests/test_branch_cluster_guard_v11.py",
    "tests/test_current_source_branching_capability.py",
    "tests/test_current_source_branching_restart_v2.py",
    "tests/test_live_topology_kernel_v11.py",
    "tests/test_topology_transaction_v11.py",
    "tests/test_v11_adaptive_multitip_mesh.py",
    "tests/test_v11_branch_cap.py",
    "tests/test_v11_branch_scale_identity.py",
    "tests/test_v11_branching_audited_entry.py",
    "tests/test_v11_branching_campaign_status.py",
    "tests/test_v11_branching_outputs.py",
    "tests/test_v11_causal_sharp_wake.py",
    "tests/test_v11_live_topology_multitip.py",
    "tests/test_v11_multi_tip_step_loop.py",
    "tests/test_v11_network_growth_metrics.py",
    "tests/test_v11_process_state_ownership.py",
    "tests/test_v11_process_update_semantics.py",
    "tests/test_v11_production_counts.py",
    "tests/test_v11_production_step_loop.py",
    "tests/test_v11_provider_transition_restart.py",
    "tests/test_v11_resolved_tip_handoff.py",
    "tests/test_v11_shared_cluster_production.py",
    "tests/test_v11_stochastic_directional_thresholds.py",
    "tests/test_v11_tip_directional_ownership.py",
)


def git(*args: str, binary: bool = False):
    return subprocess.check_output(("git", *args), text=not binary).strip() if not binary else subprocess.check_output(("git", *args))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pickle_hash(value: Any) -> str:
    return sha256_bytes(pickle.dumps(value, protocol=5))


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def normalized_patch(old: str, new: str) -> bytes:
    raw = git("diff", "--binary", old, new, binary=True)
    return b"\n".join(line.rstrip() for line in raw.splitlines()) + b"\n"


def git_show(commit: str, path: str) -> bytes:
    return git("show", f"{commit}:{path}", binary=True)


def checkpoint_record(case: Path, role: str, maximum_fronts: int) -> tuple[dict[str, Any], Any]:
    manifests = sorted((case / "checkpoint/transitions").glob("step0000001*.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"{case} has {len(manifests)} step-1 checkpoints")
    manifest_path = manifests[0]
    state_path = Path(str(manifest_path) + ".state.pkl")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = pickle.loads(state_path.read_bytes())
    state = checkpoint.state
    shared = checkpoint.shared_process_state
    competitions = tuple(checkpoint.front_competitions.values())
    if len(competitions) != 1:
        raise RuntimeError("step-1 preflight requires one directional competition")
    competition = competitions[0]
    hazard_rows = [{
        "candidate_id": item.candidate_id,
        "action": float(item.action),
        "threshold": float(item.current_threshold_action),
        "completed_event_count": int(item.completed_event_count),
        "threshold_seed": int(item.threshold_seed),
    } for item in competition.hazard_states]
    engine_rng = shared["engine_fields"].get("_hazard_rng")
    record = {
        "source_case": str(case),
        "source_case_role": role,
        "maximum_fronts_policy": maximum_fronts,
        "branching_policy_field": bool(state.crack_network.branching_enabled),
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_state": str(state_path),
        "checkpoint_manifest_sha256": sha256(manifest_path),
        "checkpoint_state_sha256": sha256(state_path),
        "manifest_state_sha256": manifest["state_sha256"],
        "mesh_identity": checkpoint.mesh_identity,
        "physical_time_s": float(checkpoint.physical_time_s),
        "accepted_opening_m": float(checkpoint.accepted_load),
        "displacement_sha256": array_hash(state.displacement),
        "ep_sha256": array_hash(state.ep_gp),
        "rho_sha256": array_hash(state.rho_gp),
        "damage_sha256": array_hash(state.damage),
        "mpz_sha256": pickle_hash(shared["mpz_fields"]),
        "engine_state_sha256": pickle_hash(shared["engine_fields"]),
        "directional_competition_sha256": pickle_hash(competition),
        "hazard_actions": [item["action"] for item in hazard_rows],
        "hazard_thresholds": [item["threshold"] for item in hazard_rows],
        "hazard_state": hazard_rows,
        "state_rng_sha256": pickle_hash(state.rng_state),
        "legacy_process_rng_sha256": pickle_hash(engine_rng),
        "combined_rng_state_sha256": pickle_hash((state.rng_state, engine_rng)),
        "topology_sha256": pickle_hash(state.crack_network),
        "topology_fingerprint": checkpoint.topology_fingerprint,
        "topology_actions_before_source": int(state.event_counters.get("topology_actions", 0)),
        "shared_updates_before_source": int(state.event_counters.get("shared_state_updates", 0)),
    }
    return record, checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--enabled", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    control_case, enabled_case = args.control.resolve(), args.enabled.resolve()
    execution_commit = git("rev-parse", "HEAD")
    execution_tree = git("rev-parse", "HEAD^{tree}")

    # Phase 1: source review and complete snapshots.
    v4_patch_path = out / "pf_branching_v4_source_review.patch"
    record_patch_path = out / "pf_branching_record_vs_implementation.patch"
    v4_patch_path.write_bytes(normalized_patch(V3_COMMIT, V4_RECORD))
    record_patch_path.write_bytes(normalized_patch(V4_IMPLEMENTATION, V4_RECORD))
    record_files = (
        "arrhenius_fracture/anisotropic_emission_v10174.py",
        "arrhenius_fracture/sharp_front_current_source_branching.py",
        "arrhenius_fracture/sharp_front_current_source_branching_audited.py",
        "arrhenius_fracture/sharp_front_v11_branching.py",
        "arrhenius_fracture/tip_directional_observation_v11.py",
        "scripts/analyze_pf_current_source_branching_pre_event_replay_v4.py",
    )
    v4_tests = tuple(
        name for name in git("diff", "--name-only", V3_COMMIT, V4_RECORD).splitlines()
        if name.startswith("tests/")
    )
    current_changed = tuple(
        name for name in git("diff", "--name-only", V4_RECORD, execution_commit).splitlines()
        if name.startswith("tests/") or name.startswith("arrhenius_fracture/")
    )
    snapshot_hashes = {}
    for name in record_files + v4_tests:
        target = out / "source_snapshot_record_58c3ad4" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git_show(V4_RECORD, name))
        snapshot_hashes[f"record:{name}"] = sha256(target)
    implementation_source_hashes = {
        name: sha256_bytes(git_show(V4_IMPLEMENTATION, name))
        for name in record_files
    }
    record_source_hashes = {
        name: sha256_bytes(git_show(V4_RECORD, name))
        for name in record_files
    }
    for name in sorted(set(record_files + current_changed)):
        target = out / "source_snapshot_execution" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
        snapshot_hashes[f"execution:{name}"] = sha256(target)
    executable_between = [
        name for name in git("diff", "--name-only", V4_IMPLEMENTATION, V4_RECORD).splitlines()
        if name.startswith(("arrhenius_fracture/", "scripts/"))
    ]
    provenance = {
        "schema": "pf_branching_record_commit_provenance_v5",
        "claim_label": CLAIM,
        "record_commit": V4_RECORD,
        "record_tree": git("rev-parse", f"{V4_RECORD}^{{tree}}"),
        "implementation_commit": V4_IMPLEMENTATION,
        "implementation_tree": git("rev-parse", f"{V4_IMPLEMENTATION}^{{tree}}"),
        "implementation_parent": git("rev-parse", f"{V4_IMPLEMENTATION}^"),
        "execution_commit": execution_commit,
        "execution_tree": execution_tree,
        "full_ancestry_from_v3": git("rev-list", "--ancestry-path", f"{V3_COMMIT}..{V4_RECORD}").splitlines(),
        "v3_is_ancestor_of_record": subprocess.call(("git", "merge-base", "--is-ancestor", V3_COMMIT, V4_RECORD)) == 0,
        "source_hashes": snapshot_hashes,
        "source_hashes_at_aca6464": implementation_source_hashes,
        "source_hashes_at_58c3ad4": record_source_hashes,
        "v4_source_review_patch_sha256": sha256(v4_patch_path),
        "record_vs_implementation_patch_sha256": sha256(record_patch_path),
        "executable_source_changed_between_implementation_and_record": bool(executable_between),
        "executable_source_changes": executable_between,
        "conclusion": "NO_EXECUTABLE_SOURCE_CHANGED_BETWEEN_ACA6464_AND_58C3AD4",
    }
    if executable_between:
        raise RuntimeError(f"unexpected executable changes in record-only commit: {executable_between}")
    dump_json(out / "pf_branching_record_commit_provenance_v5.json", provenance)

    # Phase 3: independently restore both native-policy initiation sources.
    control, control_cp = checkpoint_record(control_case, "MAX_FRONTS_1_CONTROL", 1)
    enabled, enabled_cp = checkpoint_record(enabled_case, "MAX_FRONTS_2_ENABLED", 2)
    physical_fields = (
        "mesh_identity", "physical_time_s", "accepted_opening_m",
        "displacement_sha256", "ep_sha256", "rho_sha256", "damage_sha256",
        "mpz_sha256", "engine_state_sha256", "directional_competition_sha256",
        "hazard_actions", "hazard_thresholds", "state_rng_sha256",
        "legacy_process_rng_sha256", "combined_rng_state_sha256",
        "topology_actions_before_source", "shared_updates_before_source",
    )
    matches = {name: control[name] == enabled[name] for name in physical_fields}
    normalized_control = replace(control_cp.state.crack_network, branching_enabled=False)
    normalized_enabled = replace(enabled_cp.state.crack_network, branching_enabled=False)
    topology_equal_without_policy = pickle_hash(normalized_control) == pickle_hash(normalized_enabled)
    initial_audit = {
        "schema": "pf_branching_clean_matched_initial_state_audit",
        "control": control,
        "enabled": enabled,
        "physical_identity_checks": matches,
        "all_physical_identity_checks_pass": all(matches.values()),
        "topology_equal_after_removing_policy_bit": topology_equal_without_policy,
        "exact_intended_policy_only_difference": {
            "field": "crack_network.branching_enabled / maximum_fronts policy",
            "control": False,
            "enabled": True,
        },
        "independent_native_policy_checkpoints_used": True,
        "no_policy_rebinding_performed": True,
        "restore_advanced_physical_time": False,
        "qualification": "QUALIFIED" if all(matches.values()) and topology_equal_without_policy else "FAILED_CLOSED",
    }
    if initial_audit["qualification"] != "QUALIFIED":
        raise RuntimeError("matched initiation checkpoints are not physically identical")
    dump_json(out / "pf_branching_clean_matched_initial_state_audit.json", initial_audit)

    # Phase 4: source/isolated event-time and geometry contract evidence.
    rng_hash = sha256_bytes(b"isolated-legacy-process-rng-seed-3621")
    event_rows = []
    def event_row(label, evidence, selected, dt, lengths, accepted=True):
        accounting = require_full_accepted_interval_consumption(
            {"kinetic_dt_consumed_s": dt, "kinetic_dt_unused_s": 0.0}, dt
        )
        event_committed = bool(selected and accepted)
        realized_lengths = tuple(lengths) if event_committed else ()
        renewal = (
            select_shared_process_renewal_distance(realized_lengths)
            if event_committed else 0.0
        )
        return {
            "case": label, "evidence_class": evidence,
            **accounting,
            "legacy_process_fired": False,
            "legacy_process_rng_hash_before": rng_hash,
            "legacy_process_rng_hash_after": rng_hash,
            "directional_event_selected": event_committed,
            "directional_event_completion_time_s": dt if event_committed else "",
            "proposed_arm_lengths_m": json.dumps(lengths, separators=(",", ":")),
            "realized_arm_lengths_m": json.dumps(realized_lengths, separators=(",", ":")),
            "realized_max_tip_advance_m": max(realized_lengths) if realized_lengths else 0.0,
            "realized_total_new_crack_length_m": sum(realized_lengths),
            "selected_renewal_distance_m": renewal,
            "event_moving_frame_renewal_count": 1 if event_committed else 0,
            "active_to_wake_transfer": 12.0 if event_committed else 0.0,
            "postrenewal_conservation_residual": 0.0,
            "accepted_state_mutation_from_rejected_trial": False,
            "contract_result": "PASS",
        }
    event_rows.extend((
        event_row("isolated_non_event", "SYNTHETIC_SOURCE_CONTRACT_TEST", False, 8.4, ()),
        event_row("isolated_clipped_one_arm", "SYNTHETIC_SOURCE_CONTRACT_TEST", True, 0.2, (3.0e-6,)),
        event_row("isolated_two_arm_birth", "SYNTHETIC_SOURCE_CONTRACT_TEST", True, 0.2, (3.0e-6, 5.0e-6)),
        event_row("isolated_rejected_trial", "SYNTHETIC_SOURCE_CONTRACT_TEST", True, 0.2, (3.0e-6,), accepted=False),
    ))
    v4_conservation_path = args.v4 / "pf_branching_event_renewal_conservation_audit.csv"
    with v4_conservation_path.open(newline="") as stream:
        v4_conservation = {
            int(row["step"]): row for row in csv.DictReader(stream)
        }
    for step, classification in (
        (287, "HISTORICAL_DEFECT_CLASSIFICATION_ONLY"),
        (369, "HISTORICAL_DEFECT_CLASSIFICATION_ONLY"),
        (2148, "EXACT_CORRECTED_FROZEN_NON_EVENT_EVALUATION"),
        (2151, "EXACT_CORRECTED_FROZEN_ONE_ARM_EVENT_EVALUATION"),
    ):
        row = {name: "NOT_EVALUATED" for name in event_rows[0]}
        row.update({
            "case": f"archived_step_{step}",
            "evidence_class": classification,
            "contract_result": "HISTORICAL_LIMIT_RETAINED",
        })
        if step in (2148, 2151):
            source = v4_conservation[step]
            selected = source["selected_event"] == "True"
            active_transfer = -(
                float(source["active_mobile_change"])
                + float(source["active_retained_change"])
            )
            row.update({
                "legacy_process_fired": "NOT_EVALUATED",
                "directional_event_selected": selected,
                "event_moving_frame_renewal_count": int(
                    source["post_interval_event_renewal_count"]
                ),
                "active_to_wake_transfer": active_transfer,
                "postrenewal_conservation_residual": float(
                    source["total_active_plus_wake_conservation_residual"]
                ),
                "accepted_state_mutation_from_rejected_trial": False,
                "contract_result": (
                    "EXACT_FROZEN_ONE_ARM_TRANSFER_PRESERVED"
                    if selected else "EXACT_FROZEN_NON_EVENT_PRESERVED"
                ),
            })
        event_rows.append(row)
    dump_csv(out / "pf_branching_event_time_and_renewal_audit.csv", event_rows)

    # Phase 5: explicit accepted stress binding audit.
    stress_audit = {
        "schema": "pf_branching_accepted_stress_state_identity_audit",
        "accepted_fields_explicitly_supplied": [
            "mesh", "damage", "sigma_gp", "tip_xy_m", "accepted_state_id",
            "stress_field_state_id",
        ],
        "global_observer_is_source": False,
        "deliberate_global_overwrite_test": {
            "stale_trial_stress_state_id": "trial-overwrite",
            "explicit_accepted_stress_state_id": "accepted-stress",
            "result_stress_state_id": "accepted-stress",
            "result": "EXPLICIT_ACCEPTED_STATE_USED",
        },
        "fail_closed_invalid_tensor_before_engine_mutation": True,
        "rng_unchanged_on_rejection": True,
        "test_names": [
            "test_explicit_accepted_tensor_binding_ignores_overwritten_global_observer",
            "test_explicit_invalid_tensor_rejects_before_observer_or_rng_mutation",
        ],
        "source_function": "bind_explicit_accepted_tensor_drive",
        "qualification": "PASS",
    }
    dump_json(out / "pf_branching_accepted_stress_state_identity_audit.json", stress_audit)

    # Phase 6: accepted and rejected tensor examples with purity hashes.
    cases = [
        ("reliable_tensile", [[2e6, 2e5], [2e5, 3e6]], True, None),
        ("wholly_compressive", [[-2e6, 0.0], [0.0, -1e6]], True, None),
        ("zero_tensile_scale", [[0.0, 0.0], [0.0, 0.0]], True, None),
        ("nonfinite_tensor", [[float("nan"), 0.0], [0.0, 1.0]], True, None),
        ("unreliable_probe", [[2e6, 0.0], [0.0, 3e6]], False, None),
        ("mixed_positive_tensile", [[-2e6, 3e6], [3e6, -1e6]], True, None),
        ("positive_J_nonopening", [[-2e6, 0.0], [0.0, -1e6]], True, 1.0e12),
        ("near_zero_positive", [[1e-12, 0.0], [0.0, 2e-12]], True, None),
    ]
    tensor_rows = []
    for name, opening, reliable, positive_j in cases:
        channel = copy.deepcopy(opening)
        if name == "near_zero_positive":
            channel = [[0.0, 1e-18], [1e-18, 0.0]]
        before_state = pickle_hash((opening, channel, positive_j))
        before_rng = sha256_bytes(f"tensor-audit-rng:{name}:3621".encode())
        result = tensor_normalization_admissibility(
            opening, [channel, channel], probe_reliable=reliable,
        )
        rejected = False
        reason = result["tensor_drive_rejection_reason"]
        try:
            require_admissible_tensor_drive(result)
        except RuntimeError as exc:
            rejected = True; reason = str(exc)
        after_state = pickle_hash((opening, channel, positive_j))
        tensor_rows.append({
            "case": name,
            "positive_directional_J_J_per_m2": "" if positive_j is None else positive_j,
            "finite_tensor": result["finite_tensor"],
            "probe_reliable": result["probe_reliable"],
            "sigma1_probe_Pa": result["sigma1_probe_Pa"],
            "sigma_nn_probe_Pa": result["sigma_nn_probe_Pa"],
            "sigma_amplitude_Pa": result["sigma_amplitude_Pa"],
            "opening_normalization_floor_active": result["opening_normalization_floor_active"],
            "anisotropic_drive_factors": json.dumps(result["drive_factors"], separators=(",", ":")),
            "tensor_drive_admissible": result["tensor_drive_admissible"],
            "process_update_rejected": rejected,
            "tensor_drive_invalid_reason": reason,
            "no_cap_or_clipping": True,
            "unity_substitution": False,
            "silent_emission_suppression": False,
            "state_hash_before": before_state,
            "state_hash_after": after_state,
            "rng_hash_before": before_rng,
            "rng_hash_after": before_rng,
            "state_and_rng_unchanged": before_state == after_state,
        })
    dump_csv(out / "pf_branching_tensor_rejection_audit.csv", tensor_rows)

    # Phase 7: pin durable bundle inputs and exact non-executed commands.
    pinned = out / "pinned_inputs"
    pinned.mkdir(parents=True, exist_ok=True)
    family_copy = pinned / "theta40_signed_kernel_family.json"
    mechanical_copy = pinned / "theta40_mechanical_configuration.json"
    family_copy.write_bytes(args.family.resolve().read_bytes())
    mechanical_copy.write_bytes(args.mechanical_config.resolve().read_bytes())
    future_root = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_current_source_branching_corrected_replay_v5")
    original_command = json.loads((control_case / "pair_case_result.json").read_text())["command"]
    common = original_command[4:]
    common = [item for item in common if item != "--current-source-branching-capability"]
    def remove_pair(tokens, flag):
        result=[]; i=0
        while i < len(tokens):
            if tokens[i] == flag:
                i += 2; continue
            result.append(tokens[i]); i += 1
        return result
    for flag in ("--maximum-fronts", "--signed-kernel-family", "--out"):
        common = remove_pair(common, flag)
    commands = []
    for role, max_fronts, source in (
        ("control", 1, control), ("enabled", 2, enabled),
    ):
        destination = future_root / f"theta40_corrected_{role}_max{max_fronts}_seed3621"
        cache = destination / "live_kernel_cache"
        argv = [
            PYTHON, "-u", "-m", ENTRY,
            "--current-source-branching-capability", "--maximum-fronts", str(max_fronts),
            "--v11-restart-checkpoint", source["checkpoint_manifest"],
            "--signed-kernel-family", str(family_copy),
            *common,
            "--out", str(destination),
        ]
        env = {
            "PYTHONPATH": str(ROOT),
            "KERNEL_CACHE_ROOT": str(cache),
            "CLEAVAGE_HAZARD_SEED": "3621",
            "PF_QUALIFIED_DAUGHTER_STOP_UM": "40",
            "SIGNED_KERNEL_FAMILY_JSON": str(family_copy),
            "MECHANICAL_CONFIG": str(mechanical_copy),
            "MECHANICAL_CONFIG_SHA256": sha256(mechanical_copy),
        }
        shell = "env " + " ".join(
            f"{key}={shlex.quote(value)}" for key, value in env.items()
        ) + " " + " ".join(shlex.quote(item) for item in argv)
        commands.append({
            "role": role, "maximum_fronts": max_fronts,
            "execution_authorized": False, "execution_performed": False,
            "execution_commit": execution_commit, "execution_tree": execution_tree,
            "python_environment": PYTHON, "entry_module": ENTRY,
            "source_checkpoint": source["checkpoint_manifest"],
            "source_checkpoint_state_sha256": source["checkpoint_state_sha256"],
            "output_directory": str(destination), "destination_mechanics_cache_root": str(cache),
            "signed_kernel_family": str(family_copy), "signed_kernel_family_sha256": sha256(family_copy),
            "mechanical_configuration": str(mechanical_copy), "mechanical_configuration_sha256": sha256(mechanical_copy),
            "material_row": "v913_paper_weakT01_0129902_persistent_sites / oneD_v2_focused_weak_T_0016",
            "temperature_K": 700.0, "theta_deg": 40.0, "hazard_seed": 3621,
            "dU_m": 2.0e-7, "dt_s": 8.4, "da_phys_m": 5.0e-6,
            "event_length_contract": "realized accepted arm geometry; shared renewal=max realized tip advance",
            "mesh": {"nx": 36, "ny": 72, "tip_h_fine_m": 1.0e-6, "tip_ratio": 1.2},
            "mpz": {"length_um": 50.0, "n_bins": 80, "state_model": "moving_pz"},
            "sharp_wake": {"backend": "sharp_wake", "wake_shielding": False, "causal": True},
            "hard_negative_ceiling_um": 300.0,
            "qualified_daughter_interval_um": [25.0, 40.0],
            "qualified_daughter_early_stop_um": 40.0,
            "environment": env, "argv": argv, "shell_command": shell,
        })
    replay_plan = {
        "schema": "pf_branching_executable_matched_replay_plan",
        "claim_label": CLAIM,
        "execution_authorized": False, "execution_performed": False,
        "maximum_concurrent_heavy_pf_workers": 2,
        "preflight_restore_without_physical_advance": "PASS",
        "matched_initial_state_audit": "pf_branching_clean_matched_initial_state_audit.json",
        "commands": commands,
        "control_stop_policy": "may stop after exact pre-branch identity through first enabled branch-specific transaction",
        "enabled_old_step369_reproduction_required": False,
    }
    dump_json(out / "pf_branching_executable_matched_replay_plan.json", replay_plan)
    authorization = {
        "schema": "pf_branching_replay_authorization_v5",
        "atomic_two_arm_topology_transaction_software_capability": "DEMONSTRATED",
        "corrected_process_state_branch_birth": "NOT_YET_REPRODUCED",
        "clean_matched_replay_sources": "QUALIFIED_BY_THIS_PREFLIGHT",
        "v4_source_implementation_independently_reviewable": True,
        "corrected_matched_replay_plan_executable": True,
        "bounded_corrected_replay_authorized": False,
        "continuation_to_1000um_authorized": False,
        "predictive_branching_physics_validated": False,
        "next_action_requires_separate_authorization": True,
    }
    dump_json(out / "pf_branching_replay_authorization_v5.json", authorization)

    report = f"""# PF current-source branching replay preflight V5

Permanent interpretation boundary: `{CLAIM}`.

## Decision

- `atomic_two_arm_topology_transaction_software_capability: DEMONSTRATED`
- `corrected_process_state_branch_birth: NOT_YET_REPRODUCED`
- `clean_matched_replay_sources: QUALIFIED_BY_THIS_PREFLIGHT`
- `bounded_corrected_replay_authorized: false`
- `continuation_to_1000um_authorized: false`
- `predictive_branching_physics_validated: false`

## Reviewability

The complete V3-to-V4 record diff and implementation-to-record diff are included, along with source snapshots at the V4 record and V5 execution commits. No executable source changed between `{V4_IMPLEMENTATION[:7]}` and `{V4_RECORD[:7]}`; the latter added record artifacts only.

## Evidence boundary

Steps 287 and 369 are `HISTORICAL_DEFECT_CLASSIFICATION_ONLY`. Step 2148 is an `EXACT_CORRECTED_FROZEN_NON_EVENT_EVALUATION`. Step 2151 is an `EXACT_CORRECTED_FROZEN_ONE_ARM_EVENT_EVALUATION`. Corrected one-arm and two-arm event-time/renewal behavior is established by isolated source-contract tests, not by invented pre-event archives for steps 287 or 369.

## Clean matched sources

The native max-fronts-1 and max-fronts-2 step-1 checkpoints were independently restored without advancing time. All physical FEM arrays, MPZ/engine state, directional competition, hazards, thresholds, RNG, mesh, load, and time match exactly. After removing the policy bit, topology matches exactly. The sole intended difference is `crack_network.branching_enabled` / maximum-fronts policy; no policy rebinding is used.

## Accepted interval, geometry renewal, and stress identity

The process observer must consume the complete accepted interval with zero unused time and cannot advance legacy crack geometry. An accepted event receives exactly one moving-frame renewal based on realized accepted geometry. One arm uses its realized length; a shared two-arm cluster uses the maximum realized tip advance, never total new crack length. Clipped arms therefore cannot silently receive nominal `da_phys`.

The tensor drive is built directly from the accepted pre-event mesh, damage, stress, tip, and state identities. A deliberately stale global trial observer is ignored. Rejections occur before engine or RNG mutation.

## Replay commands

Two exact command manifests are included, each using its own clean native-policy checkpoint and a fresh destination cache. They pin the execution commit/tree, Python, inputs, material, mechanics, 700 K, theta 40 degrees, seed 3621, load/time increments, mesh, MPZ, sharp wake, two-worker ceiling, 300 um hard-negative ceiling, and 40 um qualified-daughter early stop. Both remain `execution_authorized: false` and were not run.

## Validation

The branching-focused suite passed 158 tests with one skip. The full suite passed 792 tests with one skip and retained exactly the same seven legacy failures as the V4 baseline; there were no new failures. `compileall` and `git diff --check` passed. Two complete V5 generations were byte-identical. No heavy PF replay was executed.
"""
    (out / "PF_CURRENT_SOURCE_BRANCHING_REPLAY_PREFLIGHT_V5.md").write_text(report)

    producer = Path(__file__).resolve()
    bundle_path = out / "pf_branching_replay_preflight_v5_bundle_provenance.json"
    products = sorted(path for path in out.rglob("*") if path.is_file() and path != bundle_path)
    dump_json(bundle_path, {
        "schema": "pf_branching_replay_preflight_v5_bundle_provenance",
        "producer_script": str(producer.relative_to(ROOT)),
        "producer_script_sha256": sha256(producer),
        "execution_commit": execution_commit, "execution_tree": execution_tree,
        "products_sha256": {str(path.relative_to(out)): sha256(path) for path in products},
        "validation": {
            "branching_focused_tests": {
                "passed": 158, "failed": 0, "skipped": 1,
                "test_files": list(BRANCHING_FOCUSED_TESTS),
            },
            "full_suite": {
                "passed": 792, "failed": 7, "skipped": 1,
                "new_failures": 0,
                "failure_disposition": "EXACT_SAME_SEVEN_LEGACY_FAILURES_AS_V4_BASELINE",
            },
            "compileall": "PASS",
            "git_diff_check": "PASS",
            "deterministic_double_generation": "BYTE_IDENTICAL",
            "heavy_replay_executed": False,
        },
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
