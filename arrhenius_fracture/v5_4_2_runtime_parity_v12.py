"""Read-only V5.4.2 transaction/runtime replay through V12 pure state."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any

from .branch_checkpoint_v11 import restore_branch_checkpoint
from .general_multifront_v12 import (
    FrontRuntimeState, MultiFrontRuntimeState, ResourcePolicy, canonical_hash,
)
from .multifront_checkpoint_v12 import load_v11_checkpoint_as_v12


SCHEMA = "v12.v5-4-2-transaction-runtime-parity/1"
FULL_TERMINAL_SCHEMA = "v12.v5-4-2-full-terminal-transaction-runtime-parity/1"
FULL_TERMINAL_ROOT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z"
)
FULL_TERMINAL_CASES = {
    "control": {
        "directory": "theta40_v5_4_2_control_max1_seed3621",
        "accepted_event_count": 85,
        "terminal_step": 662,
        "maximum_forward_reach_um": 300.9175216390797,
        "active_front_count": 1,
    },
    "enabled": {
        "directory": "theta40_v5_4_2_enabled_max2_seed3621",
        "accepted_event_count": 86,
        "terminal_step": 813,
        "assessment_claimed_terminal_step": 807,
        "maximum_forward_reach_um": 302.23630529007903,
        "active_front_count": 2,
        "daughter_lengths_um": [30.00000000000015, 295.0000000000008],
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical_sha(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _transaction_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Fields owned by the pure proposal scheduler and topology transaction."""
    return {
        "step": int(row["step"]),
        "action_type": row["action_type"],
        "candidate_ids": list(row["candidate_ids"]),
        "event_ids": list(row["pending_event_ids"]),
        "completion_times_s": list(row["completion_times_s"]),
        "participating_front_ids": list(row["participating_front_ids"]),
        "realized_arm_lengths_m": list(row["realized_arm_lengths_m"]),
        "topology_fingerprint_before": row["topology_fingerprint_before"],
        "topology_fingerprint_after": row["topology_fingerprint_after"],
        "consumption_result": row["consumption_result"],
    }


def _atomic_scheduler_replay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay complete proposals; correlated members never compete as singles."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    selected = []
    for step in sorted(by_step):
        trials = [
            row for row in by_step[step]
            if row.get("geometry_status") == "realized"
            and row.get("equilibrium_status") == "converged"
            and float(row.get("net_energy_margin_J_per_m") or 0.0) >= 0.0
        ]
        paired = {
            event_id for row in trials if row["action_type"] == "two_arm"
            for event_id in row["pending_event_ids"]
        }
        complete = [
            row for row in trials
            if row["action_type"] == "two_arm"
            or not paired.intersection(row["pending_event_ids"])
        ]
        if not complete:
            continue
        earliest = min(max(row["completion_times_s"]) for row in complete)
        tied = [
            row for row in complete
            if max(row["completion_times_s"]) <= earliest + 1.0e-12
        ]
        choice = min(tied, key=lambda row: canonical_hash({
            "step": step,
            "candidate_ids": row["candidate_ids"],
            "event_ids": row["pending_event_ids"],
            "front_ids": row["participating_front_ids"],
        }))
        selected.append(_transaction_identity(choice))
    return selected


def replay_v5_4_2_runtime_parity(
    repository_root: str | Path, *, source_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repository_root)
    reference_path = repo / (
        "analysis_outputs/pf_current_source_branching_integrated_envelope_v5_4/"
        "pf_branching_v5_2_prefix_parity_reference.json"
    )
    reference = json.loads(reference_path.read_text())
    source = Path(source_root or reference["source_root"])
    cases = {}
    for case_name, expected in sorted(reference["semantic_cases"].items()):
        case = source / case_name
        trials_path = case / "branch_action_trials.jsonl"
        rates_path = case / "directional_rates.jsonl"
        checkpoint_path = case / "checkpoint/latest.json"
        trials = _jsonl(trials_path)
        rates = _jsonl(rates_path)
        replayed = _atomic_scheduler_replay(trials)
        consumed = [row for row in trials if row.get("consumption_result") == "consumed"]
        replay_identity = [(row["step"], row["candidate_ids"]) for row in replayed]
        consumed_identity = [(int(row["step"]), row["candidate_ids"]) for row in consumed]
        expected_identity = list(zip(
            expected["accepted_event_steps"], expected["selected_candidate_identities"]
        ))
        checkpoint = restore_branch_checkpoint(checkpoint_path)
        before_rng = hashlib.sha256(pickle.dumps(checkpoint.state.rng_state, protocol=5)).hexdigest()
        policy = ResourcePolicy(
            branching_mode=("disabled" if len(checkpoint.state.crack_network.active_tip_ids) == 1 else "mechanistic"),
            front_resource_limit=(1 if len(checkpoint.state.crack_network.active_tip_ids) == 1 else None),
        )
        runtime = load_v11_checkpoint_as_v12(checkpoint, resource_policy=policy)
        restored = MultiFrontRuntimeState.from_dict(runtime.to_dict())
        after_rng = hashlib.sha256(pickle.dumps(checkpoint.state.rng_state, protocol=5)).hexdigest()
        same_tip = all(
            row.get("controlling_scalar_K_tip_id") == row.get("tensor_probe_tip_id")
            for row in rates
        )
        archived_state_rows_exact = True
        for row in rates:
            front_state = FrontRuntimeState(
                front_id=row["tip_id"],
                competition_state={
                    "candidate_id": row["candidate_id"],
                    "accumulated_integrated_hazard_H": row[
                        "accumulated_integrated_hazard_H"
                    ],
                    "current_threshold_H_star": row["current_threshold_H_star"],
                    "directional_event_ordinal": row["directional_event_ordinal"],
                    "pending_event_ids": row["pending_event_ids"],
                    "selected_event_candidate_id": row["selected_event_candidate_id"],
                },
                candidate_ids=(row["candidate_id"],),
                lineage_rng_state={
                    "authoritative_case_rng_sha256": expected["state_rng_sha256"],
                    "front_competition_rng_and_threshold_sha256": expected[
                        "front_competition_rng_and_threshold_sha256"
                    ],
                },
                interval_count=int(row["step"]),
            )
            archived_state_rows_exact &= (
                FrontRuntimeState.from_dict(front_state.to_dict()).to_dict()
                == front_state.to_dict()
            )
        branch_rows = []
        branch_path = case / "branch_events.csv"
        if branch_path.exists():
            branch_rows = list(csv.DictReader(branch_path.open()))
        birth = next((row for row in branch_rows if int(row["step"]) == 369), None)
        branch_exact = None
        control_policy_exact = None
        if birth is None and not expected["branch_birth_at_step_369"]:
            step_369 = next(row for row in replayed if row["step"] == 369)
            step_370 = next(row for row in replayed if row["step"] == 370)
            control_policy_exact = (
                step_369["action_type"] == step_370["action_type"] == "one_arm"
            )
        renewal_distance = None
        if birth is not None:
            arms = tuple(json.loads(birth["arm_lengths_m"]))
            renewal_distance = max(arms)
            expected_birth = expected["branch_births"][0]
            branch_exact = (
                json.loads(birth["event_ids_consumed"])
                == next(row["event_ids"] for row in replayed if row["step"] == 369)
                and birth["parent_front"] == expected_birth["parent_front"]
                and json.loads(birth["arm_front_ids"]) == expected_birth["arm_front_ids"]
                and birth["topology_fingerprint"] == expected_birth["topology_fingerprint"]
            )
        cases[case_name] = {
            "source_hashes": {
                "branch_action_trials_jsonl": _sha(trials_path),
                "directional_rates_jsonl": _sha(rates_path),
                "checkpoint_manifest": _sha(checkpoint_path),
                "checkpoint_state": _sha(checkpoint_path.with_name(
                    json.loads(checkpoint_path.read_text())["state_file"]
                )),
            },
            "accepted_event_count": len(consumed),
            "replayed_event_count": len(replayed),
            "replayed_equals_consumed": replay_identity == consumed_identity,
            "consumed_equals_v5_4_2_reference": consumed_identity == expected_identity,
            "prebranch_one_arm_choices_exact": all(
                row["action_type"] == "one_arm" for row in replayed if row["step"] < 369
            ),
            "step_369_correlated_birth_exact": branch_exact,
            "step_369_correlated_birth_applicability": (
                "applicable" if birth is not None else "not_applicable"
            ),
            "step_369_branch_disabled_policy_behavior_exact": control_policy_exact,
            "step_369_renewal_distance_m": renewal_distance,
            "same_tip_scalar_tensor_ownership_exact": same_tip,
            "archived_directional_state_row_count": len(rates),
            "every_archived_action_threshold_ordinal_rng_binding_roundtrips": (
                archived_state_rows_exact
            ),
            "competition_action_threshold_ordinal_rng_roundtrip_exact": (
                restored.to_dict() == runtime.to_dict() and before_rng == after_rng
            ),
            "terminal_topology_lossless": (
                restored.crack_network.to_json() == checkpoint.state.crack_network.to_json()
            ),
            "active_front_count": len(runtime.active_front_ids),
            "renewal_ledgers_sha256": expected["renewal_ledgers_sha256"],
            "state_rng_sha256": expected["state_rng_sha256"],
        }
    gates = {
        "n1_accepted_event_identities_unchanged": all(
            row["replayed_equals_consumed"] and row["consumed_equals_v5_4_2_reference"]
            for row in cases.values()
        ),
        "n2_prebranch_sequence_unchanged": cases[
            "theta40_corrected_enabled_max2_seed3621"
        ]["prebranch_one_arm_choices_exact"],
        "step_369_two_arm_transaction_unchanged": cases[
            "theta40_corrected_enabled_max2_seed3621"
        ]["step_369_correlated_birth_exact"],
        "same_tip_ownership_exact": all(
            row["same_tip_scalar_tensor_ownership_exact"] for row in cases.values()
        ),
        "checkpoint_runtime_roundtrip_exact": all(
            row["competition_action_threshold_ordinal_rng_roundtrip_exact"]
            and row["every_archived_action_threshold_ordinal_rng_binding_roundtrips"]
            and row["terminal_topology_lossless"] for row in cases.values()
        ),
    }
    return {
        "schema": SCHEMA,
        "qualification": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "scope": "V5.4.2 transaction/runtime parity; no PF or FEM execution",
        "source_reference_sha256": _sha(reference_path),
        "source_root": str(source),
        "cases": cases,
        "gates": gates,
    }


def _selected_front_ids(
    transaction: dict[str, Any], rates_by_step: dict[int, list[dict[str, Any]]],
) -> list[str]:
    candidates = set(transaction["candidate_ids"])
    rows = rates_by_step.get(int(transaction["step"]), [])
    tips = {
        row["tip_id"] for row in rows
        if row["candidate_id"] in candidates
        and row.get("selected_event_candidate_id") in candidates
    }
    if not tips:
        tips = {
            row["tip_id"] for row in rows if row["candidate_id"] in candidates
        }
    return sorted(tips)


def replay_full_v5_4_2_terminal_runtime_parity(
    repository_root: str | Path, *, source_root: str | Path = FULL_TERMINAL_ROOT,
) -> dict[str, Any]:
    """Replay the complete V5.4.2 terminal transaction archives read-only.

    The archive is authoritative.  In particular, the enabled terminal action
    is step 813; step 807 is retained as a separately reported assessment
    discrepancy rather than treated as an expected value.
    """
    del repository_root  # Kept in the API to parallel the prefix replay.
    source = Path(source_root)
    cases: dict[str, Any] = {}
    for role, expected in FULL_TERMINAL_CASES.items():
        case = source / expected["directory"]
        trials_path = case / "branch_action_trials.jsonl"
        rates_path = case / "directional_rates.jsonl"
        checkpoint_path = case / "checkpoint/latest.json"
        fronts_path = case / "fronts.csv"
        run_complete_path = case / "run_complete.json"
        visual_path = case / "v11_visual_snapshot.json"
        trials = _jsonl(trials_path)
        rates = _jsonl(rates_path)
        rates_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rates:
            rates_by_step[int(row["step"])].append(row)
        replayed = _atomic_scheduler_replay(trials)
        consumed_rows = [
            row for row in trials if row.get("consumption_result") == "consumed"
        ]
        consumed = [_transaction_identity(row) for row in consumed_rows]
        for row in replayed:
            row["selected_front_ids"] = _selected_front_ids(row, rates_by_step)
            row["renewal_owner_id"] = row["selected_front_ids"][0]
            row["renewal_distance_m"] = max(row["realized_arm_lengths_m"])
        for row in consumed:
            row["selected_front_ids"] = _selected_front_ids(row, rates_by_step)
            row["renewal_owner_id"] = row["selected_front_ids"][0]
            row["renewal_distance_m"] = max(row["realized_arm_lengths_m"])
        active_count = 1
        for replay_row, consumed_row in zip(replayed, consumed):
            replay_row["pre_active_front_count"] = active_count
            consumed_row["pre_active_front_count"] = active_count
            if replay_row["action_type"] == "two_arm":
                active_count += 1
            replay_row["post_active_front_count"] = active_count
            consumed_row["post_active_front_count"] = active_count

        checkpoint_manifest = json.loads(checkpoint_path.read_text())
        run_complete = json.loads(run_complete_path.read_text())
        visual = json.loads(visual_path.read_text())
        checkpoint = restore_branch_checkpoint(checkpoint_path)
        policy = ResourcePolicy(
            branching_mode="disabled" if role == "control" else "mechanistic",
            front_resource_limit=1 if role == "control" else None,
        )
        runtime = load_v11_checkpoint_as_v12(checkpoint, resource_policy=policy)
        restored = MultiFrontRuntimeState.from_dict(runtime.to_dict())
        state_path = checkpoint_path.with_name(checkpoint_manifest["state_file"])

        competition_rows = [{
            "step": int(row["step"]),
            "tip_id": row["tip_id"],
            "candidate_id": row["candidate_id"],
            "accumulated_integrated_hazard_H": row[
                "accumulated_integrated_hazard_H"
            ],
            "current_threshold_H_star": row["current_threshold_H_star"],
            "directional_event_ordinal": row["directional_event_ordinal"],
            "pending_event_ids": row["pending_event_ids"],
            "selected_event_candidate_id": row["selected_event_candidate_id"],
        } for row in rates]
        directional_state_rows_roundtrip_exact = True
        for row in rates:
            front_state = FrontRuntimeState(
                front_id=row["tip_id"],
                competition_state={
                    "candidate_id": row["candidate_id"],
                    "accumulated_integrated_hazard_H": row[
                        "accumulated_integrated_hazard_H"
                    ],
                    "current_threshold_H_star": row["current_threshold_H_star"],
                    "directional_event_ordinal": row["directional_event_ordinal"],
                    "pending_event_ids": row["pending_event_ids"],
                    "selected_event_candidate_id": row[
                        "selected_event_candidate_id"
                    ],
                },
                candidate_ids=(row["candidate_id"],),
                lineage_rng_state={
                    "authoritative_checkpoint_rng_sha256": hashlib.sha256(
                        pickle.dumps(checkpoint.state.rng_state, protocol=5)
                    ).hexdigest(),
                },
                interval_count=int(row["step"]),
            )
            directional_state_rows_roundtrip_exact &= (
                FrontRuntimeState.from_dict(front_state.to_dict()).to_dict()
                == front_state.to_dict()
            )
        with fronts_path.open(newline="") as stream:
            front_rows = list(csv.DictReader(stream))
        terminal_front_rows = [
            row for row in front_rows if int(row["step"]) == expected["terminal_step"]
        ]
        accepted_terminal_step = max(int(row["step"]) for row in consumed)
        front_terminal_step = max(int(row["step"]) for row in front_rows)

        branch: dict[str, Any] | None = None
        branch_path = case / "branch_events.csv"
        if branch_path.exists():
            with branch_path.open(newline="") as stream:
                branch_rows = list(csv.DictReader(stream))
            raw = next(row for row in branch_rows if int(row["step"]) == 369)
            replay_birth = next(row for row in replayed if row["step"] == 369)
            arm_lengths = json.loads(raw["arm_lengths_m"])
            branch = {
                "step": 369,
                "action_type": replay_birth["action_type"],
                "event_ids": json.loads(raw["event_ids_consumed"]),
                "replayed_event_ids": replay_birth["event_ids"],
                "completion_times_s": replay_birth["completion_times_s"],
                "completion_time_difference_s": float(raw[
                    "completion_time_difference_s"
                ]),
                "parent_front": raw["parent_front"],
                "daughter_front_ids": json.loads(raw["arm_front_ids"]),
                "arm_lengths_m": arm_lengths,
                "renewal_owner": raw["parent_front"],
                "renewal_distance_m": max(arm_lengths),
                "topology_fingerprint": raw["topology_fingerprint"],
                "exact": (
                    replay_birth["action_type"] == "two_arm"
                    and json.loads(raw["event_ids_consumed"])
                    == replay_birth["event_ids"]
                ),
            }

        control_policy = None
        if role == "control":
            step_369 = next(row for row in consumed if row["step"] == 369)
            step_370 = next(row for row in consumed if row["step"] == 370)
            control_policy = {
                "step_369_branch_disabled_policy_behavior_exact": (
                    step_369["action_type"] == step_370["action_type"] == "one_arm"
                ),
                "step_369_correlated_birth_exact": None,
                "step_369_correlated_birth_applicability": "not_applicable",
            }

        gates = {
            "accepted_event_count_exact": (
                len(consumed) == expected["accepted_event_count"]
            ),
            "pure_scheduler_replay_exact": replayed == consumed,
            "proposal_action_front_candidate_event_time_exact": all(
                row["selected_front_ids"] for row in replayed
            ),
            "renewal_owner_distance_and_active_counts_exact": (
                replayed == consumed
                and active_count == expected["active_front_count"]
            ),
            "terminal_step_exact_to_authoritative_archive": (
                accepted_terminal_step == front_terminal_step
                == int(checkpoint_manifest["event_counters"]["accepted_steps"])
                == int(visual["step"]) == expected["terminal_step"]
            ),
            "terminal_reach_exact": (
                float(checkpoint_manifest["projected_extension_m"]) * 1e6
                == expected["maximum_forward_reach_um"]
                == float(visual["growth_metrics"][
                    "max_forward_projected_extension_um"
                ])
            ),
            "active_front_count_exact": (
                sum(row["status"] == "active" for row in terminal_front_rows)
                == checkpoint_manifest["active_front_count"]
                == expected["active_front_count"]
            ),
            "topology_roundtrip_exact": (
                restored.to_dict() == runtime.to_dict()
                and restored.crack_network.to_json()
                == checkpoint.state.crack_network.to_json()
            ),
            "visual_and_checkpoint_active_front_identity_exact": (
                sorted(visual["active_tip_ids"])
                == sorted(checkpoint_manifest["active_front_ids"])
            ),
            "competition_actions_thresholds_ordinals_rng_recorded": (
                bool(competition_rows)
                and all(row["current_threshold_H_star"] is not None for row in competition_rows)
                and bool(checkpoint_manifest.get("has_rng_state"))
                and checkpoint.state.rng_state is not None
            ),
            "competition_action_threshold_ordinal_rows_roundtrip_exact": (
                directional_state_rows_roundtrip_exact
            ),
            "run_complete_topology_action_count_exact": (
                run_complete["validation"]["topology_actions"]
                == expected["accepted_event_count"]
            ),
        }
        if branch is not None:
            gates["step_369_atomic_two_arm_exact"] = branch["exact"]
            gates["daughter_lengths_exact"] = sorted(
                float(row["arclength_m"]) * 1e6
                for row in terminal_front_rows if row["parent_front_id"]
            ) == expected["daughter_lengths_um"]
        if control_policy is not None:
            gates["step_369_branch_disabled_policy_exact"] = control_policy[
                "step_369_branch_disabled_policy_behavior_exact"
            ]

        cases[role] = {
            "source_directory": expected["directory"],
            "source_hashes": {
                "branch_action_trials_jsonl": _sha(trials_path),
                "directional_rates_jsonl": _sha(rates_path),
                "fronts_csv": _sha(fronts_path),
                "checkpoint_manifest": _sha(checkpoint_path),
                "checkpoint_state": _sha(state_path),
                "run_complete": _sha(run_complete_path),
                "visual_snapshot": _sha(visual_path),
            },
            "accepted_event_count": len(consumed),
            "replayed_event_count": len(replayed),
            "accepted_event_steps": [row["step"] for row in consumed],
            "terminal_step": accepted_terminal_step,
            "assessment_claimed_terminal_step": expected.get(
                "assessment_claimed_terminal_step"
            ),
            "assessment_terminal_step_matches_archive": (
                expected.get("assessment_claimed_terminal_step", accepted_terminal_step)
                == accepted_terminal_step
            ),
            "maximum_forward_reach_um": (
                float(checkpoint_manifest["projected_extension_m"]) * 1e6
            ),
            "transaction_identity_sha256": _canonical_sha(consumed),
            "competition_action_threshold_ordinal_sha256": _canonical_sha(
                competition_rows
            ),
            "checkpoint_rng_state_sha256": _canonical_sha(
                {
                    "pickle_protocol_5_sha256": hashlib.sha256(
                        pickle.dumps(checkpoint.state.rng_state, protocol=5)
                    ).hexdigest()
                }
            ),
            "terminal_topology_sha256": _canonical_sha(
                checkpoint_manifest["crack_network"]
            ),
            "owner_transition_and_renewal_evidence": branch,
            "branch_disabled_policy_evidence": control_policy,
            "gates": gates,
            "qualification": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        }

    global_gates = {
        "control_full_terminal_exact": cases["control"]["qualification"] == "PASS",
        "enabled_full_terminal_exact": cases["enabled"]["qualification"] == "PASS",
        "assessment_step_807_discrepancy_disclosed": (
            cases["enabled"]["assessment_claimed_terminal_step"] == 807
            and cases["enabled"]["terminal_step"] == 813
            and not cases["enabled"]["assessment_terminal_step_matches_archive"]
        ),
    }
    return {
        "schema": FULL_TERMINAL_SCHEMA,
        "scope": "read-only pure-scheduler and checkpoint-adapter replay; no PF or FEM execution",
        "source_root": str(source),
        "cases": cases,
        "gates": global_gates,
        "qualification": "PASS" if all(global_gates.values()) else "FAIL_CLOSED",
    }


__all__ = [
    "FULL_TERMINAL_ROOT", "FULL_TERMINAL_SCHEMA", "SCHEMA",
    "replay_full_v5_4_2_terminal_runtime_parity", "replay_v5_4_2_runtime_parity",
]
