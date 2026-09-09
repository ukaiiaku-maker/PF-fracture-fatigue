#!/usr/bin/env python3
"""Fresh, canonical single-front parents only; never imports a V12 checkpoint."""
import argparse
import copy
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    ROOT, CASES, ROWS, case_parts, common_arguments, campaign_environment, validate_inputs, registry_row,
    atomic_json, sha256,
)

BOUNDARY = "BRANCHING_KINETICS_MODEL_UNCALIBRATED"


class FirstParentCapture:
    def __init__(self, output):
        self.output = Path(output)
        self.interval_count = 0
        self.pre = None

    def begin(self, state, engine, physical_time, opening, runtime):
        from arrhenius_fracture.current_source_runtime_bindings import runtime_binding_inventory, callable_id
        from arrhenius_fracture.persistent_site_source_v10221 import _persistent_emit
        from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
        if callable_id(engine.mpz._emit) != callable_id(_persistent_emit):
            raise RuntimeError("clean parent is not executing intended persistent emission")
        if state.crack_network.branching_enabled or len(state.crack_network.active_tip_ids) != 1:
            raise RuntimeError("clean parent must be branch-disabled and single-front")
        if state.competition.consumed_event_ids:
            raise RuntimeError("first-parent run continued beyond an accepted cleavage")
        self.pre = (state, copy.deepcopy(_capture_shared_engine(engine)), physical_time, opening, copy.deepcopy(runtime))
        self.bindings = runtime_binding_inventory(engine)

    def accept(self, *, checkpoint, context, result, solved_pre_event, pre_event_sigma, args, cfg):
        from arrhenius_fracture.branch_checkpoint_v11 import write_branch_checkpoint
        from arrhenius_fracture.sharp_front_v11_branching import _hash, _mesh_identity
        from scripts.v13_value_fingerprint import physical_state_fingerprint
        self.interval_count += 1
        selected = [trial for trial in result.trials if trial.selected]
        atomic_json(self.output / "progress.json", {
            "status": "PARENT_CAPTURED" if selected else "LOADING",
            "step": context.step, "physical_time_s": checkpoint.physical_time_s,
            "opening_m": checkpoint.accepted_load, "accepted_interval_count": self.interval_count,
        })
        if not selected:
            return False
        if len(selected) != 1 or selected[0].proposal.action_type != "one_arm":
            raise RuntimeError("parent is not exactly one accepted single-arm event")
        chosen = selected[0]
        state, process, start_time, start_opening, pre_runtime = self.pre
        pre = replace(checkpoint, state=state, shared_process_state=process,
            physical_time_s=start_time, accepted_load=start_opening,
            provider_runtime=pre_runtime,
            boundary_condition_state={"opening_m": start_opening}, mesh_identity=_mesh_identity(state.mesh),
            topology_fingerprint=_hash(state.crack_network),
            front_competitions={tip: state.competition for tip in state.crack_network.active_tip_ids},
            branch_clusters=(), projected_extension_m=0.0, physical_extension_m=0.0, termination_reason=None)
        before = write_branch_checkpoint(pre, self.output / "parent/pre_cleavage.json")
        after = write_branch_checkpoint(checkpoint, self.output / "parent/accepted_single.json")
        payload = {
            "schema": "v13.clean-parent-event-context/1", "context": context,
            "pre_cleavage_checkpoint": pre, "accepted_single_checkpoint": checkpoint,
            "solved_pre_event_state": solved_pre_event,
            "pre_event_sigma": pre_event_sigma,
            "canonical_result": result, "args": vars(args), "configuration": cfg,
        }
        target = self.output / "parent/event_context.pkl"
        raw = pickle.dumps(payload, protocol=5)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, target)
        proposal = chosen.proposal
        record = {
            "schema": "v13.clean-physical-parent/1", "boundary": BOUNDARY,
            "status": "FIRST_BASELINE_SINGLE_CAPTURED", "fresh_initialization": True,
            "historical_process_state_used": False, "installed_bindings": self.bindings,
            "canonical_source_function": "sharp_front_v11_branching.run_2d(maximum_fronts=1)",
            "winning_candidate_id": proposal.member_candidate_ids[0],
            "event_id": proposal.member_event_ids[0], "event_ordinal": proposal.member_event_ordinals[0],
            "raw_first_passage_s": proposal.completion_times_s[0],
            "accepted_endpoint_s": checkpoint.physical_time_s, "accepted_opening_m": checkpoint.accepted_load,
            "pre_cleavage_checkpoint_sha256": before["state_sha256"],
            "accepted_single_checkpoint_sha256": after["state_sha256"],
            "event_context_sha256": hashlib.sha256(raw).hexdigest(),
            "pre_process_sha256": physical_state_fingerprint(process),
            "accepted_process_sha256": physical_state_fingerprint(checkpoint.shared_process_state),
            "cleavage_rng_and_clocks_sha256": physical_state_fingerprint(checkpoint.state.competition),
            "process_rng_sha256": physical_state_fingerprint(checkpoint.shared_process_state["engine_fields"]["_hazard_rng"]),
            "single_trial_energy_release_J_per_m": chosen.result.energy_release_J_per_m,
            "single_trial_cost_J_per_m": chosen.result.hazard_dissipation_J_per_m,
            "accepted_interval_count": self.interval_count,
            "parent_call_count_in_accepted_event_interval": 1,
        }
        atomic_json(self.output / "parent/parent_record.json", record)
        return True


def case_run(args):
    from arrhenius_fracture import sharp_front_v11_branching as production
    from arrhenius_fracture import sharp_front_v10_2_27 as paper
    case_root = args.output_root / args.case
    case_root.mkdir(parents=True, exist_ok=True)
    claim = case_root / "launch_claim.json"
    with claim.open("x") as stream:
        json.dump({"pid": os.getpid(), "case": args.case, "fresh_only": True}, stream)
    _, T, canonical, alias = case_parts(args.case)
    family = args.family.resolve()
    validation = validate_inputs(family)
    environment = campaign_environment(family)
    for key in ("V11_BRANCH_RESTART_CHECKPOINT", "PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT"):
        environment.pop(key, None)
        os.environ.pop(key, None)
    environment.update({"PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    os.environ.update(environment)
    argv = common_arguments(alias, T, family, case_root)
    argv[argv.index("--steps")+1] = "2000"
    argv.extend(["--maximum-fronts", "1"])
    row, row_hash = registry_row(canonical)
    atomic_json(case_root / "launch.json", {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "case": args.case, "arguments": argv, "material_row": row, "material_row_sha256": row_hash,
        "family_validation": validation, "baseline_seed": 3621, "fresh_initialization": True,
        "first_cleavage_stop": True, "maximum_accepted_intervals": 2000, "boundary": BOUNDARY})
    capture = FirstParentCapture(case_root)
    # Match the accepted four-class fresh initializer's explicit registry
    # routing. The generic paper defaults point at a different, uninstalled
    # registry, including older weak-T/ceramic candidate IDs.
    original_paper = paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS
    paper.DEFAULT_REGISTRY = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv"
    paper.SELECTION_RECORD = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json"
    paper.VALID_OPTIONS = {alias_value: candidate for candidate, alias_value in ROWS.values()}
    original = production.run_2d
    production.run_2d = partial(original, parent_capture=capture)
    try:
        production.main(argv)
        status = "PARENT_CAPTURED" if (case_root / "parent/parent_record.json").is_file() else "NO_PARENT_BEFORE_EXISTING_TERMINATION"
        atomic_json(case_root / "terminal.json", {"status": status, "fresh_only": True})
    except Exception as exc:
        atomic_json(case_root / "terminal.json", {"status": "STOPPED_REQUIRES_REASON_CLASSIFICATION",
            "exception_type": type(exc).__name__, "reason": str(exc), "traceback": traceback.format_exc(),
            "last_checkpoint": str(case_root / "checkpoint/latest.json"), "automatic_retry": False})
        raise
    finally:
        production.run_2d = original
        paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS = original_paper


def queue(args):
    args.output_root.mkdir(parents=True, exist_ok=True)
    active = {}
    pending = [case for case in CASES if not (args.output_root/case/"terminal.json").exists()]
    if any((args.output_root/case/"launch_claim.json").exists() for case in pending):
        raise RuntimeError("a pending case has an existing launch claim; no automatic restart")
    with (args.output_root / "queue_claim.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "maximum_workers": args.workers}, stream)
    while pending or active:
        free = shutil.disk_usage(args.output_root).free
        while pending and len(active) < args.workers and free >= 1024**3:
            case = pending.pop(0)
            output = args.output_root/case
            output.mkdir(exist_ok=True)
            log = (output/"worker.log").open("a")
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MPLCONFIGDIR="/private/tmp/pf-current-source-v13-parent-mpl")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                "--output-root", str(args.output_root.resolve()), "--family", str(args.family.resolve())]
            active[case] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        for case, (process, log) in list(active.items()):
            if process.poll() is not None:
                log.close()
                if not (args.output_root/case/"terminal.json").exists():
                    atomic_json(args.output_root/case/"terminal.json", {"status": "WORKER_EXIT_WITHOUT_TERMINAL_RECORD", "exit_code": process.returncode})
                del active[case]
        atomic_json(args.output_root/"queue_status.json", {"active": {c: p.pid for c, (p, _) in active.items()},
            "pending": pending, "free_bytes": free, "completed": [c for c in CASES if (args.output_root/c/"terminal.json").exists()]})
        if pending and not active and free < 1024**3:
            raise RuntimeError("less than 1 GiB durable space; pending cases not launched")
        if active:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    case_run(args) if args.case else queue(args)


if __name__ == "__main__":
    main()
