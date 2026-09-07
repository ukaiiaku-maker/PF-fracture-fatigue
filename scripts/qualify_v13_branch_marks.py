#!/usr/bin/env python3
"""Frozen conditional-mark validation only: no trajectory, mechanics solve or renewal.

Eight historical full states are inputs, not endorsed persistent-law histories.
The owner-cached opening is a frozen scalar test stimulus, not a reconstructed
candidate-specific current tensor or a newly accepted physical cleavage event.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import beta

from arrhenius_fracture.conditional_branch_mark_v13 import (
    BOUNDARY, ParentCleavageEvent, BranchOpportunityState, BranchMarkParameters,
    CompanionChannel, conditional_probabilities, draw_branch_mark, digest,
    apply_conditional_branch_mark,
)
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from scripts.v13_value_fingerprint import physical_state_fingerprint as _hash
from arrhenius_fracture.accepted_step_overlay_v13 import advance_marked_accepted_step
from scripts.v13_frozen_support import ATLAS, saved_owner, initialized_engine, rng_hash
from scripts.qualify_v13_process_restore import sha, safe_json, selected_state
from scripts.trace_v13_canonical_parent import fixture, replay


def parent_overlay_parity():
    data = fixture()
    records = []
    for enabled in (False, True):
        with initialized_engine(data[0].shared_process_state) as engine:
            before = _capture_shared_engine(engine)
            def forbidden(*args):
                raise RuntimeError("precleavage/disabled path invoked branch machinery")
            pre = apply_conditional_branch_mark(parent=None, baseline_single_state=data[0].state,
                parameters=BranchMarkParameters(enabled, 1e-9), channels=forbidden, exact_pair_trial=forbidden)
            assert pre.state is data[0].state
            assert _hash(before) == _hash(_capture_shared_engine(engine))
            adapter_log = []
            def step(accepted, context, **callbacks):
                def parent_factory(result, ctx):
                    selected = next(t.proposal for t in result.trials if t.selected)
                    return ParentCleavageEvent(("b00000000",), "owner:b00000000",
                        selected.member_candidate_ids[0], selected.member_event_ids[0], selected.member_event_ordinals[0],
                        ctx.physical_time_s + ctx.duration_s, data[1]["applied_displacement_m"], ctx.accepted_state_id,
                        _hash(_capture_shared_engine(engine)), _hash(result.state.competition), rng_hash(engine))
                marked = advance_marked_accepted_step(accepted, context,
                    branch_parameters=BranchMarkParameters(enabled, 1e-9),
                    parent_factory=parent_factory if enabled else forbidden,
                    channels_factory=lambda *_: (), exact_pair_trial_factory=forbidden, **callbacks)
                assert marked.state is marked.canonical_result.state
                adapter_log.append(marked.mark_result.disposition)
                return marked.canonical_result
            record, accepted, process = replay(engine, data, step_function=step)
            full = _hash(_capture_shared_engine(process["engine"]))
            parent = ParentCleavageEvent(("b00000000",), "owner:b00000000",
                record["winning_candidate_id"], record["primary_event_id"], record["primary_event_ordinal"],
                record["accepted_single_arm_endpoint_s"], record["accepted_opening_m"],
                record["starting_accepted_state_id"], full,
                digest(record["directional_clocks_after"]), rng_hash(process["engine"]))
            # No new pair mechanics authorized: an empty admissible inventory
            # exercises the exact no-companion fallback on the actual parent.
            result = apply_conditional_branch_mark(parent=parent, baseline_single_state=accepted.state,
                parameters=BranchMarkParameters(enabled, 1e-9), channels=lambda: (), exact_pair_trial=forbidden)
            assert result.state is accepted.state
            assert full == _hash(_capture_shared_engine(process["engine"]))
            records.append({"enabled": enabled, "parent_record_sha256": digest(safe_json(record)),
                "complete_process_sha256": full, "cleavage_clocks_sha256": parent.cleavage_rng_sha256,
                "process_rng_sha256": parent.process_rng_sha256, "disposition": result.disposition,
                "exact_baseline_object_retained": True, "precleavage_exact": True,
                "actual_accepted_step_overlay_disposition": adapter_log})
    assert all(records[0][key] == records[1][key] for key in (
        "parent_record_sha256", "complete_process_sha256", "cleavage_clocks_sha256", "process_rng_sha256"))
    return records


def binomial_interval(hits, n, alpha=.001):
    return (0.0 if hits == 0 else float(beta.ppf(alpha/2, hits, n-hits+1)),
            1.0 if hits == n else float(beta.ppf(1-alpha/2, hits+1, n-hits)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    if args.samples < 1000:
        raise ValueError("at least 1000 frozen draws per configuration")
    parity = parent_overlay_parity()
    cases = [f"{m}_{T}K" for m in ("Peak", "DBTT", "weakT", "ceramic") for T in (300, 1000)]
    rows, states = [], []
    ev = 1.602176634e-19
    # Explicit mathematical sensitivity settings, not fitted material values.
    settings = [(tau, q, ov) for tau in (1e-12, 1e-9, 1e-6) for q, ov in ((0, 0), (.1, .1), (.3, .1))]
    for case in cases:
        path = ATLAS / "continued_1000um_runs_v2" / case / "checkpoint/latest.v12.pkl"
        original = sha(path)
        checkpoint, saved = saved_owner(case)
        payload = saved.complete_checkpoint_payload()
        owner = next(r for r in checkpoint.runtime.process_regions.values() if r.process_engine_id == saved.engine_id)
        front_id = sorted(owner.member_front_ids)[0]
        front = checkpoint.runtime.front_runtimes[front_id]
        candidates = front.competition_state["candidates"]
        if len(candidates) != 2:
            raise RuntimeError("frozen two-plane experiment requires the unchanged two-candidate inventory")
        T = float(case.rsplit("_", 1)[1][:-1])
        with initialized_engine(payload) as engine:
            before = _hash(_capture_shared_engine(engine))
            # Read the exact existing constitutive barrier/rate at the saved
            # owner opening. No sigma_tip fixed-point update or tensor swap.
            sigma = float(engine._opening_sigma_local_Pa)
            effective, raw, barrier = copy.deepcopy(engine).lambda_cleave(sigma, T)
            state_hash = before
            state = {"case": case, "source_checkpoint": str(path), "checkpoint_sha256": original,
                "serialized_complete_engine_sha256": saved.checkpoint_payload_sha256,
                "rehydrated_complete_process_sha256": state_hash,
                "owner_id": owner.owner_id, "front_id": front_id,
                "accepted_state_id": checkpoint.runtime.accepted_state_id,
                "stress_field_state_id": checkpoint.runtime.stress_field_state_id,
                "accepted_stress_field_sha256": _hash(checkpoint.accepted_stress_field),
                "stored_energy_J_per_m": checkpoint.accepted_fem_state.stored_energy_J_per_m,
                "frozen_owner_opening_sigma_Pa": sigma,
                "effective_cleavage_rate_per_s": effective, "raw_cleavage_arrival_per_s": raw,
                "raw_barrier_J": barrier, "multihit_order": engine.f.m_hits, "tau_c_s": engine.f.tau_c,
                "process_fields": selected_state(copy.deepcopy(engine)), "candidate_inventory": candidates,
                "scope": "FROZEN_CONDITIONAL_SCALAR_STIMULUS_NOT_PHYSICAL_PARENT_REPLAY",
                "geometric_scope": "inventory_only_exact_pair_admissibility_not_evaluated",
                "baseline_material_manifest_sha256": sha(path.parents[1] / "selected_material_manifest_v10_2_22.csv")}
            primary, companion = candidates
            parent = ParentCleavageEvent((case, front_id), owner.owner_id,
                primary["candidate_id"], "frozen-test-opportunity-not-physical-event", 1,
                checkpoint.physical_time_s, checkpoint.accepted_opening_m,
                checkpoint.runtime.accepted_state_id, state_hash,
                digest(front.competition_state), rng_hash(engine))
            channel = CompanionChannel(companion["candidate_id"], raw, float(engine.f.m_hits), barrier,
                T, 0.0, float(engine.mpz.length_m), state_hash)
            state["mark_channel"] = asdict(channel)
            states.append(state)
            for tau, q, ov in settings:
                params = BranchMarkParameters(True, tau, q*ev, ov*ev, branch_seed=130001)
                p = conditional_probabilities([channel], params).companions[0][1]
                outcomes = hashlib.sha256()
                hits = 0
                for ordinal in range(args.samples):
                    mark = draw_branch_mark(BranchOpportunityState(parent, ordinal), [channel], params)
                    hit = int(mark.companion_id is not None)
                    hits += hit
                    outcomes.update(bytes([hit]))
                lo, hi = binomial_interval(hits, args.samples)
                passed = lo <= p <= hi
                if not passed:
                    raise RuntimeError(f"frozen Monte Carlo disagrees with analytic probability: {case}/{tau}/{q}")
                rows.append({"case": case, "temperature_K": T, "exposure_time_s": tau,
                    "junction_barrier_eV": q, "overlap_barrier_eV": ov,
                    "analytic_probability": p, "mark_count": hits, "samples": args.samples,
                    "sample_probability": hits/args.samples, "binomial_99p9_low": lo,
                    "binomial_99p9_high": hi, "pass": passed,
                    "branch_arrival_per_s": channel.arrival_rate(params),
                    "raw_lambda_tau_c": raw*engine.f.tau_c,
                    "branch_rng_seed": params.branch_seed, "draw_sequence_sha256": outcomes.hexdigest()})
            assert before == _hash(_capture_shared_engine(engine))
        assert sha(path) == original
        print(case, "9 frozen Monte Carlo configurations passed", flush=True)
    output = {"schema": "v13.frozen-branch-mark-qualification/1", "status": "PASS", "boundary": BOUNDARY,
        "producer_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_files": {name: sha(name) for name in (
            "arrhenius_fracture/conditional_branch_mark_v13.py", "arrhenius_fracture/marked_topology_trial_v13.py",
            "arrhenius_fracture/accepted_step_overlay_v13.py",
            "scripts/qualify_v13_branch_marks.py")},
        "parameter_status": "ILLUSTRATIVE_BRANCH_ONLY_SENSITIVITY_NOT_CALIBRATED_NOT_MATERIAL_PROMOTION",
        "physical_trajectories": 0, "new_mechanics_solves": 0, "process_intervals_in_MC": 0,
        "canonical_parent_overlay_parity": parity, "frozen_states": states, "monte_carlo": rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(safe_json(output), indent=2, sort_keys=True, allow_nan=False) + "\n"
    (args.output_root / "branch_mark_qualification.json").write_text(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
    for ax, case in zip(axes.flat, cases):
        for q, ov in ((0, 0), (.1, .1), (.3, .1)):
            selected = [r for r in rows if r["case"] == case and r["junction_barrier_eV"] == q]
            ax.plot([r["exposure_time_s"] for r in selected], [r["analytic_probability"] for r in selected],
                label=f"Qj={q}, Qov={ov} eV")
            ax.scatter([r["exposure_time_s"] for r in selected], [r["sample_probability"] for r in selected], s=20)
        ax.set(xscale="log", ylim=(-.03, 1.03), title=case, xlabel="Subgrid exposure (s)", ylabel="Conditional mark probability")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Frozen mark validation, not branch predictions\nBRANCHING_KINETICS_MODEL_UNCALIBRATED")
    fig.tight_layout()
    fig.savefig(args.output_root / "frozen_mark_validation.png", dpi=150)
    plt.close(fig)
    print("PASS", len(rows), "configurations;", len(rows)*args.samples, "mark-only draws")


if __name__ == "__main__":
    main()
