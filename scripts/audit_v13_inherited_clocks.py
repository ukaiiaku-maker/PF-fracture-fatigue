#!/usr/bin/env python3
"""Frozen source-clock and inverse-scale audit. No FEM solve or process step."""
import argparse
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import subprocess

from scipy.special import gammainc, gammaincinv
from arrhenius_fracture.production_step_loop_v11 import _logmean_rate
from arrhenius_fracture import conditional_branch_mark_v13 as marks
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine
from scripts.v13_frozen_support import initialized_engine
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
from scripts.run_pf_current_source_multifront_field_atlas_v12 import CASES, atomic_json, sha256

NULL_MODEL = "FRESH_INDEPENDENT_POST_PRIMARY_ORDER3_COMPANION_WITH_TAU_B_LE_TAU_C"


def balance(a, b):
    if min(a, b) < 0 or not all(math.isfinite(x) for x in (a, b)):
        raise ValueError("invalid rate")
    if max(a, b) == 0:
        return None
    x, y = a/max(a,b), b/max(a,b)
    return 4*x*y/(x+y)**2


def inverse_scales(raw, tau_c, temperature, order=3):
    if min(raw, tau_c, temperature) <= 0:
        raise ValueError("inverse requirements need positive finite rates/scales")
    rows = []
    for probability in (.01, .05, .5):
        x = float(gammaincinv(order, probability))
        factor = x/(raw*tau_c)
        rows.append({"target_probability": probability, "order": order,
            "required_lambda_tau": x, "required_tau_B_s": x/raw,
            "required_beta_B": factor,
            "barrier_reduction_eV": 8.617333262145e-5*temperature*math.log(factor),
            "raw_arrival_multiplicity_factor": factor,
            "independent_complete_site_multiplicity": math.log1p(-probability)/math.log1p(-float(gammainc(order,raw*tau_c))),
            "scope": "inverse_requirements_not_fitted_parameters"})
    return rows


def audit_case(case, root):
    folder = root/"physical_parents"/case/"parent"
    record = json.loads((folder/"parent_record.json").read_text())
    payload_path = folder/"event_context.pkl"
    assert sha256(payload_path) == record["event_context_sha256"]
    payload = pickle.loads(payload_path.read_bytes())
    before = payload["pre_cleavage_checkpoint"]
    after = payload["accepted_single_checkpoint"]
    context = payload["context"]
    source_fingerprint = fp((before.state,after.state))
    companion = json.loads((root/"companions"/case/"companion_qualification.json").read_text())["companions"][0]
    assert companion["post_primary_process_sha256"] == fp(after.shared_process_state)
    rates = {r.candidate_id: r for r in payload["canonical_result"].rates}
    final_hazards = {h.candidate_id: h for h in after.state.competition.hazard_states}
    T = float(case.rsplit("_",1)[1][:-1])
    rows = []
    with initialized_engine(before.shared_process_state) as engine:
        engine_before = fp(_capture_shared_engine(engine))
        for hazard in before.state.competition.hazard_states:
            candidate = next(c for c in before.state.competition.candidates if c.candidate_id == hazard.candidate_id)
            rate = rates[candidate.candidate_id]
            preview = copy.deepcopy(engine)
            effective, raw, barrier = preview.lambda_cleave(preview.sigma_tip(rate.K_directional_Pa_sqrt_m/math.sqrt(candidate.gamma_rel)), T)
            assert math.isclose(effective, rate.lambda_per_s, rel_tol=1e-12, abs_tol=1e-300)
            mean = _logmean_rate(hazard.previous_rate_per_s, effective)
            H_cross = hazard.action + mean*(record["raw_first_passage_s"]-context.physical_time_s)
            H_endpoint = hazard.action + mean*context.duration_s
            assert math.isclose(H_endpoint,final_hazards[candidate.candidate_id].action, rel_tol=1e-13)
            winner = candidate.candidate_id == record["winning_candidate_id"]
            eta = hazard.current_threshold_action
            if winner:
                assert math.isclose(H_cross, eta, rel_tol=1e-11, abs_tol=1e-12)
            residual = max(0., eta-H_cross) if not winner else 0.
            post = companion["effective_parent_law_rate_diagnostic_per_s"] if not winner else None
            rows.append({"candidate_id": candidate.candidate_id, "winner": winner,
                "raw_rate_pre_per_s": raw, "effective_rate_pre_per_s": effective,
                "raw_barrier_pre_J": barrier, "interval_logmean_rate_per_s": mean,
                "H_last_accepted": hazard.action, "H_at_raw_primary_crossing_left_limit": H_cross,
                "H_pre_topology_at_accepted_endpoint": H_endpoint,
                "eta_fixed_for_parent_event": eta, "residual_eta_minus_H_at_crossing": residual,
                "normalized_progress_at_crossing": H_cross/eta,
                "normalized_progress_at_endpoint": H_endpoint/eta,
                "residual_at_endpoint": max(0., eta-H_endpoint),
                "event_ordinal": hazard.completed_event_count+1,
                "threshold_rng_identity": {"scheme":"v11-directional-threshold", "seed":hazard.threshold_seed,
                    "candidate_id":candidate.candidate_id, "ordinal":hazard.completed_event_count+1},
                "clock_snapshot_sha256": fp(hazard),
                "process_rng_sha256": fp(before.shared_process_state["engine_fields"]["_hazard_rng"]),
                "predicted_remaining_pre_s": residual/effective if effective else None,
                "post_primary_effective_rate_per_s": post,
                "post_primary_raw_rate_per_s": companion["raw_arrival_per_s"] if not winner else None,
                "predicted_remaining_post_s_from_raw_crossing": residual/post if post else (0. if winner else None),
                "predicted_remaining_post_s_from_endpoint": max(0.,eta-H_endpoint)/post if post else (0. if winner else None),
                "post_primary_scope": "saved_actual_single_to_pair_companion_drive" if not winner else "primary_already_consumed_no_second_event_rate_invented",
                "exact_pair_energy_margin_J_per_m": companion["pair_energy_margin_J_per_m"]})
        assert fp(_capture_shared_engine(engine)) == engine_before
        tau = float(engine.f.tau_c)
    assert fp((before.state,after.state)) == source_fingerprint and sha256(payload_path) == record["event_context_sha256"]
    nonwinner = next(row for row in rows if not row["winner"])
    # Fixed diagnostic definition, not a probability fit or branching switch.
    near = nonwinner["normalized_progress_at_crossing"] >= .9 or nonwinner["predicted_remaining_post_s_from_raw_crossing"] <= tau
    return {"case": case, "parent_record_sha256":sha256(folder/"parent_record.json"),
        "event_context_sha256":record["event_context_sha256"], "raw_primary_crossing_s":record["raw_first_passage_s"],
        "accepted_endpoint_s":record["accepted_endpoint_s"], "candidates":rows,
        "raw_rate_balance_pre":balance(*(r["raw_rate_pre_per_s"] for r in rows)),
        "effective_rate_balance_pre":balance(*(r["effective_rate_pre_per_s"] for r in rows)),
        "tau_c_s":tau, "near_completion_diagnostic":near,
        "inverse_requirements":inverse_scales(companion["raw_arrival_per_s"],tau,T),
        "order_sensitivity":[{"order":order,"probability_at_tau_c_zero_extra_barrier":float(gammainc(order,companion["raw_arrival_per_s"]*tau)),
            "inverse_requirements":inverse_scales(companion["raw_arrival_per_s"],tau,T,order)} for order in (1,2,3)],
        "canonical_quadrature_reproduced":True, "source_unchanged":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    cases=[audit_case(case,args.source_root) for case in CASES]
    text=inspect.getsource(marks.draw_branch_mark)
    assert "gammaincinv(channel.multihit_order, u)" in text
    assert not any(s in inspect.signature(marks.CompanionChannel).parameters for s in ("action","threshold","eta","H"))
    result={"schema":"v13.inherited-clock-audit/1", "null_model":NULL_MODEL,
        "boundary":marks.BOUNDARY, "source_record_commit":"d10c3ebd8a3513edec5624faebdc7faf8656d106",
        "producer_code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "cases":cases, "existing_mark_inherits_baseline_action":False,
        "fresh_gamma_source_evidence":text,
        "near_completion_definition":"H/eta >= 0.9 at raw crossing OR frozen post-primary residual time <= tau_c; diagnostic only",
        "mechanism_decision":"INHERITED_RESIDUAL_CLOCK" if any(c["near_completion_diagnostic"] for c in cases) else "SEPARATE_COOPERATIVE_PAIR_TRANSITION_REQUIRED_FOR_APPRECIABLE_BRANCHING",
        "time_interpretation":"H at raw crossing is reconstructed exactly under canonical constant logmean interval quadrature, not a separately archived continuous physical state. Pre rates use actual frozen pre-update process and solved endpoint mechanics. Residual completion times freeze post-primary mechanics/process; no global-time advance.",
        "fem_solves":0,"process_steps":0,"new_parent_runs":0,"new_rng_draws":0}
    atomic_json(args.output_root/"clock_audit.json",result)
    for c in cases:
        r=next(r for r in c["candidates"] if not r["winner"])
        print(c["case"],"rho",r["normalized_progress_at_crossing"],"post residual s",r["predicted_remaining_post_s_from_raw_crossing"])
    print(result["mechanism_decision"])


if __name__=="__main__":
    main()
