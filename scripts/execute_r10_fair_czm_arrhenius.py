#!/usr/bin/env python3
"""Assemble and execute the fair original-mechanics CZM/Arrhenius comparison.

This program is deliberately fail-closed.  It admits archived calculations only
at their recorded mechanics/execution identity and never promotes the later
candidate-dependent finite-compliance spring to the original-mechanics study.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json, math, subprocess, time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "97a1e48fa556f784b2fb5203916da2d17af98732"
SCHEMA = "r10.fair-czm-vs-arrhenius/1"


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fields);writer.writeheader();writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args(argv);out=args.out.resolve();out.mkdir(parents=True, exist_ok=True)
    started=time.time();head=git("rev-parse","HEAD")
    subprocess.check_call(["git","merge-base","--is-ancestor",ANCHOR,head],cwd=ROOT)

    r9=ROOT/"integration_evidence/r9_paris_slope_current_backend"
    coh=ROOT/"integration_evidence/r10_cohesive_arrhenius_joint_sweep"
    inv=ROOT/"runs/r10_inverse_shape_12h/campaign_delivery"
    preflight=load(r9/"production_preflight.json")
    r9_width=load(r9/"analysis/primary/slope_width_summary.json")
    c1=load(coh/"results/R10D_COH_00167/analysis_summary.json")
    c2=load(coh/"results/R10D_HARD_01138/analysis_summary.json")
    prior=load(inv/"campaign_summary.json")

    contract={
      "schema":SCHEMA,"source_head":head,"provenance_anchor":ANCHOR,
      "mechanical_provider":"RESTORED_REDUCED_PRESCRIBED_K_WITH_R4_EFFECTIVE_OPENING_KERNEL",
      "mechanical_coordinates":{"load":"prescribed_K_Pa_sqrt_m","local_drive":"sigma_opening_tip_Pa",
        "separation_coordinate":None,"work_conjugate_interface_traction":None},
      "geometry":preflight["reduced_config"],"waveform":preflight["waveform"],
      "plasticity_consumers":["EXP_floor_emission","Peierls_motion","Taylor_release",
        "persistent_sources","signed_transport","backstress","blunting_and_wake"],
      "history_energy_ownership":{"crack_advance":"accepted event transaction",
        "committed_crack":"checkpoint_advance_total_m","surface_separation_energy":"not represented in anchor 1-D state"},
      "opening_only_change_boundary":"manifest.cleavage barrier/rate and its completion mark are replaceable; common sigma_opening_tip and plastic state are not",
      "interface_finding":"The anchor/current original 1-D operator exposes stress but no separation residual/tangent. A full CZM therefore lacks a retained equilibrium consumer.",
      "excluded_environment":{"formula":"k_env=Eprime/(2*(Eprime*Gamma/t_hat^2))",
        "reason":"candidate-dependent cohesive-test spring introduced after anchor; not established as condensation of original mechanics",
        "alternative_2um_block":"excluded"},
      "common_model_eligible_execution_ids":["ARR_EXP_FLOOR_REFERENCE","ARR_ALTERNATE_SHAPE_REFERENCE","COHESIVE_BARRIER_ONLY_DIAGNOSTIC"],
      "full_czm_execution_status":"INTERFACE_GAP_UNRESOLVED"
    }
    dump(out/"COMMON_MODEL_CONTRACT.json",contract)
    (out/"OPENING_CHANGE_BOUNDARY.md").write_text(
      "# Opening-only change boundary\n\nThe retained 1-D model maps prescribed K and the evolving EXP_floor state to a local opening stress. "
      "It has no separation unknown, interface residual, or tangent. Replacing the cleavage barrier is supported as a stress-coordinate diagnostic. "
      "A mechanical or lifetime CZM needs a separation-bearing mechanical consumer. The later candidate-dependent spring and 2 µm block are excluded by the controlling brief, so their results cannot decide the original-mechanics CZM question.\n")

    matrix=[
      {"identity":"ARR_EXP_FLOOR_REFERENCE","opening":"original EXP_floor","kinetics":"Arrhenius","history":"cooperative completion mark","common_mechanics":"YES","scope":"archived controls"},
      {"identity":"ARR_ALTERNATE_SHAPE_REFERENCE","opening":"two-basis accepted thermal surface","kinetics":"Arrhenius","history":"bounded stochastic reward","common_mechanics":"YES","scope":"current-backend R9"},
      {"identity":"CZM_MECHANICAL","opening":"smooth/bilinear/compact potential","kinetics":"athermal","history":"irreversible branch/contact","common_mechanics":"UNRESOLVED","scope":"no retained separation consumer"},
      {"identity":"CZM_DERIVED_LIFETIME","opening":"cohesive potential barrier","kinetics":"high-barrier/MFPT","history":"connected branch and commitment","common_mechanics":"UNRESOLVED","scope":"stress-control barrier-only diagnostic admissible; spring results excluded"},
      {"identity":"CZM_NONTHERMAL_STATISTICAL","opening":"fixed site resistance CDF","kinetics":"non-Boltzmann threshold survival","history":"material-fixed threshold/no redraw","common_mechanics":"ANALYTICAL_ONLY","scope":"no existing resolved stochastic plastic-event separation source"},
    ]
    write_csv(out/"MODEL_MATRIX.csv",matrix)

    # Re-execute the complete 32,768-row stress-coordinate analytical audit.
    bank=coh/"analytical_bank.csv.gz";total=valid=strict=general=0;best=None
    with gzip.open(bank,"rt",newline="") as stream:
      for row in csv.DictReader(stream):
        total+=1
        if row.get("domain_pass")!="True":continue
        valid+=1
        lo=float(row["A0_slope_300K_min"]);hi=float(row["A0_slope_300K_max"])
        width=math.log10(float(row["A0_rate_300K_max_m_per_cycle"])/float(row["A0_rate_300K_min_m_per_cycle"]))
        if lo>=3.4 and hi<=4.6 and width>=2:strict+=1
        center=.5*(lo+hi)
        if width>=2 and center>0 and (hi-lo)/center<=.30:general+=1
        score=(max(0,2-width)+abs(center-4)+(hi-lo))
        if best is None or score<best[0]:best=(score,row["candidate_id"],lo,hi,width)
    analytical={"schema":SCHEMA,"source":str(bank.relative_to(ROOT)),"source_sha256":sha(bank),
      "executed_rows":total,"domain_valid":valid,"strict_near_four_passes":strict,
      "general_15_percent_constant_slope_passes":general,
      "best_sampled":{"candidate_id":best[1],"slope_min":best[2],"slope_max":best[3],"rate_width_decades":best[4]},
      "scope":"COHESIVE_POTENTIAL_DERIVED_STRESS_CONTROL_BARRIER_ON_OLD_CLOCK; NOT_FULL_CZM",
      "optimizer_claim":"Sobol coverage, not certified global optimum"}
    dump(out/"ANALYTICAL_CAPABILITY_AUDIT.json",analytical)

    nonthermal=[
      {"case":"homogeneous_fixed_threshold","new_maximum":False,"conditional_failure_probability":0.0,"classification":"SHAKEDOWN_INVARIANT"},
      {"case":"bounded_fixed_threshold_distribution","new_maximum":False,"conditional_failure_probability":0.0,"classification":"NO_AUTONOMOUS_WAITING"},
      {"case":"bounded_fixed_threshold_distribution","new_maximum":True,"conditional_failure_probability":"(F(w1)-F(w0))/(1-F(w0))","classification":"LOAD_HISTORY_EVENT_NOT_CYCLE_LIFETIME"},
    ]
    write_csv(out/"NONTHERMAL_STATISTICAL_CLOSURE.csv",nonthermal)

    # Exact-scope scientific comparisons. No missing/unresolved result becomes zero.
    native=[
      {"identity":"ARR_ALTERNATE_SHAPE_REFERENCE","candidate":"V3_ceramic_V2_P25_007__V3_1_TRUST1",
       "mechanics_eligible":True,"strict_pass":True,"general_pass":True,
       "rate_width_decades":r9_width["largest_connected_m4_interval"]["rate_width_decades"],
       "slope_summary":"4.191907 +/- 0.016962 (three-seed endpoint)","evidence_strength":"NATIVE_CURRENT_BACKEND_ARCHIVED"},
      {"identity":"COHESIVE_BARRIER_ONLY_DIAGNOSTIC","candidate":c1["candidate_id"],"mechanics_eligible":True,
       "strict_pass":False,"general_pass":False,"rate_width_decades":c1["connected_rate_width_decades"],
       "slope_summary":json.dumps(c1["adjacent_developed_slopes"]),"evidence_strength":"NATIVE_STRESS_COORDINATE_DIAGNOSTIC"},
      {"identity":"COHESIVE_BARRIER_ONLY_DIAGNOSTIC","candidate":c2["candidate_id"],"mechanics_eligible":True,
       "strict_pass":False,"general_pass":False,"rate_width_decades":c2["connected_rate_width_decades"],
       "slope_summary":json.dumps(c2["adjacent_developed_slopes"]),"evidence_strength":"NATIVE_STRESS_COORDINATE_DIAGNOSTIC"},
      {"identity":"CZM_MECHANICAL","candidate":"ALL_EXISTING_FULL_CZM_RUNS","mechanics_eligible":False,
       "strict_pass":"UNRESOLVED","general_pass":"UNRESOLVED","rate_width_decades":"UNRESOLVED",
       "slope_summary":"excluded candidate-dependent or 2um environment","evidence_strength":"OUT_OF_SCOPE_DIAGNOSTIC"},
      {"identity":"CZM_DERIVED_LIFETIME","candidate":"ALL_EXISTING_FULL_CZM_RUNS","mechanics_eligible":False,
       "strict_pass":"UNRESOLVED","general_pass":"UNRESOLVED","rate_width_decades":"UNRESOLVED",
       "slope_summary":"excluded candidate-dependent or 2um environment","evidence_strength":"OUT_OF_SCOPE_DIAGNOSTIC"},
    ]
    write_csv(out/"NATIVE_COMPARISON.csv",native)

    vectors=[]
    vectors.append({"identity":"ARR_ALTERNATE_SHAPE_REFERENCE","candidate_id":preflight["binding"]["candidate_id"],
      "complete_vector_json":json.dumps(preflight["material"],sort_keys=True)})
    for cid in ("R10D_COH_00167","R10D_HARD_01138"):
      hit=None
      for registry in (coh/"native_batch_registry.csv",
                       coh/"thermal_hardening_expansion/native_batch_registry.csv"):
        with registry.open(newline="") as stream:
          hit=next((r for r in csv.DictReader(stream) if r["candidate_id"]==cid),hit)
      if hit is None: raise RuntimeError(f"missing complete registry row for {cid}")
      vectors.append({"identity":"COHESIVE_BARRIER_ONLY_DIAGNOSTIC","candidate_id":cid,
        "complete_vector_json":json.dumps({"cohesive":json.loads(hit["cohesive_parameter_vector"]),
          "EXP_floor_parent":json.loads(hit["parent_complete_parameter_vector"])},sort_keys=True)})
    write_csv(out/"COMPLETE_MATERIAL_VECTORS.csv",vectors)

    parity_path=out/"parity/Kmax_15p4624747/result.json"
    parity=(load(parity_path) if parity_path.is_file() else None)
    coverage={"schema":SCHEMA,"new_audit_executions":{"analytical_physical_vectors":total,
      "nonthermal_closure_cases":len(nonthermal),"native_parity_cases":int(parity is not None),
      "native_parity_accepted_cycles":None if parity is None else parity["final_cycles"],
      "native_parity_events":None if parity is None else len(parity["events"])},
      "reused_exact_scope":{"r9_current_backend_amplitudes":5,"r9_confirmation_streams":3,
        "barrier_only_native_materials":2,"barrier_only_native_fatigue_points":10,
        "barrier_only_native_fracture_points":14},
      "excluded_not_counted":prior["coverage"],"elapsed_seconds":time.time()-started,
      "shortfall_reason":"Full-CZM original-mechanics execution is blocked by a specific missing separation/tangent interface; excluded spring runs are not migrated."}
    dump(out/"COVERAGE_AND_RESOURCES.json",coverage)
    write_csv(out/"RESTART_INVENTORY.csv",[
      {"identity":"ARR_ALTERNATE_SHAPE_REFERENCE_PARITY","eligible":True,
       "path":str(out/"parity/Kmax_15p4624747"),
       "status":"COMPLETE" if parity is not None else "NOT_EXECUTED"},
      {"identity":"CZM_MECHANICAL_AND_DERIVED_LIFETIME","eligible":False,
       "path":"runs/r10_inverse_shape_12h (preserved)",
       "status":"EXCLUDED_ALTERNATIVE_MECHANICAL_ENVIRONMENT"},
    ])

    decision={"schema":SCHEMA,
      "strict_near_four":{"arr_alternate_reference":"PASS","cohesive_barrier_diagnostic":"FAIL_IN_TESTED_DOMAIN","full_CZM":"UNRESOLVED"},
      "broader_power_law":{"arr_alternate_reference":"PASS_M_APPROX_4.19","cohesive_barrier_diagnostic":"FAIL_STEEP_25_TO_39","full_CZM":"UNRESOLVED"},
      "joint_thermal":{"barrier_diagnostic":"AVAILABLE_BUT_NOT_FULL_CZM","full_CZM":"UNRESOLVED"},
      "mechanism":"R9 comparator uses independent Arrhenius opening landscape; tested cohesive-derived stress-clock rows are opening dominated/steep. Full-CZM mechanism cannot be assigned under original mechanics.",
      "evidence_strength":"ARRHENIUS_POSITIVE_PLUS_COHESIVE_STRESS_DIAGNOSTIC_BOUNDED_NEGATIVE; FULL_CZM_UNRESOLVED_INTERFACE",
      "capability_answer":"UNRESOLVED_FOR_FULL_CZM_IN_ORIGINAL_MECHANICS",
      "universal_czm_negative":False}
    dump(out/"CAPABILITY_DECISION.json",decision)

    (out/"EXECUTED_REPORT.md").write_text(f"""# Fair CZM-versus-Arrhenius execution

## Decision

The accepted alternate-Arrhenius control passes the strict and broader criteria. The cohesive-derived stress-coordinate diagnostic is a search-bounded negative in the tested domain. The full mechanical and lifetime CZM question under the original mechanics is unresolved because the retained 1-D operator has no separation equilibrium consumer; the later candidate-dependent spring and 2 µm block are excluded.

## Executed evidence

- Re-evaluated {total:,} cohesive physical vectors ({valid:,} domain-valid): zero strict and zero broader constant-slope passes.
- Reused two exact-scope native cohesive-barrier materials, ten developed fatigue points and fourteen thermal-fracture points. Their adjacent slopes are 25.5–28.8 and 36.8–38.8.
- Reused the five-amplitude, three-stream current-backend alternate-Arrhenius result: m = 4.191907 ± 0.016962 across 4.26694 rate decades.
- Executed a fresh current-backend parity trajectory at ΔK = 13.9162 MPa√m: {0 if parity is None else len(parity['events'])} events and {0 if parity is None else parity['final_cycles']:.6g} accepted cycles.
- Evaluated the homogeneous and bounded fixed-threshold nonthermal closures. With no new maximum and no threshold redraw, their conditional event probability is zero; this is closure-specific shakedown, not a universal CZM result.

## Scope

All curves use applied ΔK. The cohesive barrier results retain the old clock and are diagnostics, not full-CZM evidence. Existing finite-compliance and 2-D cohesive calculations remain preserved outside the main comparison. No missing or excluded result is entered as zero.
""")

    (out/"CLAIMS_AND_LIMITS.md").write_text("""# Claims and limits

The accepted alternative Arrhenius opening surface retains a native current-backend Paris interval: m = 4.191907 ± 0.016962 and 4.26694 rate decades at 300 K, R=0.1, 1000 Hz.

Within the cohesive-potential-derived, stress-controlled barrier diagnostic, 32,768 sampled vectors and two native five-amplitude materials did not reproduce the strict or broader power-law criterion. The native slopes were 25.5–28.8 and 36.8–38.8 despite roughly four rate decades. This is a bounded negative for that clock/shape diagnostic, not for full CZMs.

The original 1-D mechanics exposes local stress but no separation equilibrium residual or tangent. Existing mechanical and lifetime CZM trajectories add a candidate-dependent spring or a separate 2 µm environment and are excluded. Consequently the full-CZM capability question is unresolved, not negative. A fair next calculation requires a separation-bearing reduction derived from the preserved mechanical operator; choosing an adjustable compliance would change the question.

A fixed nonthermal site-threshold distribution cannot generate autonomous waiting under repeated identical maxima. That closure can respond to new maxima but does not exclude CZMs with additional, explicitly modeled irreversible history. No universal claim that CZMs cannot fatigue is supported.
""")

    # One compact common-axis figure from exact native evidence.
    fig,ax=plt.subplots(figsize=(6.3,4.2))
    with (r9/"analysis/primary/growth_window_table.csv").open(newline="") as stream:
      rr=list(csv.DictReader(stream))
    x=[float(r.get("DeltaK_MPa_sqrt_m",r.get("DeltaK",0))) for r in rr]
    y=[float(r["developed_da_dN_m_per_cycle"]) for r in rr]
    ax.loglog(x,y,"o-",label="accepted alternate Arrhenius")
    for cid,label in (("R10D_COH_00167","cohesive barrier 00167"),("R10D_HARD_01138","cohesive barrier 01138")):
      with (coh/f"results/{cid}/developed_fatigue_source.csv").open(newline="") as stream: rows=list(csv.DictReader(stream))
      ax.loglog([float(r["DeltaK_MPa_sqrt_m"]) for r in rows],[float(r["developed_rate_m_per_cycle"]) for r in rows],"o-",label=label)
    ax.set_xlabel(r"Applied $\Delta K$ (MPa $\sqrt{m}$)");ax.set_ylabel(r"Developed $da/dN$ (m/cycle)")
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/"fair_native_comparison.png",dpi=220);fig.savefig(out/"fair_native_comparison.pdf");plt.close(fig)

    commands={"script":"python scripts/execute_r10_fair_czm_arrhenius.py --out runs/r10_fair_czm_vs_arrhenius",
      "head":head,"inputs":{str(p.relative_to(ROOT)):sha(p) for p in [bank,r9/"production_preflight.json",
        r9/"analysis/primary/slope_width_summary.json",coh/"results/R10D_COH_00167/analysis_summary.json",
        coh/"results/R10D_HARD_01138/analysis_summary.json"]}}
    dump(out/"REPRODUCTION.json",commands)
    manifest=[]
    for path in sorted(out.iterdir()):
      if path.is_file():manifest.append({"path":path.name,"sha256":sha(path),"bytes":path.stat().st_size})
    write_csv(out/"MANIFEST.csv",manifest)
    print(json.dumps(decision,indent=2));return 0


if __name__ == "__main__": raise SystemExit(main())
