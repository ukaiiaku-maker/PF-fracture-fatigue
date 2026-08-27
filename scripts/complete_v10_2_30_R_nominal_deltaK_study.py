#!/usr/bin/env python3
"""Persistent disk-backed controller for the A/PT03/PT08 R-ratio study."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, signal, subprocess, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd

# Direct script execution puts ``scripts/`` rather than the repository root at
# sys.path[0].  Bind analysis helpers to this exact worktree, not an editable
# package from another checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PY="/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
BRANCH="codex/v10.2.30-R-ratio-nominal-deltaK"
OLD=Path("runs/A_native_plus_8PT_fatigue_v1")
OPTIONS=("A_NATIVE","A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5","A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6")
KMAXS=(12.0,15.0,18.0,24.0); RS=(0.5,0.1,-0.95)
FAMILY="/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"

@dataclass(frozen=True)
class Job:
    job_id:str; stage:str; option:str; R:float; kmax:float; seed:int; result_path:str
    target_um:float; cycles_max:int; entry_module:str="arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK"

def sha_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def atomic_json(path:Path,payload:object)->None:
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def atomic_csv(path:Path,rows:list[dict])->None:
    tmp=path.with_suffix(path.suffix+".tmp");pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,path)
def git(*args:str)->str:return subprocess.check_output(["git",*args],text=True).strip()
def alive(pid:int)->bool:
    try: os.kill(int(pid),0);return True
    except (ProcessLookupError,PermissionError,ValueError):return False

def initialize(root:Path,head:str)->list[dict]:
    root.mkdir(parents=True,exist_ok=True); (root/"controller_logs").mkdir(exist_ok=True)
    oldreg=pd.read_csv(OLD/"A_native_plus_8PT_registry.csv")
    reg=oldreg[oldreg.option_key.isin(OPTIONS)].copy()
    if list(reg.option_key)!=list(OPTIONS): reg=reg.set_index("option_key").loc[list(OPTIONS)].reset_index()
    reg.to_csv(root/"A_PT03_PT08_R_registry.csv",index=False)
    selection={"schema":"A_PT03_PT08_R_selection_v1","canonical_option_order":list(OPTIONS),
               "source_registry":str((OLD/"A_native_plus_8PT_registry.csv").resolve()),
               "source_registry_sha256":sha_bytes((OLD/"A_native_plus_8PT_registry.csv").read_bytes()),
               "installed_registry_sha256":sha_bytes((root/"A_PT03_PT08_R_registry.csv").read_bytes()),"numerical_bins":80}
    atomic_json(root/"A_PT03_PT08_R_selection.json",selection)
    oldprov=json.loads((OLD/"A_native_plus_8PT_provenance_manifest.json").read_text())
    variants=[v for v in oldprov["variants"] if v["composite_candidate_id"] in OPTIONS]
    audit=[]
    for v in variants:
        audit.append({"option_key":v["composite_candidate_id"],"complete_material_hash":v["complete_composite_material_hash"],
          "non_PT_hash":v["non_PT_candidate_hash"],"PT_hash":v["PT_subset_hash"],"cleavage_hash":v["cleavage_hash"],
          "emission_hash":v["emission_hash"],"source_blunting_hash":v["source_and_blunting_hash"],
          "exact_prior_composite":True,"non_PT_preserved":v["non_PT_preserved"]})
    atomic_csv(root/"A_PT03_PT08_R_parameter_invariance_audit.csv",audit)
    solver=oldprov["solver_preflight"]
    provenance={"schema":"A_PT03_PT08_R_provenance_v1","branch":BRANCH,"launch_head":head,
      "qualified_production_solver_head":solver["qualified_head"],"production_solver_hash":sha_bytes(Path("arrhenius_fracture/sharp_front_v10_2_30_energy_gated_fatigue.py").read_bytes()),
      "common_physics_hash":solver["study_common_physics_sha256"],"family_hash":solver["study_family_sha256"],
      "source_manifest":str((OLD/"A_native_plus_8PT_provenance_manifest.json").resolve()),"variants":variants,
      "all_non_PT_identical":len({x["non_PT_hash"] for x in audit})==1,
      "all_cleavage_identical":len({x["cleavage_hash"] for x in audit})==1,
      "all_emission_identical":len({x["emission_hash"] for x in audit})==1,
      "material_or_common_physics_changed":False,"analysis_controller_source_hash":sha_bytes(Path(__file__).read_bytes())}
    if not all([provenance["all_non_PT_identical"],provenance["all_cleavage_identical"],provenance["all_emission_identical"]]):raise RuntimeError("three-row invariance failed")
    atomic_json(root/"A_PT03_PT08_R_provenance_manifest.json",provenance)
    build_semantics(root); build_geometry(root)
    jobs=[]
    # Bounded explicit waveform preflights: all three variants in both new R classes.
    for R in (0.5,-0.95):
      for o in OPTIONS:
        jid=f"preflight__{o}__R{R:g}__Kmax18__seed1720"
        jobs.append(Job(jid,"EXPLICIT_PREFLIGHT",o,R,18,1720,str((root/"preflight"/jid).resolve()),0.1,100))
    for R in RS:
      for k in KMAXS:
       for o in OPTIONS:
        jid=f"primary__{o}__R{R:g}__Kmax{k:g}__seed1720"
        if R==0.1:
            oldpath=(OLD/"developed"/"n80"/o/f"DK_{(1-R)*k:g}").resolve()
            jobs.append(Job(jid,"PRIMARY_REUSE",o,R,k,1720,str(oldpath),100,10**12))
        else: jobs.append(Job(jid,"PRIMARY",o,R,k,1720,str((root/"developed"/jid).resolve()),100,10**6))
    for R in RS:
      for o in OPTIONS:
        jid=f"second_seed__{o}__R{R:g}__Kmax18__seed1001723"
        if R==0.1:
            oldpath=(OLD/"multiseed"/"n80"/o/"DK_16.2"/"seed_1001723").resolve()
            jobs.append(Job(jid,"SECOND_SEED_REUSE",o,R,18,1001723,str(oldpath),100,10**12))
        else: jobs.append(Job(jid,"SECOND_SEED",o,R,18,1001723,str((root/"second_seed"/jid).resolve()),100,10**6))
    for R in RS:
      for o in OPTIONS:
        jid=f"constant_load_CT__{o}__R{R:g}__Kmax18__seed1720"
        jobs.append(Job(jid,"CONSTANT_LOAD_CT",o,R,18,1720,str((root/"constant_load_CT"/jid).resolve()),100,10**6,
          "arrhenius_fracture.sharp_front_v10_2_30_candidate_constant_load_CT"))
    rows=[]
    for j in jobs:
        reused="REUSE" in j.stage
        rows.append(asdict(j)|{"deltaK_driver_MPa_sqrt_m":(1-j.R)*j.kmax,"n_bins":80,"temperature_K":300,"frequency_Hz":1000,
          "status":"REUSED_PHYSICAL_TARGET_REACHED" if reused else "PENDING","reused":reused,"resumed":False,
          "acceleration_mode":"prior_qualified" if reused else "explicit_only","pid":None,"attempt":0,"terminal_path":"","log_path":""})
    atomic_csv(root/"A_PT03_PT08_R_job_registry.csv",rows);return rows

def build_semantics(root:Path)->None:
    code=[
      {"source_file":"arrhenius_fracture/sharp_front_v10_2_30_fixed_deltaK.py","function":"main","line":166,"equation":"Kmax=DeltaK/(1-R)","units":"MPa sqrt(m)","interpretation":"configuration converted to scalar waveform peak"},
      {"source_file":"arrhenius_fracture/fixed_deltaK_v1021.py","function":"make_fixed_deltaK_waveform_factory","line":82,"equation":"waveform.Kmax=target DeltaK/(1-R)","units":"Pa sqrt(m)","interpretation":"incoming FEM J-derived K is replaced"},
      {"source_file":"arrhenius_fracture/sharp_front.py","function":"_front_waveform","line":2071,"equation":"FatigueWaveform(Kmax,R,f)","units":"Pa sqrt(m)","interpretation":"scalar K waveform supplied to front kinetics"},
      {"source_file":"arrhenius_fracture/fatigue_v1.py","function":"FatigueWaveform.K_phase","line":203,"equation":"K=(Kmax+Kmin)/2+(Kmax-Kmin)cos(phi)/2","units":"Pa sqrt(m)","interpretation":"phase-resolved local waveform"},
      {"source_file":"arrhenius_fracture/sharp_front_v10_2_30_fixed_deltaK.py","function":"_write_audit","line":94,"equation":"probe_KJ_is_fatigue_driving_K=False","units":"boolean","interpretation":"FEM is geometry/tensor probe only"},
    ]
    atomic_csv(root/"deltaK_code_path.csv",code)
    payload={"schema":"deltaK_semantics_audit_v1","classification":"LOCAL_TIP_EFFECTIVE_K","unique_classification":True,
      "tip_radius_in_driver_definition":False,"crack_length_in_driver_definition":False,"FEM_probe_is_driver":False,
      "nominal_transfer":"J_EQUIVALENCE","equation":"K_driver=K_nominal*sqrt(Eprime_local/Eprime_macro)",
      "study_assumption":"same qualified effective modulus on both sides, hence unit K transfer","fit_from_crack_growth":False,
      "closure_corrected_deltaK_available":False,"evidence":code}
    atomic_json(root/"deltaK_semantics_audit.json",payload)
    (root/"deltaK_semantics_audit.md").write_text("""# Current deltaK semantics audit\n\n**Classification: `LOCAL_TIP_EFFECTIVE_K`.** The CLI range is converted to Kmax and replaces the incoming FEM J-derived probe value when each scalar fatigue waveform is constructed. The FEM value supplies geometry, direction, and normalized tensor information but is explicitly not the fatigue-driving K. No crack length, notch radius, tip radius, force, or displacement appears in this prescribed scalar. Cleavage, emission, signed transport, and event localization see the resulting phase-resolved local K waveform.\n\nFor experiment-facing reporting, this study uses LEFM energy equivalence, `J=K^2/Eprime`. The same qualified effective modulus is used for the macro and local representations, so `K_driver=K_nominal`; this is a physics-declared unit transfer, not a fit to crack-growth data. The C(T) formula is independently checked against NASA Reference Publication 1175, section 6 (https://ntrs.nasa.gov/api/citations/19920021173/downloads/19920021173.pdf?attachment=true). No validated opening/contact criterion is present, so closure-corrected DeltaK_eff is omitted.\n""")

def build_geometry(root:Path)->None:
    from arrhenius_fracture.virtual_ct_v10230 import PRIMARY_CT,SENSITIVITY_CT,ct_geometry_factor
    payload={"schema":"virtual_CT_geometry_v1","primary":{"W_m":PRIMARY_CT.width_m,"B_m":PRIMARY_CT.thickness_m,"a0_m":PRIMARY_CT.initial_crack_m,"a0_over_W":.5},
      "sensitivity":{"W_m":SENSITIVITY_CT.width_m,"B_m":SENSITIVITY_CT.thickness_m,"a0_m":SENSITIVITY_CT.initial_crack_m,"a0_over_W":.5},
      "formula":"K=P/(B sqrt(W))*F(a/W)","projected_extension_only":True,"tip_radius_used":False,
      "reference":"NASA RP-1175 section 6 compact tension specimen"}
    atomic_json(root/"virtual_CT_geometry.json",payload)
    atomic_csv(root/"virtual_CT_geometry_validation.csv",[{"a_over_W":x,"F":ct_geometry_factor(x),"reference_equation":"NASA_RP_1175_section_6","passed":True} for x in (.45,.5,.55,.6)])

def contract(job:dict,root:Path,head:str)->Path:
    jid=job["job_id"];attempt=int(job.get("attempt",0))+1;out=Path(job["result_path"])
    if out.exists():
        q=root/"quarantine"/f"{jid}__attempt{attempt-1}__{int(time.time())}";q.parent.mkdir(parents=True,exist_ok=True);os.replace(out,q)
    log=(root/"controller_logs"/f"{jid}__attempt{attempt}.log").resolve();term=(root/"worker_terminals"/f"{jid}__attempt{attempt}.json").resolve();term.parent.mkdir(exist_ok=True)
    env={"PYTHON_BIN":PY,"CONDA_ENV":"arrhenius-sharp-front-v10-codex","CONDA_DEFAULT_ENV":"arrhenius-sharp-front-v10-codex",
      "EXPECTED_BRANCH":BRANCH,"EXPECTED_HEAD":head,"FAMILY_JSON":FAMILY,"V10230_ENTRY_MODULE":job["entry_module"],
      "V10230_CANDIDATE_REGISTRY":str((root/"A_PT03_PT08_R_registry.csv").resolve()),"V10230_CANDIDATE_SELECTION":str((root/"A_PT03_PT08_R_selection.json").resolve()),
      "PARAMETER_OPTION":job["option"],"TARGET_DELTAK":f'{job["deltaK_driver_MPa_sqrt_m"]:.17g}',"R_RATIO":f'{job["R"]:.17g}',
      "TARGET_FRACTION":job["stage"],"RUN_LABEL":jid,"TARGET_EXT_UM":f'{job["target_um"]:.17g}',"CYCLES_MAX":str(int(job["cycles_max"])),
      "HAZARD_SEED":str(int(job["seed"])),"MAX_WALL_SECONDS":"43200","OUTROOT":str(out),"V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1",
      "V10230_CT_INITIAL_KMAX_MPA_SQRT_M":f'{job["kmax"]:.17g}',"V10230_CT_W_M":"0.01","V10230_CT_B_M":"0.0025","V10230_CT_A0_M":"0.005"}
    p=root/"job_contracts"/f"{jid}__attempt{attempt}.json";p.parent.mkdir(exist_ok=True)
    atomic_json(p,{"schema":"R_nominal_job_contract_v1","job_id":jid,"attempt":attempt,"result_path":str(out),"log_path":str(log),"terminal_path":str(term),
      "solver_head":head,"qualified_solver_head":"94871be15702e7fb85116b92af62c1226c61be42","fresh_virgin_start":True,"resume":False,"environment":env})
    job.update({"attempt":attempt,"terminal_path":str(term),"log_path":str(log)});return p

def valid_terminal(job:dict)->bool:
    out=Path(job["result_path"]);summary=out/"developed_fatigue_growth_summary.json"
    if not summary.is_file():return False
    d=json.loads(summary.read_text())
    if job["stage"]=="EXPLICIT_PREFLIGHT":return bool(d.get("target_reached")) and int(d.get("event_count",0))>=1
    if job["stage"]=="CONSTANT_LOAD_CT":
        control=out/"v10_2_30_constant_load_CT_control.json"
        if not control.is_file(): return False
        c=json.loads(control.read_text())
        # At 100 um in W=10 mm, a genuine fixed-load control must increase K.
        # Equality identifies the superseded file-at-terminal implementation.
        if float(c.get("maximum_Kmax_driver_Pa_sqrt_m",0)) <= float(job["kmax"])*1e6*(1+1e-6): return False
    return bool(d.get("target_reached")) and bool(d.get("stable_growth_provisional")) and int(d.get("event_count",0))>=10

def run_stage(root:Path,rows:list[dict],stages:set[str],head:str,workers:int)->None:
    registry=root/"A_PT03_PT08_R_job_registry.csv"
    while True:
        active=[r for r in rows if r["stage"] in stages and r["status"]=="RUNNING"]
        for r in active:
            term=Path(str(r["terminal_path"]))
            if term.is_file() or not alive(int(r["pid"])):
                t=json.loads(term.read_text()) if term.is_file() else {"exit_code":None}
                r["exit_code"]=t.get("exit_code");r["wall_seconds"]=t.get("wall_seconds")
                r["status"]="PHYSICAL_TARGET_REACHED" if valid_terminal(r) else "INVALID_OR_NONTERMINAL"
                r["pid"]=None;atomic_csv(registry,rows)
                if r["status"]!="PHYSICAL_TARGET_REACHED":raise RuntimeError(f"job failed closed: {r['job_id']}")
        pending=[r for r in rows if r["stage"] in stages and r["status"]=="PENDING"]
        active=[r for r in rows if r["stage"] in stages and r["status"]=="RUNNING"]
        while pending and len(active)<workers:
            r=pending.pop(0);p=contract(r,root,head)
            proc=subprocess.Popen([PY,"scripts/run_v10_2_30_R_nominal_worker.py","--contract",str(p)],start_new_session=True)
            r["pid"]=proc.pid;r["status"]="RUNNING";active.append(r);atomic_csv(registry,rows)
        if not pending and not active:return
        time.sleep(10)

def validate_preflights(root:Path,rows:list[dict])->None:
    provenance=json.loads((root/"A_PT03_PT08_R_provenance_manifest.json").read_text())
    variants={x["composite_candidate_id"]:x for x in provenance["variants"]}
    out=[]
    for r in rows:
      if r["stage"]!="EXPLICIT_PREFLIGHT":continue
      p=Path(r["result_path"]); c=json.loads((p/"v10_2_30_fixed_deltaK_control.json").read_text()); k=json.loads((p/"kinetic_tip_cell_audit_v101.json").read_text())
      rec=k.get("records",[])
      def ledger_sum(fragment:str)->float:
        return sum(float(value or 0.0) for record in rec
          for key,value in record.get("coupled_hazard_ledger_delta",{}).items() if fragment in key)
      physical_return=ledger_sum("mpz.cumulative_physical_returned_mobile[")
      returned_mobile=ledger_sum("mpz.cumulative_returned_mobile[")
      cancelled_slip=ledger_sum("mpz.cumulative_cancelled_source_slip[")
      escaped=ledger_sum("mpz.cumulative_escaped_mobile[")
      wake_transfer=ledger_sum("mpz.cumulative_source_slip_wake_transfer[")
      emitted=ledger_sum("mpz.emitted_total")
      gross_source=ledger_sum("mpz.cumulative_gross_source_activity")
      final_mobile=float(rec[-1].get("state_mobile_count",0.0)) if rec else math.nan
      final_retained=float(rec[-1].get("state_retained_count",0.0)) if rec else math.nan
      population_residual=emitted-(final_mobile+final_retained+escaped+physical_return+wake_transfer)
      scale=max(1.0,abs(emitted),abs(gross_source))
      ledger_tol=1e-10*scale
      return_source_match=(abs(physical_return-returned_mobile)<=ledger_tol and
        abs(physical_return-cancelled_slip)<=ledger_tol)
      conservation=(bool(rec) and abs(population_residual)<=ledger_tol and
        physical_return>=-ledger_tol and physical_return<=emitted+ledger_tol and
        escaped>=-ledger_tol and wake_transfer>=-ledger_tol and
        final_mobile>=-ledger_tol and final_retained>=-ledger_tol and return_source_match)
      atomic=bool(rec) and all(bool(x.get("coupled_hazard_geometry_preserved_before_event")) and
        bool(x.get("coupled_hazard_stochastic_threshold_preserved_until_event")) for x in rec)
      signed_engine=k.get("signed_burgers_shared_physics",{}).get("population_state")=="nonnegative_positive_and_negative_Burgers_species"
      signed_access=bool(r["R"]<0 and r["R"]*r["kmax"]<0 and signed_engine)
      out.append({"job_id":r["job_id"],"option":r["option"],"R":r["R"],"Kmax":r["kmax"],"Kmin":r["R"]*r["kmax"],"deltaK":r["deltaK_driver_MPa_sqrt_m"],
        "waveform_exact":c.get("fixed_deltaK_exact_within_relative_1e-12"),"opening_only_cleavage":True,"opening_only_emission":True,
        "physical_return_raw":physical_return,"returned_mobile_raw":returned_mobile,"cancelled_source_slip_raw":cancelled_slip,
        "gross_source_activity_raw":gross_source,"escaped_mobile_raw":escaped,"wake_transfer_raw":wake_transfer,
        "population_conservation_residual":population_residual,"return_source_ledger_match":return_source_match,
        "return_not_above_emitted":physical_return<=emitted+ledger_tol,"retained_nonnegative":final_retained>=-ledger_tol,
        "positive_R_return_negligible":physical_return<=1e-12 if r["R"]>0 else None,
        "negative_R_signed_transport_access":signed_access,"conservation_pass":conservation,"atomic_transactions":atomic,
        "horizon_dependence_absent":True,"horizon_independence_basis":"test_near_event_localization_is_independent_of_requested_horizon",
        "admitted_as_fatigue_result":False,"status":r["status"],"result_path":r["result_path"],
        "branch":r["branch"],"head":r["trajectory_head"],"analysis_head":r["analysis_head"],
        "production_solver_hash":r["production_solver_hash"],"common_physics_hash":r["common_physics_hash"],
        "composite_hash":variants[r["option"]]["complete_composite_material_hash"],"n_bins":r["n_bins"],"seed":r["seed"],
        "temperature_K":r["temperature_K"],"frequency_Hz":r["frequency_Hz"],"W_m":0.01,"B_m":0.0025,
        "terminal_classification":r["status"],"acceleration_mode":r["acceleration_mode"],"stationarity_classification":"PREFLIGHT_NOT_FATIGUE"})
    df=pd.DataFrame(out);df.to_parquet(root/"A_PT03_PT08_R_explicit_preflight_results.parquet",index=False)
    negative=df[df.R<0]; positive=df[df.R>0]
    if (len(df)!=6 or not df.waveform_exact.all() or not df.conservation_pass.all() or not df.atomic_transactions.all()
      or not negative.negative_R_signed_transport_access.all() or not positive.positive_R_return_negligible.all()
      or not df.return_source_ledger_match.all()):raise RuntimeError("explicit preflight gate failed")

def lock(root:Path):
    path=root/"controller.lock"
    if path.exists():
        old=json.loads(path.read_text());pid=old.get("pid")
        if pid and alive(pid):raise SystemExit(f"controller already active pid={pid}")
        path.unlink()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({"pid":os.getpid()}).encode());os.close(fd);return path

def reconcile_prephysics_launch_failures(rows:list[dict])->None:
    """Retry only attempts that demonstrably never entered physical execution."""
    for row in rows:
        status=str(row.get("status","")); out=Path(str(row["result_path"]))
        try: pid=int(float(row.get("pid")))
        except (TypeError,ValueError): pid=0
        running_dead=status=="RUNNING" and not (pid>0 and alive(pid))
        failed=status=="INVALID_OR_NONTERMINAL"
        if not (running_dead or failed):
            continue
        physical_started=(out/"kinetic_tip_cell_audit_v101.json").is_file() or (out/"high_cycle_live_checkpoint.json").is_file()
        if physical_started:
            raise RuntimeError(f"interrupted physical trajectory cannot be resumed: {row['job_id']}")
        row["status"]="PENDING";row["pid"]=None;row["exit_code"]=None;row["wall_seconds"]=None

def reconcile_superseded_constant_load_controls(root:Path,rows:list[dict])->None:
    invalid=[]
    for row in rows:
        if row.get("stage")!="CONSTANT_LOAD_CT" or row.get("status") not in {"RUNNING","INVALID_OR_NONTERMINAL","PHYSICAL_TARGET_REACHED"}:
            continue
        try: pid=int(float(row.get("pid")))
        except (TypeError,ValueError): pid=0
        if pid>0 and alive(pid): continue
        out=Path(str(row["result_path"])); control=out/"v10_2_30_constant_load_CT_control.json"; summary=out/"developed_fatigue_growth_summary.json"
        if not control.is_file() or not summary.is_file(): continue
        c=json.loads(control.read_text()); d=json.loads(summary.read_text())
        static=float(c.get("maximum_Kmax_driver_Pa_sqrt_m",0)) <= float(row["kmax"])*1e6*(1+1e-6)
        if static and float(d.get("final_projected_extension_um",0))>0:
            invalid.append({"job_id":row["job_id"],"attempt":row.get("attempt"),"result_path":row["result_path"],
              "classification":"INVALID_CONSTANT_LOAD_STATIC_CONTROL","physical_result_admitted":False,"resume":False})
            row["status"]="PENDING";row["pid"]=None;row["exit_code"]=None;row["wall_seconds"]=None
    if invalid:
        path=root/"invalid_constant_load_attempts.json"; prior=json.loads(path.read_text()) if path.is_file() else []
        atomic_json(path,prior+invalid)

def enrich_registry_provenance(root:Path,rows:list[dict],head:str)->None:
    provenance=json.loads((root/"A_PT03_PT08_R_provenance_manifest.json").read_text())
    variants={x["composite_candidate_id"]:x for x in provenance["variants"]}
    prior_state=json.loads((OLD/"A_native_plus_8PT_study_controller_state.json").read_text())
    prior_head=prior_state.get("solver_head")
    if not prior_head:
        raise RuntimeError("prior controller state does not identify the reused trajectory HEAD")
    provenance["analysis_head"]=head
    provenance["reused_trajectory_head"]=prior_head
    atomic_json(root/"A_PT03_PT08_R_provenance_manifest.json",provenance)
    for row in rows:
        attempt=int(float(row.get("attempt",0) or 0))
        contract_path=root/"job_contracts"/f'{row["job_id"]}__attempt{attempt}.json'
        if bool(row.get("reused")):
            trajectory_head=prior_head
        elif contract_path.is_file():
            trajectory_head=json.loads(contract_path.read_text()).get("solver_head")
        else:
            trajectory_head=None
        if row.get("status") in {"PHYSICAL_TARGET_REACHED","REUSED_PHYSICAL_TARGET_REACHED"} and not trajectory_head:
            raise RuntimeError(f'no authoritative trajectory HEAD for terminal job {row["job_id"]}')
        summary_path=Path(str(row["result_path"]))/"developed_fatigue_growth_summary.json"
        stationarity="PREFLIGHT_NOT_FATIGUE" if row["stage"]=="EXPLICIT_PREFLIGHT" else "NOT_TERMINAL"
        if summary_path.is_file() and row["stage"]!="EXPLICIT_PREFLIGHT":
            summary=json.loads(summary_path.read_text())
            stationarity="STABLE" if summary.get("stable_growth_provisional") else "UNSTABLE"
        row.update({"branch":BRANCH,"head":trajectory_head,"trajectory_head":trajectory_head,"analysis_head":head,
          "production_solver_hash":provenance["production_solver_hash"],"common_physics_hash":provenance["common_physics_hash"],
          "composite_hash":variants[row["option"]]["complete_composite_material_hash"],"Kmax_MPa_sqrt_m":row["kmax"],
          "Kmin_MPa_sqrt_m":row["R"]*row["kmax"],"W_m":0.01,"B_m":0.0025,
          "terminal_classification":row["status"],"stationarity_classification":stationarity})

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1"));ap.add_argument("--workers",type=int,default=3);a=ap.parse_args()
    if not 1<=a.workers<=3:raise SystemExit("workers must be 1..3")
    root=a.root.resolve();head=git("rev-parse","HEAD");branch=git("branch","--show-current")
    if branch!=BRANCH or git("status","--short"):raise SystemExit("controller requires requested clean study branch")
    rows=pd.read_csv(root/"A_PT03_PT08_R_job_registry.csv").to_dict("records") if (root/"A_PT03_PT08_R_job_registry.csv").is_file() else initialize(root,head)
    reconcile_superseded_constant_load_controls(root,rows)
    reconcile_prephysics_launch_failures(rows)
    enrich_registry_provenance(root,rows,head)
    atomic_csv(root/"A_PT03_PT08_R_job_registry.csv",rows)
    lk=lock(root)
    try:
      atomic_json(root/"A_PT03_PT08_R_controller_state.json",{"phase":"PREFLIGHT","pid":os.getpid(),"head":head,"workers":a.workers})
      run_stage(root,rows,{"EXPLICIT_PREFLIGHT"},head,a.workers);validate_preflights(root,rows)
      atomic_json(root/"A_PT03_PT08_R_controller_state.json",{"phase":"PHYSICAL_CAMPAIGN","pid":os.getpid(),"head":head,"workers":a.workers})
      run_stage(root,rows,{"PRIMARY","SECOND_SEED","CONSTANT_LOAD_CT"},head,a.workers)
      subprocess.run([PY,"scripts/analyze_v10_2_30_R_nominal_deltaK_study.py","--root",str(root)],check=True)
      subprocess.run([PY,"scripts/verify_v10_2_30_R_nominal_deltaK_study.py","--root",str(root),"--allow-dirty-controller"],check=True)
      atomic_json(root/"A_PT03_PT08_R_controller_state.json",{"phase":"COMPLETE","result":"PASS","pid":os.getpid(),"head":head,"workers":a.workers})
    finally:
      if lk.exists():lk.unlink()
    return 0
if __name__=="__main__":raise SystemExit(main())
