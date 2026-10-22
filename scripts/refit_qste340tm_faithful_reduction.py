#!/usr/bin/env python3
"""Joint QSTE340TM refit using the native-order retained-state reduction.

The four measured constant-amplitude histories are replayed in source order.
Plastic state is advanced independently of opening, so each plastic trajectory
is cached while the one-scale smooth-broad opening law is fitted conditionally.
No population cap or empirical R multiplier is present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_stateful_exp_floor import (
    ATTEMPT_FREQUENCY_S, COOPERATIVE_HITS, COOPERATIVE_TAU_S,
    BURGERS_M, C_BLUNT, EVENT_LENGTH_M, NOMINAL_FREQUENCY_HZ, PHASE_STEPS,
    PROCESS_ZONE_M, RHO0_M2, SHEAR_MODULUS_PA, SMOOTHING_RATIO,
    TEMPERATURE_K, TIP_RADIUS_M, build_records,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.qste340tm_faithful_reduction import (
    PlasticParameters, apply_cycles, cycle_operator, embrittlement_eV,
    one_cycle_with_transfers, stress_cycle,
)

DATA = ROOT / "literature_data/fatigue_recovered_review"
CONTROL = DATA / "model_conditional/stateful_exp_floor_capacity_policy/continuous_birth_exact/fit_uncapped_continuous_birth.json"
OUT = DATA / "model_conditional/stateful_exp_floor_capacity_policy/faithful_retained_reduction"
PARAMETER_NAMES = (
    "opening_floor_over_kBT", "opening_amplitude_over_kBT",
    "log_opening_sigma_b_GPa", "log_opening_exponent",
    "cleavage_backstress_fraction", "emission_G0_298_eV",
    "log_plastic_sigc_298_GPa", "transport_G0_298_eV",
    "log_retained_recovery_per_s", "log1p_initial_mobile_count",
    "log1p_initial_retained_count",
)
LOWER = np.array([8., 0., math.log(.05), math.log(.15), 0., .30,
                  math.log(.10), .30, math.log(1e-8), 0., 0.])
UPPER = np.array([30., 120., math.log(100.), math.log(8.), 2., 4.0,
                  math.log(30.), 5.0, math.log(1e4), math.log1p(5000.),
                  math.log1p(5000.)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(vector: np.ndarray) -> dict[str, float]:
    return {
        "opening_floor_over_kBT": float(vector[0]),
        "opening_amplitude_over_kBT": float(vector[1]),
        "opening_sigma_b_Pa": float(math.exp(vector[2])*1e9),
        "opening_exponent": float(math.exp(vector[3])),
        "cleavage_backstress_fraction": float(vector[4]),
        "emission_G0_298_eV": float(vector[5]),
        "plastic_sigc_298_Pa": float(math.exp(vector[6])*1e9),
        "transport_G0_298_eV": float(vector[7]),
        "retained_recovery_per_s": float(math.exp(vector[8])),
        "initial_mobile_count": float(math.expm1(vector[9])),
        "initial_retained_count": float(math.expm1(vector[10])),
        "initial_source_activity": 1.0,
        "plastic_EXP_floor_n": 1.0,
    }


def pack(parameters: dict[str, float], *, mobile: float = 0.) -> np.ndarray:
    return np.array([
        parameters["opening_floor_over_kBT"], parameters["opening_amplitude_over_kBT"],
        math.log(parameters["opening_sigma_b_Pa"]/1e9), math.log(parameters["opening_exponent"]),
        parameters["cleavage_backstress_fraction"], parameters["emission_G0_298_eV"],
        math.log(parameters["plastic_sigc_298_Pa"]/1e9), parameters["escape_G0_298_eV"],
        math.log(parameters["recovery_per_s"]), math.log1p(mobile),
        math.log1p(parameters["initial_retained_count"]),
    ])


def plastic_parameters(vector: np.ndarray) -> PlasticParameters:
    p = unpack(vector)
    return PlasticParameters(
        p["emission_G0_298_eV"], p["plastic_sigc_298_Pa"],
        p["transport_G0_298_eV"], p["retained_recovery_per_s"],
        p["initial_mobile_count"], p["initial_retained_count"],
        p["initial_source_activity"], p["plastic_EXP_floor_n"],
    )


def plastic_trajectory(vector: np.ndarray, records: list[dict], *,
                       interval_substeps: int = 2,
                       waveform_steps: int = PHASE_STEPS,
                       evolve: bool = True) -> dict[str, np.ndarray]:
    pp = plastic_parameters(vector)
    n = len(records)
    keys = ("mobile", "retained", "source_activity", "density_m2", "radius",
            "backstress", "emission", "trapped", "released", "recovered",
            "escaped", "inventory_balance_residual", "mean_peierls_rate_s",
            "mean_release_rate_s", "mean_encounter_rate_s", "mean_escape_rate_s")
    out = {key: np.zeros(n) for key in keys}
    out["stress"] = np.zeros((n, waveform_steps))
    groups: dict[float, list[int]] = {}
    for i, row in enumerate(records):
        groups.setdefault(float(row["R"]), []).append(i)
    for r_value, indices in groups.items():
        state = np.array([pp.initial_mobile_count, pp.initial_retained_count,
                          pp.initial_source_activity], dtype=float)
        prior_cycles = float(records[indices[0]]["cycles"])
        prior_dk = float(records[indices[0]]["DeltaK_MPa_sqrt_m"])
        for position, i in enumerate(indices):
            row = records[i]
            cycles = float(row["cycles"])
            dk = float(row["DeltaK_MPa_sqrt_m"])
            delta_cycles = max(cycles-prior_cycles, 0.) if position else 0.
            if evolve and delta_cycles > 0.:
                for substep in range(max(interval_substeps, 1)):
                    fraction = (substep+.5)/max(interval_substeps, 1)
                    local_dk = prior_dk+fraction*(dk-prior_dk)
                    operator, _ = cycle_operator(local_dk, r_value, state, pp, waveform_steps)
                    state = apply_cycles(operator, state, delta_cycles/max(interval_substeps, 1))
            stress, radius, backstress = stress_cycle(dk, r_value, state[0], state[1], waveform_steps)
            _, transfers = one_cycle_with_transfers(dk, r_value, state, pp, waveform_steps)
            out["stress"][i] = stress
            out["mobile"][i], out["retained"][i], out["source_activity"][i] = state
            out["density_m2"][i] = 5e12+(state[0]+state[1])/1e-12
            out["radius"][i] = radius
            out["backstress"][i] = backstress
            for key in ("emitted", "trapped", "released", "recovered", "escaped",
                        "inventory_balance_residual", "mean_peierls_rate_s",
                        "mean_release_rate_s", "mean_encounter_rate_s", "mean_escape_rate_s"):
                out["emission" if key == "emitted" else key][i] = transfers[key]
            prior_cycles, prior_dk = cycles, dk
    return out


def opening_prediction(vector: np.ndarray, trajectory: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(vector)
    basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO)
    opening_stress = np.maximum(trajectory["stress"]-p["cleavage_backstress_fraction"]*trajectory["backstress"][:,None], 0.)
    raw = p["opening_floor_over_kBT"]+p["opening_amplitude_over_kBT"]*basis.value(opening_stress)
    embrittlement = np.array([embrittlement_eV(x) for x in trajectory["retained"]])
    raw = raw-embrittlement[:,None]/(KB_EV_PER_K*TEMPERATURE_K)
    elementary = ATTEMPT_FREQUENCY_S*np.exp(-np.clip(raw, 0., 700.))
    completed = gammainc(COOPERATIVE_HITS, np.minimum(elementary*COOPERATIVE_TAU_S, 1e12))/COOPERATIVE_TAU_S
    prediction = EVENT_LENGTH_M*np.mean(completed, axis=1)/NOMINAL_FREQUENCY_HZ
    return np.maximum(prediction, 1e-300), raw, opening_stress


def metrics(records: list[dict], prediction: np.ndarray, trajectory: dict[str, np.ndarray]) -> dict:
    obs = np.array([float(row["observed_dadn_m_per_cycle"]) for row in records])
    rv = np.array([float(row["R"]) for row in records])
    dk = np.array([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    error = np.log10(prediction/obs)
    blocks = {}
    for r_value in sorted(set(rv)):
        mask = rv == r_value
        order = np.argsort(dk[mask])
        logk=np.log10(dk[mask][order]);keep=np.r_[True,np.diff(logk)>1e-12]
        logk=logk[keep];logp=np.log10(prediction[mask][order])[keep];logo=np.log10(obs[mask][order])[keep]
        local=np.diff(logp)/np.diff(logk);observed_local=np.diff(logo)/np.diff(logk)
        blocks[str(r_value)] = {
            "points": int(mask.sum()),
            "rmse_log10_rate": float(np.sqrt(np.mean(error[mask]**2))),
            "bias_log10_rate": float(np.mean(error[mask])),
            "observed_broad_loglog_slope": float(np.polyfit(np.log10(dk[mask]), np.log10(obs[mask]), 1)[0]),
            "predicted_broad_loglog_slope": float(np.polyfit(np.log10(dk[mask]), np.log10(prediction[mask]), 1)[0]),
            "observed_local_slope_q10_q50_q90": np.quantile(observed_local,[.1,.5,.9]).tolist(),
            "predicted_local_slope_q10_q50_q90": np.quantile(local,[.1,.5,.9]).tolist(),
            "mobile_final": float(trajectory["mobile"][mask][-1]),
            "retained_final": float(trajectory["retained"][mask][-1]),
        }
    return {
        "overall_rmse_log10_rate": float(np.sqrt(np.mean(error**2))),
        "overall_bias_log10_rate": float(np.mean(error)),
        "block_metrics": blocks,
        "max_mobile_count": float(np.max(trajectory["mobile"])),
        "max_retained_count": float(np.max(trajectory["retained"])),
        "max_density_m2": float(np.max(trajectory["density_m2"])),
        "max_backstress_Pa": float(np.max(trajectory["backstress"])),
        "max_tip_radius_m": float(np.max(trajectory["radius"])),
    }


class JointResidual:
    def __init__(self, records: list[dict], substeps: int, waveform_steps: int):
        self.records = records
        rv = np.array([float(row["R"]) for row in records])
        self.weights = np.empty(len(records))
        for r_value in sorted(set(rv)):
            self.weights[rv == r_value] = 1/math.sqrt(np.sum(rv == r_value))
        self.observed = np.log10([float(row["observed_dadn_m_per_cycle"]) for row in records])
        self.substeps = substeps
        self.waveform_steps = waveform_steps
        self._plastic_key = None
        self._trajectory = None

    def trajectory(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        key = np.asarray(vector[5:], float).tobytes()
        if key != self._plastic_key:
            self._trajectory = plastic_trajectory(vector, self.records,
                interval_substeps=self.substeps, waveform_steps=self.waveform_steps)
            self._plastic_key = key
        return self._trajectory

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        trajectory = self.trajectory(vector)
        prediction, raw, _ = opening_prediction(vector, trajectory)
        error = (np.log10(prediction)-self.observed)*self.weights
        validity = max(0., .25-float(np.min(raw)))*10.
        return np.r_[error, validity]


def starting_vectors(records: list[dict]) -> list[tuple[str,np.ndarray]]:
    prior = json.loads(CONTROL.read_text())["parameters"]
    base = pack(prior)
    vectors = [("5703_control_vector", base)]
    historical=json.loads((DATA/"model_conditional/stateful_exp_floor/fit.json").read_text())
    for label,item in (("63be_main",historical["fit"]),("63be_excluding_R0p3",historical["leave_one_R_out"][1])):
        old=item["parameters"];mapped={"opening_floor_over_kBT":old["opening_floor_over_kBT"],"opening_amplitude_over_kBT":old["opening_amplitude_over_kBT"],"opening_sigma_b_Pa":old["opening_sigma_b_Pa"],"opening_exponent":old["opening_exponent"],"cleavage_backstress_fraction":old["cleavage_backstress_fraction"],"emission_G0_298_eV":old["emission_G0_298_eV"],"plastic_sigc_298_Pa":old["plastic_sigc_298_Pa"],"escape_G0_298_eV":old["escape_G0_298_eV"],"recovery_per_s":old["recovery_per_s"],"initial_retained_count":0.}
        vectors.append((label,pack(mapped)))
    # Plastic variations span active/weak source and transport ownership.
    for label, emit_g, transport_g, recovery, mobile, retained in (
        ("balanced_mobile",1.2,1.0,1e-2,20.,100.),
        ("slow_transport",1.2,2.0,1e-3,5.,100.),
        ("active_transfer",.8,.8,1e-1,20.,20.),
        ("high_barrier",2.0,1.5,1e-4,0.,400.),
    ):
        x=base.copy();x[5]=emit_g;x[7]=transport_g;x[8]=math.log(recovery)
        x[9]=math.log1p(mobile);x[10]=math.log1p(retained)
        vectors.append((label,x))
    # Active opening knees are explicitly placed in the consumed stress domain.
    for i, sigma in enumerate((.6,1.0,2.0,4.0,8.0)):
        x=vectors[i%len(vectors)][1].copy();x[0]=17.;x[1]=25.;x[2]=math.log(sigma);x[3]=math.log(1.)
        vectors.append((f"active_opening_{sigma:g}GPa",x))
    x=base.copy();x[1]=0.;x[0]=17.;vectors.append(("stress_independent_opening",x))
    return vectors


def fit(records: list[dict], *, max_nfev: int, substeps: int, waveform_steps: int) -> dict:
    rv=np.array([float(row["R"]) for row in records]);observed=np.log10([float(row["observed_dadn_m_per_cycle"]) for row in records]);weights=np.empty(len(records))
    for r_value in sorted(set(rv)):weights[rv==r_value]=1/math.sqrt(np.sum(rv==r_value))
    base=starting_vectors(records)[0][1].copy()
    def fit_opening(trajectory: dict[str,np.ndarray], seed: np.ndarray | None=None, nfev: int=24):
        stress=trajectory["stress"]
        positive=stress[stress>0.]
        quantiles=np.quantile(positive,[.2,.55,.85]) if len(positive) else np.ones(3)*1e9
        starts=[]
        if seed is not None:starts.append(np.asarray(seed[:5],float))
        for sigma in quantiles:
            starts.append(np.array([17.,25.,math.log(max(sigma/1e9,.05)),0.,.2]))
        starts.append(np.array([17.,0.,math.log(max(np.median(quantiles)/1e9,.05)),0.,0.]))
        lo=LOWER[:5];hi=UPPER[:5]
        def opening_residual(op):
            vector=base.copy();vector[:5]=op
            prediction,raw,_=opening_prediction(vector,trajectory)
            raw_error=np.log10(prediction)-observed
            centered=np.empty_like(raw_error)
            for r_value in sorted(set(rv)):
                mask=rv==r_value;centered[mask]=raw_error[mask]-np.mean(raw_error[mask])
            # Preserve rate calibration while making measured within-R
            # slope/curvature an explicit part of selection.  Otherwise a
            # single mean plateau wins the absolute least-squares compromise.
            error=np.r_[.5*raw_error*weights,centered*weights]
            validity=max(0.,.25-float(np.min(raw)))*10.
            return np.r_[error,validity]
        candidates=[]
        for initial in starts:
            try:
                sol=least_squares(opening_residual,np.clip(initial,lo,hi),bounds=(lo,hi),max_nfev=nfev,x_scale="jac",ftol=1e-9,xtol=1e-9,gtol=1e-9)
                candidates.append((float(np.dot(opening_residual(sol.x),opening_residual(sol.x))),sol))
            except (FloatingPointError,ValueError,np.linalg.LinAlgError):pass
        if not candidates:raise FloatingPointError("no finite conditional opening fit")
        return min(candidates,key=lambda item:item[0])
    profile_cache={}
    def profile(plastic: np.ndarray, seed: np.ndarray | None=None, detailed: bool=False):
        key=np.asarray(plastic,float).tobytes()
        if key in profile_cache and not detailed:return profile_cache[key][0]
        vector=base.copy();vector[5:]=plastic
        try:
            trajectory=plastic_trajectory(vector,records,interval_substeps=substeps,waveform_steps=waveform_steps)
            objective,opening_solution=fit_opening(trajectory,seed)
            # This is a validity penalty, not a population cap: states are never clipped.
            max_density=float(np.max(trajectory["density_m2"]))
            density_penalty=max(0.,math.log10(max_density)-18.)**2*10.
            objective+=density_penalty
            result=(objective,opening_solution,trajectory)
        except (FloatingPointError,ValueError,np.linalg.LinAlgError,OverflowError):
            result=(1e12,None,None)
        profile_cache[key]=result
        return result if detailed else result[0]
    scored=[]
    for name,vector in starting_vectors(records):
        objective,opening_solution,trajectory=profile(vector[5:],vector[:5],True)
        if opening_solution is not None:vector=vector.copy();vector[:5]=opening_solution.x
        scored.append((float(objective),name,vector))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"start_scores.json").write_text(json.dumps([{"start":name,"weighted_sse":score,"parameters":unpack(np.clip(x,LOWER,UPPER))} for score,name,x in sorted(scored)],indent=2,sort_keys=True)+"\n")
    runs=[]
    for objective,name,vector in sorted(scored)[:6]:
        trajectory=plastic_trajectory(vector,records,interval_substeps=substeps,waveform_steps=waveform_steps);prediction,raw,opening_stress=opening_prediction(vector,trajectory);result_metrics=metrics(records,prediction,trajectory)
        run={"start":name,"parameters_vector":vector.tolist(),"parameters":unpack(vector),"profile_weighted_sse":objective,"metrics":result_metrics,"optimizer":{"kind":"conditional_opening_profile","success":True},"opening_stress_range_Pa":[float(np.min(opening_stress)),float(np.max(opening_stress))],"raw_barrier_over_kBT_range":[float(np.min(raw)),float(np.max(raw))]}
        runs.append((result_metrics["overall_rmse_log10_rate"],vector,run));(OUT/f"checkpoint_{name}.json").write_text(json.dumps(run,indent=2,sort_keys=True)+"\n")
    print("[faithful] bounded profiled plastic acquisition",flush=True);started=time.time()
    with ThreadPool(4) as pool:
        global_solution=differential_evolution(profile,list(zip(LOWER[5:],UPPER[5:])),seed=20261021,popsize=4,maxiter=max(1,max_nfev//12),polish=False,workers=pool.map,updating="deferred",tol=2e-4)
    objective,opening_solution,trajectory=profile(global_solution.x,None,True)
    vector=base.copy();vector[5:]=global_solution.x;vector[:5]=opening_solution.x
    prediction,raw,opening_stress=opening_prediction(vector,trajectory);result_metrics=metrics(records,prediction,trajectory)
    global_run={"start":"bounded_profiled_global","parameters_vector":vector.tolist(),"parameters":unpack(vector),"profile_weighted_sse":objective,"metrics":result_metrics,"optimizer":{"kind":"differential_evolution_with_conditional_opening","success":bool(global_solution.success),"message":str(global_solution.message),"nfev":int(global_solution.nfev),"elapsed_s":time.time()-started},"opening_stress_range_Pa":[float(np.min(opening_stress)),float(np.max(opening_stress))],"raw_barrier_over_kBT_range":[float(np.min(raw)),float(np.max(raw))]}
    runs.append((result_metrics["overall_rmse_log10_rate"],vector,global_run));(OUT/"checkpoint_bounded_profiled_global.json").write_text(json.dumps(global_run,indent=2,sort_keys=True)+"\n")
    _,selected,_=min(runs,key=lambda item:item[0])
    # Refit opening at the declared resolution without altering plastic state.
    trajectory=plastic_trajectory(selected,records,interval_substeps=substeps,waveform_steps=waveform_steps)
    _,opening_solution=fit_opening(trajectory,selected[:5],max_nfev);vector=selected.copy();vector[:5]=opening_solution.x
    prediction,raw,opening_stress=opening_prediction(vector,trajectory)
    frozen=plastic_trajectory(vector,records,interval_substeps=substeps,waveform_steps=waveform_steps,evolve=False)
    frozen_prediction,_,_=opening_prediction(vector,frozen)
    singular=np.linalg.svd(opening_solution.jac,compute_uv=False)
    class ProfileSolution:pass
    solution=ProfileSolution();solution.x=vector;solution.success=bool(opening_solution.success);solution.status=int(opening_solution.status);solution.message="profiled plastic acquisition; "+str(opening_solution.message);solution.nfev=int(global_solution.nfev+opening_solution.nfev);solution.cost=float(opening_solution.cost);solution.optimality=float(opening_solution.optimality)
    return {"solution":solution,"trajectory":trajectory,"prediction":prediction,
            "frozen":frozen,"frozen_prediction":frozen_prediction,"raw":raw,
            "opening_stress":opening_stress,"runs":runs,"jacobian_singular_values":singular}


def write_outputs(records: list[dict], fitted: dict, substeps: int, waveform_steps: int) -> dict:
    solution=fitted["solution"];trajectory=fitted["trajectory"];prediction=fitted["prediction"]
    frozen_prediction=fitted["frozen_prediction"]
    observed=np.log10([float(row["observed_dadn_m_per_cycle"]) for row in records])
    independent_vector=solution.x.copy();independent_vector[1]=0.
    independent_solution=least_squares(lambda z:np.log10(opening_prediction(np.r_[z[0],independent_vector[1:]],trajectory)[0])-observed,
        np.array([independent_vector[0]]),bounds=(LOWER[:1],UPPER[:1]),max_nfev=80)
    independent_vector[0]=independent_solution.x[0]
    independent_prediction,_,_=opening_prediction(independent_vector,trajectory)
    convergence={"waveform":{},"history":{}}
    for steps in (32,64,128,256):
        tr=plastic_trajectory(solution.x,records,interval_substeps=substeps,waveform_steps=steps)
        pr,_,_=opening_prediction(solution.x,tr); convergence["waveform"][str(steps)]=metrics(records,pr,tr)
    for steps in (1,2,4,8):
        tr=plastic_trajectory(solution.x,records,interval_substeps=steps,waveform_steps=waveform_steps)
        pr,_,_=opening_prediction(solution.x,tr); convergence["history"][str(steps)]=metrics(records,pr,tr)
    payload={
        "schema":"qste340tm-faithful-native-order-reduction-refit/1",
        "scope":"state-informed source-order analytical replay; not native spatial 1-D",
        "population_policy":"uncapped",
        "signed_return_scope":"inactive for the strictly positive R=0.1--0.7 prescribed waveforms; no signed-return law was removed or refitted",
        "temperature_K":TEMPERATURE_K,"assumed_frequency_Hz":NOMINAL_FREQUENCY_HZ,
        "fixed_constants":{"attempt_frequency_s":ATTEMPT_FREQUENCY_S,"cooperative_hits":COOPERATIVE_HITS,
            "cooperative_tau_s":COOPERATIVE_TAU_S,"event_length_m":EVENT_LENGTH_M,
            "tip_radius_m":TIP_RADIUS_M,"process_zone_m":PROCESS_ZONE_M,"burgers_m":BURGERS_M,
            "shear_modulus_Pa":SHEAR_MODULUS_PA,"forest_density_floor_m2":RHO0_M2,
            "blunting_coefficient":C_BLUNT,"opening_smoothing_ratio":SMOOTHING_RATIO},
        "parameters_vector":solution.x.tolist(),"parameter_names":PARAMETER_NAMES,
        "parameters":unpack(solution.x),"metrics":metrics(records,prediction,trajectory),
        "frozen_metrics":metrics(records,frozen_prediction,fitted["frozen"]),
        "stress_independent_opening_comparator":{"parameters":unpack(independent_vector),
            "metrics":metrics(records,independent_prediction,trajectory),
            "definition":"opening amplitude fixed exactly to zero; same fitted plastic trajectory"},
        "opening_stress_range_Pa":[float(np.min(fitted["opening_stress"])),float(np.max(fitted["opening_stress"]))],
        "raw_barrier_over_kBT_range":[float(np.min(fitted["raw"])),float(np.max(fitted["raw"]))],
        "optimizer":{"success":bool(solution.success),"status":int(solution.status),"message":str(solution.message),
                     "nfev":int(solution.nfev),"cost":float(solution.cost),"optimality":float(solution.optimality),
                     "jacobian_singular_values":fitted["jacobian_singular_values"].tolist(),
                     "jacobian_condition_number":float(fitted["jacobian_singular_values"][0]/max(fitted["jacobian_singular_values"][-1],1e-300)),
                     "jacobian_scope":"five conditional opening coordinates only",
                     "profiled_plastic_curvature":"not estimated; bounded differential-evolution acquisition reported in multi_start"},
        "multi_start":[run for _,_,run in fitted["runs"]],"convergence":convergence,
        "source_bindings":{"rate_csv":"literature_data/fatigue_recovered_review/normalized/qste340tm_constant_amplitude.csv",
                           "history_csv":"literature_data/fatigue_recovered_review/normalized/qste340tm_crack_histories.csv",
                           "rate_sha256":sha256(DATA/"normalized/qste340tm_constant_amplitude.csv"),
                           "history_sha256":sha256(DATA/"normalized/qste340tm_crack_histories.csv"),
                           "preserved_controls":["5703aee3","1bb0928e"]},
        "limitations":["frequency is the user-authorized provisional 10 Hz assumption",
            "minimal aggregate native-order reduction is not a spatial native 1-D equivalence claim",
            "positive-R histories do not exercise signed return",
            "the profiled search is finite and does not prove that all EXP_floor parameterizations fail"],
    }
    (OUT/"fit_faithful_reduction.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    fields=list(records[0])+["predicted_dadn_m_per_cycle","frozen_dadn_m_per_cycle","stress_independent_dadn_m_per_cycle","residual_log10"]+[key for key in trajectory if key != "stress"]
    with (OUT/"predictions_faithful_reduction.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for i,row in enumerate(records):
            extra={"predicted_dadn_m_per_cycle":prediction[i],"frozen_dadn_m_per_cycle":frozen_prediction[i],
                   "stress_independent_dadn_m_per_cycle":independent_prediction[i],
                   "residual_log10":math.log10(prediction[i]/float(row["observed_dadn_m_per_cycle"]))}
            extra.update({key:value[i] for key,value in trajectory.items() if key != "stress"})
            writer.writerow(dict(row)|extra)
    return payload


def make_plots(records: list[dict], fitted: dict) -> None:
    rv=np.array([float(r["R"]) for r in records]);dk=np.array([float(r["DeltaK_MPa_sqrt_m"]) for r in records]);obs=np.array([float(r["observed_dadn_m_per_cycle"]) for r in records]);pred=fitted["prediction"];frozen=fitted["frozen_prediction"]
    independent_vector=fitted["solution"].x.copy();independent_vector[1]=0.
    independent_floor=least_squares(lambda z:np.log10(opening_prediction(np.r_[z[0],independent_vector[1:]],fitted["trajectory"])[0])-np.log10(obs),np.array([independent_vector[0]]),bounds=(LOWER[:1],UPPER[:1]),max_nfev=80)
    independent_vector[0]=independent_floor.x[0];independent=opening_prediction(independent_vector,fitted["trajectory"])[0]
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r;order=np.argsort(dk[m]);ax.scatter(dk[m],obs[m],s=11,c="#333333",alpha=.45,label="measured")
        ax.plot(dk[m][order],pred[m][order],c="#0072B2",lw=2,label="full retained state")
        ax.plot(dk[m][order],frozen[m][order],c="#D55E00",lw=1.5,ls="--",label="frozen initial state")
        ax.plot(dk[m][order],independent[m][order],c="#CC79A7",lw=1.3,ls=":",label="stress-independent opening")
        ax.set(xscale="log",yscale="log",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",title=f"R={r:.1f}");ax.grid(True,which="both",alpha=.2)
    axes[0,0].set_ylabel("da/dN (m/cycle)");axes[1,0].set_ylabel("da/dN (m/cycle)")
    h,l=axes[0,0].get_legend_handles_labels();fig.suptitle("QSTE340TM faithful retained-state joint refit",y=.995);fig.legend(h,l,ncol=4,loc="upper center",bbox_to_anchor=(.5,.965),frameon=False);fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/"common_axis_data_fit.png",dpi=240,bbox_inches="tight");fig.savefig(OUT/"common_axis_data_fit.svg",bbox_inches="tight");plt.close(fig)
    tr=fitted["trajectory"];fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r;ax.plot(dk[m],tr["mobile"][m],label="mobile",c="#56B4E9");ax.plot(dk[m],tr["retained"][m],label="retained",c="#009E73");ax.set(title=f"R={r:.1f}",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel="count in 1 μm zone");ax.grid(True,alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.suptitle("Separate mobile and retained histories",y=.995);fig.legend(h,l,ncol=2,loc="upper center",bbox_to_anchor=(.5,.965),frameon=False);fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/"mobile_retained_histories.png",dpi=240,bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r
        for key,label,color in (("emission","emission","#0072B2"),("trapped","trap","#009E73"),("released","release","#E69F00"),("escaped","mobile escape","#D55E00"),("recovered","retained recovery","#CC79A7")):
            ax.plot(dk[m],np.maximum(tr[key][m],1e-300),label=label,color=color,lw=1.2)
        ax.set(yscale="log",title=f"R={r:.1f}",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel="transfer / cycle");ax.grid(True,which="both",alpha=.2)
    h,l=axes[0,0].get_legend_handles_labels();fig.suptitle("Native-order one-cycle process transfers at observed states",y=.995);fig.legend(h,l,ncol=5,loc="upper center",bbox_to_anchor=(.5,.965),frameon=False);fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/"separate_process_transfers.png",dpi=240,bbox_inches="tight");plt.close(fig)
    p=unpack(fitted["solution"].x);stress=np.linspace(0,max(1e9,float(np.max(fitted["opening_stress"])))*1.05,600);basis=SmoothBroadBasis(p["opening_sigma_b_Pa"],p["opening_exponent"],SMOOTHING_RATIO);g=(p["opening_floor_over_kBT"]+p["opening_amplitude_over_kBT"]*basis.value(stress))*KB_EV_PER_K*TEMPERATURE_K
    fig,ax=plt.subplots(figsize=(8.5,5.5));ax.plot(stress/1e9,g,lw=2,c="#0072B2");ax.axvspan(float(np.min(fitted["opening_stress"]))/1e9,float(np.max(fitted["opening_stress"]))/1e9,color="#E69F00",alpha=.18,label="consumed stress support");ax.set(xlabel="local opening stress (GPa)",ylabel=r"$G^*_{open}$ (eV)",title="Fitted opening barrier and active stress support");ax.grid(True,alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"Gstar_vs_stress.png",dpi=240,bbox_inches="tight");fig.savefig(OUT/"Gstar_vs_stress.svg",bbox_inches="tight");plt.close(fig)


def transfer_audit(records: list[dict]) -> dict:
    prior=json.loads(CONTROL.read_text())["parameters"];vector=pack(prior);pp=plastic_parameters(vector);loads=[]
    for r_value in (.1,.3,.5,.7):
        block=[row for row in records if float(row["R"])==r_value]
        for label,row in (("low",block[0]),("mid",block[len(block)//2]),("high",block[-1])):
            initial=np.array([pp.initial_mobile_count,pp.initial_retained_count,pp.initial_source_activity]);final,flow=one_cycle_with_transfers(float(row["DeltaK_MPa_sqrt_m"]),r_value,initial,pp,256)
            loads.append({"R":r_value,"position":label,"DeltaK_MPa_sqrt_m":row["DeltaK_MPa_sqrt_m"],"initial":initial.tolist(),"final":final.tolist(),"transfers":flow})
    zero=one_cycle_with_transfers(0.,.1,np.array([10.,20.,.4]),pp,256)
    strong=PlasticParameters(.3,2e9,.3,0.,0.,0.,1.);strong_case=one_cycle_with_transfers(45.,.1,np.array([0.,0.,1.]),strong,256)
    payload={"schema":"qste340tm-native-order-transfer-audit/1","operator_order":["source_activity_and_mobile_emission","mobile_retained_exchange","retained_recovery","mobile_transport_escape"],"experimental_low_mid_high":loads,"zero_load_continuation":{"final":zero[0].tolist(),"transfers":zero[1]},"strong_escape_continuation":{"final":strong_case[0].tolist(),"transfers":strong_case[1]},"qualification":"minimal 0-D reduction uses the exact native exchange map and native operator order; it is not a spatial native-owner equivalence claim"}
    (OUT/"transfer_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");return payload


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--max-nfev",type=int,default=180);parser.add_argument("--substeps",type=int,default=2);parser.add_argument("--waveform-steps",type=int,default=128);parser.add_argument("--audit-only",action="store_true");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);records=build_records(DATA);transfer_audit(records)
    if args.audit_only:return 0
    fitted=fit(records,max_nfev=args.max_nfev,substeps=args.substeps,waveform_steps=args.waveform_steps);payload=write_outputs(records,fitted,args.substeps,args.waveform_steps);make_plots(records,fitted);print(json.dumps({"metrics":payload["metrics"],"frozen_metrics":payload["frozen_metrics"],"parameters":payload["parameters"],"optimizer":payload["optimizer"]},indent=2));return 0


if __name__ == "__main__":
    raise SystemExit(main())
