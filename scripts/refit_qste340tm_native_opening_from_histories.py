#!/usr/bin/env python3
"""Refit the authorized opening surface on developed native state histories.

The native stress/blunting/shielding histories are held fixed for this acquisition
step.  Only the existing smooth sloping-floor opening coordinates are optimized;
the selected vector must subsequently be validated by fresh native trajectories.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.refit_qste340tm_resolved_sloping_barrier import (
    ATTEMPT_FREQUENCY_S, COOPERATIVE_HITS, COOPERATIVE_TAU_S,
    KB_EV_PER_K, NOMINAL_FREQUENCY_HZ, TEMPERATURE_K,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
DEFAULT_MATRIX = BASE / "native_signed_bulk_matrix_corrected_transfer_20261028"
DEFAULT_FIT = BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028/fit.json"
DEFAULT_DATA = BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028/predictions_signed_bulk.csv"


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def observed_rate(data, R, dk):
    block = [x for x in data if abs(float(x["R"])-R) < 1e-12]
    xx = np.asarray([float(x["DeltaK_MPa_sqrt_m"]) for x in block])
    yy = np.asarray([float(x["observed_dadn_m_per_cycle"]) for x in block])
    order = np.argsort(xx)
    return float(10**np.interp(np.log10(dk), np.log10(xx[order]), np.log10(yy[order])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    receipt = json.loads((ns.matrix/"MATRIX.json").read_text())
    data = rows(ns.data)
    cases=[]
    phase = 2*np.pi*(np.arange(48)+.5)/48
    for entry in receipt["cases"]:
        R=float(entry["R"]); dk=float(entry["DeltaK"]); trace=rows(entry["csv"])
        first = next((i for i,x in enumerate(trace) if float(x["n_fire"]) >= .5), 0)
        peak=np.asarray([float(x["sigma_tip_Pa"]) for x in trace[first:]])
        fraction=.5*(1+R)+.5*(1-R)*np.cos(phase)
        cases.append((R,dk,(peak[:,None]*fraction[None,:]).reshape(-1),observed_rate(data,R,dk)))
    kbt=KB_EV_PER_K*TEMPERATURE_K
    def predict(v, stress):
        floor,amp,logsigma,logexp=v
        basis=SmoothBroadBasis(math.exp(logsigma)*1e9, math.exp(logexp), 1e-4)
        barrier=floor+amp*basis.value(stress)
        elementary=ATTEMPT_FREQUENCY_S*np.exp(-np.clip(barrier,0,700))
        completed=gammainc(COOPERATIVE_HITS,np.minimum(elementary*COOPERATIVE_TAU_S,1e12))/COOPERATIVE_TAU_S
        return 5e-6*float(np.mean(completed))/NOMINAL_FREQUENCY_HZ
    def residual(v):
        return np.asarray([math.log10(max(predict(v,s),1e-300)/obs) for _,_,s,obs in cases])
    bounds=[(6,28),(2,70),(math.log(.01),math.log(2.0)),(math.log(.02),math.log(.5))]
    de=differential_evolution(lambda v: float(residual(v)@residual(v)),bounds,seed=20261028,popsize=10,maxiter=60,polish=False)
    ls=least_squares(residual,de.x,bounds=np.asarray(bounds).T,max_nfev=300)
    parent=json.loads(ns.fit.read_text()); fit=json.loads(ns.fit.read_text())
    p=fit["parameters"]
    p.update(opening_floor_over_kBT=float(ls.x[0]),opening_amplitude_over_kBT=float(ls.x[1]),opening_sigma_b_Pa=float(math.exp(ls.x[2])*1e9),opening_exponent=float(math.exp(ls.x[3])))
    if "parameters_vector" in fit:
        fit["parameters_vector"][:4] = [float(value) for value in ls.x]
    predictions=[]
    for R,dk,stress,obs in cases:
        predictions.append({"R":R,"DeltaK_MPa_sqrt_m":dk,"observed_rate_m_per_cycle":obs,"frozen_native_state_prediction_m_per_cycle":predict(ls.x,stress)})
    names=("opening_floor_over_kBT","opening_amplitude_over_kBT",
           "log_opening_sigma_b_GPa","log_opening_exponent")
    bound_contact={name:{"lower":bool(np.isclose(value,lo,rtol=0,atol=1e-8)),
                         "upper":bool(np.isclose(value,hi,rtol=0,atol=1e-8))}
                   for name,value,(lo,hi) in zip(names,ls.x,bounds)}
    fit["native_history_refit"]={"scope":"frozen developed native stress histories; requires fresh native validation","source_matrix":str(ns.matrix),"rmse_log10":float(np.sqrt(np.mean(residual(ls.x)**2))),"predictions":predictions,"parent_parameters":parent["parameters"],
        "optimization_bounds":{name:[float(lo),float(hi)] for name,(lo,hi) in zip(names,bounds)},
        "bound_contact":bound_contact,
        "bound_warning":"amplitude and exponent contact upper acquisition bounds; this is not an interior optimum"}
    ns.out.parent.mkdir(parents=True,exist_ok=True);ns.out.write_text(json.dumps(fit,indent=2,sort_keys=True)+"\n")
    print(json.dumps(fit["native_history_refit"]|{"parameters":p},indent=2,sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
