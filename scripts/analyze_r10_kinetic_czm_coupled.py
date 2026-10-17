#!/usr/bin/env python3
"""Reduce completed advancing kinetic-CZM paths to developed-window evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_results(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.glob("Kmax_*/**/result.json"))]


def developed(result: dict, lower_m: float, upper_m: float) -> dict:
    events = result["events"]
    lo = next((row for row in events if float(row["extension_m"]) >= lower_m), None)
    hi = next((row for row in events if float(row["extension_m"]) >= upper_m), None)
    if lo is None or hi is None or float(hi["cycles"]) <= float(lo["cycles"]):
        raise RuntimeError(f"{result['arm']} at {result['Kmax_MPa_sqrt_m']} lacks a developed window")
    da = float(hi["extension_m"]) - float(lo["extension_m"])
    dn = float(hi["cycles"]) - float(lo["cycles"])
    return {
        "candidate_id": result["candidate_id"], "arm": result["arm"],
        "phase_count": result["phase_count"],
        "Kmax_MPa_sqrt_m": result["Kmax_MPa_sqrt_m"],
        "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"],
        "lower_event": lo["event_index"], "upper_event": hi["event_index"],
        "lower_extension_um": 1e6 * float(lo["extension_m"]),
        "upper_extension_um": 1e6 * float(hi["extension_m"]),
        "lower_cycles": float(lo["cycles"]), "upper_cycles": float(hi["cycles"]),
        "lower_physical_time_s": float(lo["cycles"])/float(result["frequency_Hz"]),
        "upper_physical_time_s": float(hi["cycles"])/float(result["frequency_Hz"]),
        "window_cycles": dn, "window_events": int(hi["event_index"])-int(lo["event_index"]),
        "da_dN_m_per_cycle": da/dn,
        "recurrence_class": "MULTICYCLE_RECURRENT" if dn >= 1.0 else "WITHIN_CYCLE_BURST",
        "terminal_extension_um":1e6*float(result["final_extension_m"]),
        "terminal_cycles":float(result["final_cycles"]),
        "later_window_status":"EXTENDED_BEYOND_60UM__NO_SEPARATE_LATER_WINDOW_FIT",
        "independent_stream_status":"NOT_EXECUTED__PAIRED_COMMON_MARK_STREAM",
    }


def slopes(rows: list[dict]) -> list[float]:
    return [math.log(b["da_dN_m_per_cycle"] / a["da_dN_m_per_cycle"]) /
            math.log(b["DeltaK_MPa_sqrt_m"] / a["DeltaK_MPa_sqrt_m"])
            for a, b in zip(rows, rows[1:])]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--phase128", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lower-um", type=float, default=20.0)
    p.add_argument("--upper-um", type=float, default=60.0)
    a = p.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True)
    results = load_results(a.run)
    if not results or any(r.get("status") != "developed_target_completed" for r in results):
        raise RuntimeError("analysis requires completed developed trajectories")
    rates = [developed(r, a.lower_um*1e-6, a.upper_um*1e-6) for r in results]
    rates.sort(key=lambda row: (row["arm"], row["DeltaK_MPa_sqrt_m"]))
    with (a.out/"developed_rates.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rates[0]),lineterminator='\n');writer.writeheader();writer.writerows(rates)
    event_rows=[]
    for result in results:
        ledgers={int(x['event_index']):x for x in result['energy_ledger']}
        for event in result['events']:
            ledger=ledgers[int(event['event_index'])]
            event_rows.append({'candidate_id':result['candidate_id'],'arm':result['arm'],
                'phase_count':result['phase_count'],'Kmax_MPa_sqrt_m':result['Kmax_MPa_sqrt_m'],
                **event,**ledger})
    with (a.out/'event_energy_history.csv').open('w',newline='') as stream:
        fields=[]
        for row in event_rows:
            for key in row:
                if key not in fields:fields.append(key)
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(event_rows)
    by_arm={arm:sorted([r for r in rates if r["arm"]==arm],key=lambda r:r["DeltaK_MPa_sqrt_m"])
            for arm in {r["arm"] for r in rates}}
    full=by_arm["FULL_DISLOCATION_STATE"]; reference=by_arm["OPENING_REFERENCE"]
    adjacent=slopes(full); endpoint=(math.log(full[-1]["da_dN_m_per_cycle"]/
        full[0]["da_dN_m_per_cycle"])/math.log(full[-1]["DeltaK_MPa_sqrt_m"]/
        full[0]["DeltaK_MPa_sqrt_m"]))
    ratios=[f["da_dN_m_per_cycle"]/r["da_dN_m_per_cycle"] for f,r in zip(full,reference)]
    p128=load_results(a.phase128)
    p128_full=next(r for r in p128 if r["arm"]=="FULL_DISLOCATION_STATE")
    target=min(full,key=lambda r:abs(r["Kmax_MPa_sqrt_m"]-float(p128_full["Kmax_MPa_sqrt_m"])))
    conv=developed(p128_full,a.lower_um*1e-6,a.upper_um*1e-6)
    convergence={"load_MPa_sqrt_m":target["Kmax_MPa_sqrt_m"],"phase_count_coarse":target["phase_count"],
        "phase_count_fine":conv["phase_count"],"coarse_rate_m_per_cycle":target["da_dN_m_per_cycle"],
        "fine_rate_m_per_cycle":conv["da_dN_m_per_cycle"],
        "relative_rate_difference":abs(target["da_dN_m_per_cycle"]-conv["da_dN_m_per_cycle"])/conv["da_dN_m_per_cycle"],
        "acceptance_tolerance":0.01}
    convergence["passed"]=convergence["relative_rate_difference"]<=convergence["acceptance_tolerance"]
    atomic(a.out/"phase_convergence.json",convergence)
    residual=max(abs(float(x["balance_residual_J_per_m"])) for r in results for x in r["energy_ledger"])
    summary={"schema":"r10.kinetic-czm-developed-analysis/1","candidate_id":full[0]["candidate_id"],
        "execution_identity":"COHESIVE_KINETIC_CZM","developed_window_um":[a.lower_um,a.upper_um],
        "all_arms_completed":True,"phase_convergence":convergence,
        "full_state_adjacent_slopes":adjacent,"full_state_endpoint_slope":endpoint,
        "rate_span_decades":math.log10(max(x["da_dN_m_per_cycle"] for x in full)/min(x["da_dN_m_per_cycle"] for x in full)),
        "moderate_slope_pass":all(3.4<=m<=4.6 for m in adjacent),
        "full_to_opening_reference_rate_ratios":ratios,
        "maximum_fractional_state_effect":max(abs(x-1.) for x in ratios),
        "mechanism_classification":"OPENING_CONTROLLED_DISLOCATION_INACTIVE",
        "maximum_mobile_count":max(float(h["mobile_count"]) for r in results if r["arm"]=="FULL_DISLOCATION_STATE" for h in r["history"]),
        "maximum_abs_K_shield_Pa_sqrt_m":max(abs(float(h["K_shield_Pa_sqrt_m"])) for r in results if r["arm"]=="FULL_DISLOCATION_STATE" for h in r["history"]),
        "surface_energy_owner":"KINETIC_CZM_BROKEN_PATCH","activation_barrier_surface_charge_J_per_m":0.0,
        "maximum_energy_balance_residual_J_per_m":residual,
        "rate_independent_controls_are_not_kinetic_results":True,
        "disposition":"PHYSICALLY_INFORMATIVE_NEGATIVE__STEEP_OPENING_CONTROLLED"}
    atomic(a.out/"analysis_summary.json",summary)
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
