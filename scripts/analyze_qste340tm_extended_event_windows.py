#!/usr/bin/env python3
"""Audit rate stabilization over seven-event native QSTE trajectories."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DA=5e-6

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("--matrix",type=Path,required=True)
    ap.add_argument("--baseline-rates",type=Path,
        help="Optional accepted three-event CSV for a transparent comparison.")
    ns=ap.parse_args(argv);receipt=json.loads((ns.matrix/"MATRIX.json").read_text());rows=[]
    baseline={}
    if ns.baseline_rates:
        for item in csv.DictReader(open(ns.baseline_rates,newline="",encoding="utf-8")):
            baseline[(float(item["R"]),float(item["DeltaK_MPa_sqrt_m"]))]=float(
                item["growth_rate_m_per_cycle"])
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for entry in sorted(receipt["cases"],key=lambda x:(x["R"],x["DeltaK"])):
        trace=list(csv.DictReader(open(entry["csv"],newline="",encoding="utf-8")))
        cycles=np.cumsum([float(x["fatigue_cycles"]) for x in trace])
        events=np.asarray([cycles[i] for i,x in enumerate(trace) if float(x["n_fire"])>=.5])
        waits=np.diff(events)
        if len(events)<7: raise RuntimeError(f"{entry} has only {len(events)} events")
        rates={"all_six_wait_rate_m_per_cycle":DA/float(np.mean(waits)),
               "first_two_wait_rate_m_per_cycle":DA/float(np.mean(waits[:2])),
               "last_two_wait_rate_m_per_cycle":DA/float(np.mean(waits[-2:])),
               "last_four_wait_rate_m_per_cycle":DA/float(np.mean(waits[-4:]))}
        shift=math.log10(rates["last_two_wait_rate_m_per_cycle"]/
                         rates["first_two_wait_rate_m_per_cycle"])
        key=(float(entry["R"]),float(entry["DeltaK"])); prior=baseline.get(key)
        campaign=json.loads((Path(entry["csv"]).parents[1]/"CAMPAIGN.json").read_text())
        row={"R":key[0],"DeltaK_MPa_sqrt_m":key[1],
             "event_cycles":events.tolist(),"interevent_waits":waits.tolist(),
             "extension_at_seventh_event_um":float(len(events)*DA*1e6),
             "tail_over_early_log10_rate_shift":shift,
             "driver_L_pz_m":campaign.get("driver_L_pz_m"),
             "blunting_length_um":campaign.get("blunting_length_um"),
             "accepted_three_event_rate_m_per_cycle":prior,
             "all_six_over_accepted_log10_rate_shift":(
                 math.log10(rates["all_six_wait_rate_m_per_cycle"]/prior)
                 if prior else None),
             **rates,"source_csv":entry["csv"]}
        rows.append(row);ax.plot(np.arange(2,len(events)+1),waits,marker="o",
            label=f"R={entry['R']}, ΔK={entry['DeltaK']:g}")
    ax.set(xlabel="Event ending inter-event window",ylabel="Waiting cycles",
           title="QSTE340TM seven-event rate-window audit")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(ns.matrix/"interevent_waits.png",dpi=220);plt.close(fig)
    (ns.matrix/"EXTENDED_EVENT_ANALYSIS.json").write_text(json.dumps({
        "schema":"qste340tm-extended-event-window-audit/1","event_length_m":DA,
        "cases":rows,"decision":"rate_settled only if late-window change is acceptably small; no automatic developed label"},indent=2)+"\n")
    with (ns.matrix/"extended_event_rates.csv").open("w",newline="",encoding="utf-8") as h:
        flat=[]
        for row in rows: flat.append({k:(json.dumps(v) if isinstance(v,list) else v) for k,v in row.items()})
        w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(rows,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
