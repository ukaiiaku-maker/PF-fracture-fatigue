"""Single resumable launcher for the complete Ast 2018 paired-model study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .a2 import make_a2_check
from .barriers import make_barrier_outputs
from .search_bounded import resume as bounded_search
from .confirm import confirm_all
from .config import AstAssumptions
from .convergence import audit as audit_convergence
from .curve_support import check_all
from .ensemble import run as run_ensemble
from .environment import write_receipt
from .identifiability import analyze as analyze_identifiability
from .integrity import verify_archive,verify_extracted
from .holdout import analyze as analyze_holdouts
from .local_spatial import PRIORITY11,local_refine
from .outcomes import check_endpoints
from .plots import make_all
from .refine import refine
from .report import build_report
from .selection import archive_superseded_final,retained_vector_path
from .runtime import account


def _exists_complete(path:Path,expected:int|None=None)->bool:
    if not path.exists():return False
    if expected is None:return True
    rows=json.loads(path.read_text())
    return len(rows)==expected


def _needs_local(rows,observed):
    by={(r["level"],r["condition_id"]):r for r in rows}
    for cid in PRIORITY11:
        r=by.get(("reduced",cid));s=by.get(("spatial",cid))
        if r is None or s is None:raise RuntimeError("incomplete priority confirmation")
        if r["status"].startswith("numerical") or s["status"].startswith("numerical"):
            return True
        kr=r["onset_K_Jcontinuous_MPa_sqrt_m"]
        ks=s["onset_K_Jcontinuous_MPa_sqrt_m"]
        if (kr is None)!=(ks is None):return True
        if kr is not None and abs(kr-ks)>.25*max(.2,.05*observed.get(cid,10)):
            return True
        rb={(b["family"],b["curve_id"]):b for b in r["residual_blocks"]}
        for b in s["residual_blocks"]:
            old=rb.get((b["family"],b["curve_id"]))
            if old and abs(old["rms_standardized"]-b["rms_standardized"])>.25:
                return True
    return False


def run_all(source:Path,archive:Path,run_dir:Path,*,workers=8,
            primary_starts=128,alternative_starts=32):
    if workers<1 or workers>8:raise ValueError("workers must be 1..8")
    verify_archive(archive,run_dir)
    write_receipt(run_dir)
    for scenario,starts,seed in (("TABLE_PRIMARY",primary_starts,2018),
                                 ("FIGURE5_ALTERNATIVE",alternative_starts,2019)):
        summary=run_dir/f"search_summary_{scenario.lower()}.json"
        if not summary.exists():
            bounded_search(source,archive,run_dir,scenario=scenario,
                           starts=starts,workers=workers,seed=seed,
                           nominal_K_step=.05,curve_points=25,
                           timeout_s=900.0)
        best=run_dir/f"best_{scenario.lower()}.json"
        if not best.exists():
            warm=None
            if scenario=="FIGURE5_ALTERNATIVE":
                warm=[json.loads((run_dir/"best_table_primary.json").read_text())["x"]]
            refine(source,run_dir,scenario,workers=workers,
                   n_starts=8 if scenario=="TABLE_PRIMARY" else 5,
                   rounds=3 if scenario=="TABLE_PRIMARY" else 2,
                   warm_vectors=warm)
    verify_extracted(run_dir)
    data_dir=run_dir/"input/Ast2018_joint_fit_data"
    from .data import AstData
    data=AstData.load(data_dir)
    for scenario in ("TABLE_PRIMARY","FIGURE5_ALTERNATIVE"):
        cfg=json.loads((run_dir/f"config_{scenario.lower()}.json").read_text())
        assumptions=AstAssumptions(**cfg["ast_assumptions"])
        analytical=run_dir/f"best_{scenario.lower()}.json"
        x=json.loads(analytical.read_text())["x"]
        if not (run_dir/f"convergence_{scenario.lower()}.json").exists():
            audit_convergence(source,run_dir,scenario,analytical)
        priority=run_dir/f"confirmation_{scenario.lower()}_priority11/summary.json"
        if not _exists_complete(priority,22):
            confirm_all(source,data_dir,run_dir,scenario,x,assumptions,
                        levels=("reduced","spatial"),workers=workers,
                        condition_ids=list(PRIORITY11),stage="priority11")
        priority_rows=json.loads(priority.read_text())
        observed={r["condition_id"]:float(r["value"]) for r in data.toughness_scenario(scenario)
                  if r["observable"]=="Kq_Jcontinuous_onset"}
        local=run_dir/f"best_1d_refined_{scenario.lower()}.json"
        if _needs_local(priority_rows,observed) and not local.exists():
            local_refine(source,run_dir,scenario,analytical,indices=(0,3,6),
                         step=.04,rounds=3,workers=workers)
        retained=retained_vector_path(run_dir,scenario)
        xr=json.loads(retained.read_text())["x"]
        if retained.name.endswith("_profile_seeded.json"):
            archive_superseded_final(run_dir,scenario,xr)
        final=run_dir/f"confirmation_{scenario.lower()}/summary.json"
        if not _exists_complete(final,62):
            confirm_all(source,data_dir,run_dir,scenario,xr,assumptions,
                        levels=("reduced","spatial"),workers=workers)
        endpoint_ids=sorted({r["condition_id"] for r in data.no_cleavage})
        endpoint=run_dir/f"confirmation_{scenario.lower()}_endpoint_validation/summary.json"
        if not _exists_complete(endpoint,2*len(endpoint_ids)):
            confirm_all(source,data_dir,run_dir,scenario,xr,assumptions,
                        levels=("reduced","spatial"),workers=workers,
                        condition_ids=endpoint_ids,stage="endpoint_validation",
                        validation_endpoint_horizon=True)
        ident=run_dir/f"identifiability_{scenario.lower()}/profiles.json"
        if not ident.exists():
            analyze_identifiability(source,run_dir,scenario,analytical,workers=workers)
        if not (run_dir/f"holdouts_{scenario.lower()}.json").exists():
            analyze_holdouts(source,run_dir,scenario,analytical,workers=min(workers,7))
        from .sensitivity import analyze as analyze_sensitivity
        if not (run_dir/f"sensitivity_{scenario.lower()}.json").exists():
            analyze_sensitivity(source,run_dir,scenario,analytical,workers=workers)
        if not all((run_dir/f"barriers_{scenario.lower()}/{filename}").exists()
                   for filename in ("barrier_surfaces.csv","barrier_visited_event_rates.csv",
                                    "barrier_sensitivity_envelope.csv",
                                    "barrier_receipt.json","barrier_derivatives.svg",
                                    "barrier_rates.svg")):
            make_barrier_outputs(source,run_dir,scenario,retained)
        if not (run_dir/f"curve_support_{scenario.lower()}.json").exists():
            check_all(data_dir,run_dir,scenario)
        if not (run_dir/f"no_cleavage_outcomes_{scenario.lower()}.json").exists():
            check_endpoints(data_dir,run_dir,scenario)
        reduced_ensemble=run_dir/f"ensemble_{scenario.lower()}_reduced/summary.json"
        reduced_runs=run_dir/f"ensemble_{scenario.lower()}_reduced/runs.json"
        if not _exists_complete(reduced_ensemble,31) or not _exists_complete(reduced_runs,31*8):
            run_ensemble(source,data_dir,run_dir,scenario,retained,
                         level="reduced",n_seeds=8,workers=workers)
        spatial_ensemble=run_dir/f"ensemble_{scenario.lower()}_spatial/summary.json"
        spatial_runs=run_dir/f"ensemble_{scenario.lower()}_spatial/runs.json"
        if not _exists_complete(spatial_ensemble,len(PRIORITY11)) or not _exists_complete(spatial_runs,len(PRIORITY11)*8):
            run_ensemble(source,data_dir,run_dir,scenario,retained,
                         level="spatial",n_seeds=8,workers=workers,
                         condition_ids=list(PRIORITY11))
        figures=run_dir/f"figures_{scenario.lower()}"
        required=("onset_by_rate","cleavage_validation_only","J_resistance_curves",
                  "nominal_K_loading_traces","relative_stiffness","residual_blocks",
                  "unchanged_transfer_onset","unchanged_transfer_J_curves",
                  "unchanged_transfer_native_state")
        if not all((figures/f"{name}.svg").exists() and
                   (figures/f"{name}.png").exists() for name in required):
            make_all(data_dir,run_dir,scenario)
        if not (figures/"A2_protocol_check.svg").exists() or \
           not (figures/"A2_protocol_check.png").exists():
            make_a2_check(data_dir,run_dir,scenario)
    account(run_dir)
    report=build_report(source,data_dir,run_dir)
    print(f"Complete Ast fit report: {report}",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source",type=Path,required=True)
    parser.add_argument("--archive",type=Path,default=Path("Ast2018_tungsten_joint_fit_data.zip"))
    parser.add_argument("--run-dir",type=Path,default=Path("runs/ast2018_joint_fit_v1"))
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--primary-starts",type=int,default=128)
    parser.add_argument("--alternative-starts",type=int,default=32)
    args=parser.parse_args()
    run_all(args.native_source,args.archive,args.run_dir,workers=args.workers,
            primary_starts=args.primary_starts,alternative_starts=args.alternative_starts)


if __name__=="__main__":main()
