"""Derive next-stage admissions from completed evidence, never from a desired slope."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_v10_2_30_prospective_campaign import read,grid_gate
WORK=ROOT/'runs/prospective_paris_transfer_v1/analysis_work'
ART=ROOT/'artifacts/prospective_paris_candidates'


def main():
    rows=read(WORK/'harvested_results.json');pilots=read(WORK/'pilot_decisions.json')
    admitted={};decisions={}
    freeze=read(ART/'transfer_candidate_freeze_v1.json')
    for candidate in freeze['candidates']:
        cid=candidate['candidate_id'];target=candidate['target'];p=next(r for r in pilots if r['candidate_id']==cid)
        admitted[cid]=[]
        primary=[r for r in rows if r['candidate_id']==cid and r['R']==.1 and r['seed']==1720]
        if p['completed_pilots']!=3:
            decisions[cid]={'status':'PILOT_INCOMPLETE'};continue
        if p['pilot_prediction_gate_passed']:
            admitted[cid].append('GRID');decisions[cid]={'status':'PILOT_PASSED'}
        elif target=='P40':
            admitted[cid].append('GRID');decisions[cid]={'status':'PILOT_FAILED_P40_GEN2_FULL_GRID_STILL_REQUIRED_BY_MISSION'}
        else:
            decisions[cid]={'status':'TERMINAL_BOUNDED_PILOT_FAILURE','classification':'TARGET_NOT_TRANSFERRED_WITH_SINGLE_EXP_FLOOR'}
        if len(primary)==9:
            gate=grid_gate(primary,target);decisions[cid]=gate
            if gate['classification'] in ('PARIS_WINDOW_TRANSFER_VALIDATED','EFFECTIVE_GLOBAL_SLOPE_ONLY'):
                admitted[cid].append('SEED2')
                second=[r for r in rows if r['candidate_id']==cid and r['stage']=='SEED2']
                if len(second)==3:
                    # R transfer remains a diagnostic of the same fixed row;
                    # seed gate failures are retained explicitly in the decision.
                    from scripts.verify_v10_2_30_prospective_campaign import validate_seed_transfer
                    try:
                        m1,m2=validate_seed_transfer(primary,second)
                        decisions[cid]['seed_transfer_passed']=True
                    except ValueError as error:
                        decisions[cid]['seed_transfer_passed']=False
                        decisions[cid]['seed_transfer_error']=str(error)
                    admitted[cid].append('R_TRANSFER')
    result=dict(admitted_stages=admitted,evidence_decisions=decisions,
                source_results='runs/prospective_paris_transfer_v1/analysis_work/harvested_results.json',
                automatic_promotion_from_global_slope_alone=False)
    (WORK/'transfer_stage_decisions.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
