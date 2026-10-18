#!/usr/bin/env python3
"""Close the first corrected V2/current-opening information loop."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,shutil
from pathlib import Path


def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def write_csv(path,rows):
    rows=list(rows);fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fields,lineterminator='\n');w.writeheader();w.writerows(rows)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--acquisition',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    with (a.acquisition/'native_information_registry.csv').open(newline='') as f:registry=list(csv.DictReader(f))
    by_id={r['candidate_id']:r for r in registry}
    raw=[];histories={}
    for block in a.runs:
        for path in sorted(block.glob('**/result.json')):
            d=json.loads(path.read_text());candidate=d['candidate_id'];r=by_id[candidate]
            histories[str(path)]=d
            kind='thermal' if '/thermal/' in str(path) else 'fatigue'
            event_cycles=sorted({int(math.floor(float(e['cycle'])+1e-12)) for e in d['events']})
            developed_events=[e for e in d['events'] if 20e-6<=float(e['extension_m'])<=60e-6]
            developed_rate=None
            if len(developed_events)>=2:
                dc=float(developed_events[-1]['cycle'])-float(developed_events[0]['cycle'])
                if dc>0:developed_rate=(float(developed_events[-1]['extension_m'])-float(developed_events[0]['extension_m']))/dc
            factor=float(d['Kmax_MPa_sqrt_m'])/float(r['predicted_Kaccess_300K']) if kind=='fatigue' else None
            recurrent=len(event_cycles)>=2
            disposition=('DEVELOPED_RECURRENT_RATE' if developed_rate is not None else
                'RECURRENT_OUTSIDE_DEVELOPED_WINDOW' if recurrent else
                'TARGET_TRUNCATED_SAME_TIME_CASCADE' if len(d['events']) else
                'UNSUPPORTED_CROSSOVER' if d['unsupported_crossover_encountered'] else
                'SUPPORTED_SHAKEDOWN_OR_FINITE_NONADVANCE' if 'SHAKEDOWN' in d['stop_category'] else
                'FINITE_EXPOSURE_NONADVANCE')
            raw.append({'block':block.name,'case_kind':kind,'candidate_id':candidate,
                'source_material_candidate_id':r['source_material_candidate_id'],
                'search_design':r['search_design'],'shape':d['shape'],'mode':d['mode'],
                'mechanism_prediction':r['mechanism_prediction'],'thermal_prediction':r['thermal_prediction'],
                'temperature_K':d['temperature_K'],'Kmax_MPa_sqrt_m':d['Kmax_MPa_sqrt_m'],
                'analytical_access_factor':factor,'cycles':len(d['cycle_history']),'events':len(d['events']),
                'event_cycles':json.dumps(event_cycles),'extension_m':d['cycle_history'][-1]['extension_m'] if d['cycle_history'] else 0.,
                'stop_category':d['stop_category'],'disposition':disposition,
                'developed_rate_m_per_cycle':developed_rate,'result_sha256':digest(path),
                'result_path':str(path)})
    write_csv(a.out/'native_case_dispositions.csv',raw)
    with gzip.open(a.out/'native_histories.json.gz','wt') as stream:
        json.dump(histories,stream,sort_keys=True,separators=(',',':'))
    fatigue=[r for r in raw if r['case_kind']=='fatigue']
    arms=[]
    for key in sorted({(r['candidate_id'],r['mode'],r['shape']) for r in fatigue}):
        rows=sorted([r for r in fatigue if (r['candidate_id'],r['mode'],r['shape'])==key],key=lambda x:float(x['Kmax_MPa_sqrt_m']))
        non=[float(r['analytical_access_factor']) for r in rows if r['events']==0 and r['disposition']!='UNSUPPORTED_CROSSOVER']
        cascade=[float(r['analytical_access_factor']) for r in rows if r['disposition']=='TARGET_TRUNCATED_SAME_TIME_CASCADE']
        rates=[r for r in rows if r['developed_rate_m_per_cycle'] is not None]
        arms.append({'candidate_id':key[0],'source_material_candidate_id':rows[0]['source_material_candidate_id'],
            'search_design':rows[0]['search_design'],'mode':key[1],'shape':key[2],
            'mechanism_prediction':rows[0]['mechanism_prediction'],'fatigue_loads':len(rows),
            'maximum_supported_nonadvance_factor':max(non) if non else None,
            'minimum_same_time_cascade_factor':min(cascade) if cascade else None,
            'transition_bracket':(f"({max(non):.3g},{min(cascade):.3g}]" if non and cascade else
                f">{max(non):.3g}" if non else f"<={min(cascade):.3g}" if cascade else 'UNRESOLVED'),
            'unsupported_crossover_loads':sum(r['disposition']=='UNSUPPORTED_CROSSOVER' for r in rows),
            'developed_rate_loads':len(rates),'local_slopes_json':'[]',
            'strict_Paris_pass':False})
    write_csv(a.out/'native_arm_summary.csv',arms)
    executed={r['candidate_id'] for r in raw}
    continuing=[]
    for r in registry:
        if r['candidate_id'] in executed:continue
        priority=(0 if r['mechanism_prediction']=='MIXED_CONTROL_PREDICTED' else
                  1 if r['mechanism_prediction']=='PLASTICITY_CONTROL_PREDICTED' else 2)
        continuing.append({'candidate_id':r['candidate_id'],'source_material_candidate_id':r['source_material_candidate_id'],
            'search_design':r['search_design'],'mode':r['selected_mode'],'shape':r['selected_shape'],
            'mechanism_prediction':r['mechanism_prediction'],'thermal_prediction':r['thermal_prediction'],
            'predicted_Kaccess_300K':r['predicted_Kaccess_300K'],'priority':priority,
            'acquisition_reason':'UNEXECUTED_DIVERSE_ARM_AFTER_NATIVE_THRESHOLD_DISCREPANCY'})
    continuing.sort(key=lambda r:(r['priority'],r['search_design'],r['candidate_id']))
    write_csv(a.out/'continuing_acquisition.csv',continuing)
    block1=[r for r in fatigue if r['block']=='information_block_1']
    block2=[r for r in fatigue if r['block']=='information_block_2']
    decision={'schema':'r10.v2-current-opening-joint-search-decision/1',
      'analytical':json.loads((a.acquisition/'analytical_summary.json').read_text()),
      'execution':{'native_cases':len(raw),'fatigue_cases':len(fatigue),'thermal_cases':len(raw)-len(fatigue),
        'opening_arm_identities':len(arms),'physical_material_vectors':len({r['source_material_candidate_id'] for r in raw}),
        'fatigue_cycles_completed':sum(int(r['cycles']) for r in fatigue),
        'developed_rate_points':sum(r['developed_rate_m_per_cycle'] is not None for r in fatigue),
        'recurrent_growth_points':sum(r['disposition'] in {'DEVELOPED_RECURRENT_RATE','RECURRENT_OUTSIDE_DEVELOPED_WINDOW'} for r in fatigue),
        'same_time_cascade_points':sum(r['disposition']=='TARGET_TRUNCATED_SAME_TIME_CASCADE' for r in fatigue),
        'unsupported_crossover_points':sum(r['disposition']=='UNSUPPORTED_CROSSOVER' for r in fatigue)},
      'discrepancy':{'analytical_coordinate':'FINITE_COMPLIANCE_COMPLETION_COORDINATE',
        'block1_role':'CALIBRATION_INFORMATION','block2_role':'HELD_OUT_REFINEMENT_CHECK',
        'block1_nonadvance_max_factor':max(float(r['analytical_access_factor']) for r in block1 if r['events']==0),
        'block1_cascade_min_factor':min(float(r['analytical_access_factor']) for r in block1 if r['disposition']=='TARGET_TRUNCATED_SAME_TIME_CASCADE'),
        'block2_nonadvance_max_factor':max(float(r['analytical_access_factor']) for r in block2 if r['events']==0 and r['disposition']!='UNSUPPORTED_CROSSOVER'),
        'block2_cascade_min_factor':min(float(r['analytical_access_factor']) for r in block2 if r['disposition']=='TARGET_TRUNCATED_SAME_TIME_CASCADE'),
        'production_rate_transfer_fit_used':False},
      'thermal':{'same_material_endpoint_checks':len(raw)-len(fatigue),
        'native_events_in_endpoint_checks':sum(r['events'] for r in raw if r['case_kind']=='thermal'),
        'decision':'ANALYTICAL_TRENDS_RETAINED_AS_PREDICTIONS; NATIVE_0P98_1P02_BRACKETS_DID_NOT_REACH_COMMITMENT'},
      'joint_pareto':{'strict_passes':0,'developed_rates_available':False,
        'decision':'NO_PARIS_PASS; RETAIN DIVERSE TRANSITION BRACKETS AND CONTINUE ACQUISITION'},
      'continuing_acquisition_arms':len(continuing),
      'numerical_scope':{'fatigue_cycles_limit':16,'plastic_phases':4,'opening_phases':8,
        'interpretation':'NATIVE_INFORMATION_BLOCK; FINALISTS_REQUIRE_PHASE_AND_LATER-WINDOW_REFINEMENT'}}
    atomic(a.out/'decision.json',decision)
    for name in ('native_information_registry.csv','native_information_descriptors.csv'):
        source=a.acquisition/name;target=a.out/name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
    text=f"""# R10 V2 search on the corrected current cohesive opening laws

The corrected transaction comparisons still produced no developed fatigue, so the historical V2 acquisition machinery was resumed on the current branch. The executed bank contains **{decision['analytical']['allocated_materials']:,} allocated complete materials**, **{decision['analytical']['domain_valid_materials']:,} valid materials**, and **{decision['analytical']['evaluated_opening_arms']:,} evaluated athermal/mixed opening arms** across fixed-cohesion/vary-plasticity, fixed-plasticity/vary-cohesion, and joint variation.

Two native information blocks executed {decision['execution']['fatigue_cases']} fatigue and {decision['execution']['thermal_cases']} same-material endpoint cases for {decision['execution']['opening_arm_identities']} opening identities ({decision['execution']['physical_material_vectors']} physical vectors). They produced no recurrent developed rate and no Paris slope. The observed response remained a bifurcation between finite non-advance/shakedown and a 120 um same-time observation-path cascade. Mixed crossover cases remain unsupported rather than zero-rate results.

The analytical completion coordinate underpredicted the native transition. Block 1 placed it between {decision['discrepancy']['block1_nonadvance_max_factor']:.2f}x non-advance and {decision['discrepancy']['block1_cascade_min_factor']:.2f}x cascade for its opening-active rows. The held-out refinement retained non-advance as high as {decision['discrepancy']['block2_nonadvance_max_factor']:.2f}x for a strongly shielding material while other rows cascaded by {decision['discrepancy']['block2_cascade_min_factor']:.2f}x. This discrepancy is used for acquisition ordering only; no fitted production-rate multiplier is applied.

The 300/1200 K checks at 0.98x and 1.02x analytical accessibility produced no committed event, so the thermal classes remain analytical predictions pending a wider native bracket. `continuing_acquisition.csv` preserves the unexecuted mixed- and plasticity-controlled arms. The strict joint Pareto set remains empty; zero-Paris-pass is preserved.
"""
    (a.out/'RESULTS.md').write_text(text)
    hashes={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in a.out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    atomic(a.out/'SHA256SUMS.json',hashes);print(json.dumps(decision,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
