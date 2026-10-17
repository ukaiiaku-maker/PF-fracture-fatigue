#!/usr/bin/env python3
"""Export decision tables and publication-style figures for one cohesive row."""
from __future__ import annotations

import argparse,csv,json,math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import CohesiveThermalLaw,traction_Pa,traction_barrier


def write(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate-id',required=True)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--additional-run-root',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    a.out.mkdir(parents=True,exist_ok=True)
    fingerprint_path=a.run_root/'consumed_runtime_fingerprint.json'
    if fingerprint_path.exists():
        (a.out/'consumed_runtime_fingerprint.json').write_text(fingerprint_path.read_text())
    rec=next(r for r in csv.DictReader(a.registry.open()) if r['candidate_id']==a.candidate_id);c=json.loads(rec['cohesive_parameter_vector'])
    thermal=CohesiveThermalLaw(c['Gamma_ref_J_m2'],c['t_hat_ref_Pa'],c['S_Gamma_J_m2_K'],c['T_ref_K'])
    law=[]
    for T in (300.,450.,600.,750.,900.,1050.,1200.):
        for x in np.linspace(0,8,321):
            stress_ratio=float(traction_Pa(x*thermal.delta0_m,thermal,T)/thermal.t_hat(T))
            probe=min(max(stress_ratio,1e-12),1-1e-12)*thermal.t_hat(T)
            b=traction_barrier(probe,thermal,T,c['activation_area_m2'])
            law.append({'temperature_K':T,'separation_over_delta0':x,'separation_m':x*thermal.delta0_m,
                'traction_Pa':float(traction_Pa(x*thermal.delta0_m,thermal,T)),'traction_over_peak':stress_ratio,
                'barrier_eV_at_same_traction_ratio':b['barrier_eV'],'activation_volume_m3':b['activation_volume_m3']})
    write(a.out/'cohesive_law_source.csv',law)
    barrier=[]
    for T in (300.,750.,1200.):
        for s in np.concatenate(([1e-8],np.linspace(.002,.998,250),[1.-1e-8])):
            b=traction_barrier(float(s)*thermal.t_hat(T),thermal,T,c['activation_area_m2'])
            barrier.append({'temperature_K':T,'traction_over_peak':float(s),
                'barrier_eV':b['barrier_eV'],'activation_volume_m3':b['activation_volume_m3'],
                'state':b['state']})
    write(a.out/'cohesive_barrier_source.csv',barrier)
    fracture=[]
    for path in sorted((a.run_root/'fracture').glob('T_*K.json')):
        x=json.loads(path.read_text());fracture.append({'temperature_K':x['temperature_K'],'status':x['status'],
            'K_init_MPa_sqrt_m':x['K_init_MPa_sqrt_m'],'K_final_event_MPa_sqrt_m':x['K_final_event_MPa_sqrt_m'],
            'event_count':x['event_count'],'extension_um':x['final_extension_um']})
    if fracture:write(a.out/'thermal_fracture_source.csv',fracture)
    fatigue=[];history=[]
    summaries=[];observed={}
    for root in [a.run_root,*a.additional_run_root]:
        summary_path=root/'fatigue/campaign_summary.json'
        if summary_path.exists():summaries.append(json.loads(summary_path.read_text()))
        for result_path in (root/'fatigue').glob('Kmax_*/result.json'):
            result=json.loads(result_path.read_text());events=result['events']
            for event in events:
                history.append({'Kmax_MPa_sqrt_m':result['Kmax_MPa_sqrt_m'],
                    'event_index':event['event_index'],'cycles':event['cycles'],
                    'extension_m':event['extension_m'],'event_advance_m':event['event_advance_m'],
                    'continuous_mpz_progress_m':event['continuous_mpz_progress_m'],
                    'mobile_count':event['mobile_count'],'retained_count':event['retained_count'],
                    'tip_radius_m':event['tip_radius_m']})
            left=next((e for e in events if e['extension_m']>=20e-6),None)
            right=next((e for e in events if e['extension_m']>=60e-6),None)
            window=None if left is None or right is None else {
                'start_extension_m':left['extension_m'],'end_extension_m':right['extension_m'],
                'start_cycles':left['cycles'],'end_cycles':right['cycles'],
                'rate_m_per_cycle':(right['extension_m']-left['extension_m'])/(right['cycles']-left['cycles']),
                'events_inclusive':right['event_index']-left['event_index']+1}
            observed[result['Kmax_MPa_sqrt_m']]={'Kmax_MPa_sqrt_m':result['Kmax_MPa_sqrt_m'],
                'DeltaK_MPa_sqrt_m':result['DeltaK_MPa_sqrt_m'],'status':result['status'],
                'events':len(events),'cycles':result['final_cycles'],'extension_m':result['final_extension_m'],
                'developed_window':window}
    for summary in summaries:
      for row in summary['loads']:observed[row['Kmax_MPa_sqrt_m']]=row
    if observed:
        for row in observed.values():
            window=row['developed_window'] or {}
            fatigue.append({**{k:row[k] for k in ('Kmax_MPa_sqrt_m','DeltaK_MPa_sqrt_m','status','events','cycles','extension_m')},
                'developed_rate_m_per_cycle':window.get('rate_m_per_cycle'),'window_start_m':window.get('start_extension_m'),
                'window_end_m':window.get('end_extension_m'),'window_events':window.get('events_inclusive')})
        developed=[r for r in fatigue if r['developed_rate_m_per_cycle'] is not None]
        developed.sort(key=lambda r:r['DeltaK_MPa_sqrt_m'])
        for i,row in enumerate(developed):
            row['adjacent_slope_from_previous']=None if i==0 else math.log(row['developed_rate_m_per_cycle']/developed[i-1]['developed_rate_m_per_cycle'])/math.log(row['DeltaK_MPa_sqrt_m']/developed[i-1]['DeltaK_MPa_sqrt_m'])
        write(a.out/'developed_fatigue_source.csv',fatigue)
    if history:
        history.sort(key=lambda r:(r['Kmax_MPa_sqrt_m'],r['event_index']))
        write(a.out/'representative_event_history_source.csv',history)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(4.5,3.4))
        for T in (300.,750.,1200.):
            subset=[r for r in law if r['temperature_K']==T]
            ax.plot([r['separation_over_delta0'] for r in subset],[r['traction_Pa']/1e9 for r in subset],label=f'{int(T)} K')
        ax.set_xlabel(r'$\delta/\delta_0$');ax.set_ylabel('Traction (GPa)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out/'cohesive_traction.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(4.5,3.4))
        for T in (300.,750.,1200.):
            subset=[r for r in barrier if r['temperature_K']==T]
            ax.semilogy([r['traction_over_peak'] for r in subset],[max(r['barrier_eV'],1e-15) for r in subset],label=f'{int(T)} K')
        ax.set_xlabel(r'$\sigma_n/\hat{t}$');ax.set_ylabel('Activation barrier (eV)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out/'cohesive_barrier.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(4.5,3.4))
        for T in (300.,750.,1200.):
            subset=[r for r in barrier if r['temperature_K']==T]
            ax.semilogy([r['traction_over_peak'] for r in subset],[r['activation_volume_m3'] for r in subset],label=f'{int(T)} K')
        ax.set_xlabel(r'$\sigma_n/\hat{t}$');ax.set_ylabel(r'Activation volume (m$^3$)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(a.out/'cohesive_activation_volume.png',dpi=180);plt.close(fig)
        if fracture:
            fig,ax=plt.subplots(figsize=(4.5,3.4));ax.plot([r['temperature_K'] for r in fracture],[r['K_init_MPa_sqrt_m'] for r in fracture],'o-')
            ax.set_xlabel('Temperature (K)');ax.set_ylabel(r'Initiation $K$ (MPa $\sqrt{m}$)');fig.tight_layout();fig.savefig(a.out/'thermal_fracture.png',dpi=180);plt.close(fig)
        developed=[r for r in fatigue if r.get('developed_rate_m_per_cycle') is not None]
        if developed:
            fig,ax=plt.subplots(figsize=(4.5,3.4));ax.loglog([r['DeltaK_MPa_sqrt_m'] for r in developed],[r['developed_rate_m_per_cycle'] for r in developed],'o-')
            ax.set_xlabel(r'$\Delta K$ (MPa $\sqrt{m}$)');ax.set_ylabel(r'$da/dN$ (m cycle$^{-1}$)');fig.tight_layout();fig.savefig(a.out/'developed_fatigue.png',dpi=180);plt.close(fig)
        if history:
            fig,ax=plt.subplots(figsize=(4.5,3.4))
            loads=sorted({r['Kmax_MPa_sqrt_m'] for r in history})
            for K in loads:
                subset=[r for r in history if r['Kmax_MPa_sqrt_m']==K]
                ax.plot([r['cycles'] for r in subset],[1e6*r['extension_m'] for r in subset],label=f'{K:.3f}')
            ax.set_xlabel('Cycles');ax.set_ylabel(r'Committed extension ($\mu$m)');ax.legend(frameon=False,title=r'$K_{max}$');fig.tight_layout();fig.savefig(a.out/'representative_event_history.png',dpi=180);plt.close(fig)
    except Exception as exc:(a.out/'plot_failure.json').write_text(json.dumps({'error':repr(exc)},indent=2)+'\n')
    developed=[r for r in fatigue if r.get('developed_rate_m_per_cycle') is not None]
    developed.sort(key=lambda r:r['DeltaK_MPa_sqrt_m'])
    slopes=[r.get('adjacent_slope_from_previous') for r in developed[1:]]
    endpoint=(None if len(developed)<2 else math.log(developed[-1]['developed_rate_m_per_cycle']/developed[0]['developed_rate_m_per_cycle'])/
              math.log(developed[-1]['DeltaK_MPa_sqrt_m']/developed[0]['DeltaK_MPa_sqrt_m']))
    width=(None if len(developed)<2 else math.log10(developed[-1]['developed_rate_m_per_cycle']/developed[0]['developed_rate_m_per_cycle']))
    target=bool(len(developed)>=2 and width>=2 and all(3.4<=x<=4.6 for x in slopes))
    decision={'candidate_id':a.candidate_id,'family':rec['family'],'complete_row_sha256':rec['complete_row_sha256'],
        'fracture_native_points':len(fracture),'fatigue_native_points':len(fatigue),
        'developed_fatigue_points':sum(r.get('developed_rate_m_per_cycle') is not None for r in fatigue),
        'adjacent_developed_slopes':slopes,'endpoint_developed_slope':endpoint,
        'connected_rate_width_decades':width,'primary_slope_target_pass':target,
        'classification':('PENDING_CAUSAL_AND_REPLICATE_QUALIFICATION' if not developed else
          ('DEVELOPED_PRIMARY_SLOPE_TARGET' if target else 'DEVELOPED_RECURRENT_NON_TARGET_SLOPE'))}
    (a.out/'analysis_summary.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n');print(json.dumps(decision,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
