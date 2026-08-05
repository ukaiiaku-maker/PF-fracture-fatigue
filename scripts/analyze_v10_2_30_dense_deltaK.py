#!/usr/bin/env python3
"""Final measured/censored dense-DeltaK analysis; no empirical curve fitting."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
from scripts import v10230_dense_deltaK_supervisor as dense

GRID=(.55,.75,.775,.8,.825,.85,.875,.9,.925,.95)

def load(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return {} if default is None else default
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def weighted(rows):
    dn=sum(float(r['delta_N']) for r in rows if r.get('delta_N') is not None)
    return sum(float(r['committed_delta_a_m']) for r in rows if r.get('delta_N') is not None)/dn if dn>0 else None

def source_case(label,seed,fraction,qualification,dense_root,developed):
    token=str(fraction).replace('.','p') if fraction in {.55,.75,.95} else f'{fraction:.3f}'.replace('.','p')
    name=f'{label}_f{token}_seed{seed}'
    root=qualification if fraction==.55 else developed if fraction==.95 else dense_root
    return root/name,root

def analyze_case(label,option,seed,critical,fraction,case,source_root,family_hash):
    out=case/'output'; summary=load(out/'developed_fatigue_growth_summary.json'); control=load(out/'v10_2_30_fixed_deltaK_control.json')
    status=load(case/'qualification_status.json'); descriptor=load(out/'run_state_checkpoint.json'); events=[]
    for event in summary.get('event_measurements',[]):
        dn=event.get('cycles_between_events'); da=event.get('projected_advance_m')
        events.append({'class':label,'option':option,'seed':seed,'fraction':fraction,'deltaK_MPa_sqrt_m':critical*fraction,
            'Kmax_MPa_sqrt_m':critical*fraction/.9,'event_index':event.get('event_index'),'cycle_start':event.get('cycles_pre'),
            'cycle_end':event.get('cycles_post'),'delta_N':dn,'extension_start_m':event.get('projected_extension_pre_m'),
            'extension_end_m':event.get('projected_extension_post_m'),'committed_delta_a_m':da,
            'interval_da_dN_m_per_cycle':float(da)/float(dn) if da is not None and dn else None,
            'threshold':event.get('threshold_action'),'physical_hazard_action':event.get('physical_hazard_action'),
            'proposal_length_m':event.get('stochastic_proposed_advance_m'),'selected_direction':event.get('selected_direction'),
            'selected_endpoint':event.get('selected_endpoint'),'energy_gate_result':event.get('energy_gate_outcome'),
            'geometry_commit_result':event.get('geometry_commit_inserted'),'acceleration_mode':event.get('acceleration_modes'),
            'restart_provenance':event.get('restart_provenance')})
    final_ext=summary.get('final_projected_extension_um',status.get('crack_extension_um',0.0))
    final_cycles=summary.get('cycles_consumed',control.get('cycles_max'))
    reached=summary.get('target_reached') is True
    classification='completed_growth' if reached else ('right_censored_after_growth' if events else 'right_censored_no_growth') if status.get('status')=='censored' or 'right_censored' in str(control.get('censor_status','')) else status.get('status','incomplete_restartable')
    late=[e for e in events if e.get('extension_end_m') is not None and float(e['extension_end_m'])>float(final_ext or 0)*1e-6-50e-6]
    return {'class':label,'option':option,'seed':seed,'fraction':fraction,'absolute_deltaK_MPa_sqrt_m':critical*fraction,
        'Kmax_MPa_sqrt_m':critical*fraction/.9,'terminal_classification':classification,'cycle_horizon':1e12 if fraction==.55 else 1e14 if fraction<.95 else 1e12,
        'final_cycles':final_cycles,'final_extension_um':final_ext,'event_count':len(events),'full_trajectory_da_dN_m_per_cycle':weighted(events) if reached else None,
        'late_50um_da_dN_m_per_cycle':weighted(late) if reached else None,'censor_flag':classification.startswith('right_censored'),
        'source_campaign':str(source_root.resolve()),'source_or_restart_checkpoint':descriptor.get('generation'),
        'git_head':summary.get('provenance',{}).get('git_head'),'family_hash':family_hash},events

def plots(cases,events,out):
    files=[]
    for label in dense.q.OPTIONS:
        selected=[r for r in cases if r['class']==label]; ev=[r for r in events if r['class']==label]
        for xkey,suffix,xlabel in [('absolute_deltaK_MPa_sqrt_m','absolute','DeltaK (MPa sqrt(m))'),('fraction','fraction','Normalized DeltaK fraction')]:
            fig,ax=plt.subplots(figsize=(8,5))
            grown=[r for r in selected if r['late_50um_da_dN_m_per_cycle'] is not None]
            ax.plot([r[xkey] for r in grown],[r['late_50um_da_dN_m_per_cycle'] for r in grown],'o-',label='late 50 um')
            ax.plot([r[xkey] for r in grown],[r['full_trajectory_da_dN_m_per_cycle'] for r in grown],'s--',label='full trajectory')
            cens=[r for r in selected if r['censor_flag']]
            if grown:
                floor=min(r['late_50um_da_dN_m_per_cycle'] for r in grown)/5
                ax.scatter([r[xkey] for r in cens],[floor]*len(cens),marker='v',facecolors='none',edgecolors='k',label='censored (annotated horizon)')
                for r in cens: ax.annotate(f"censor {r['cycle_horizon']:.0e} N",(r[xkey],floor),rotation=75,fontsize=7)
            ax.set_yscale('log');ax.set(xlabel=xlabel,ylabel='da/dN (m/cycle)',title=label);ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout()
            name=f'{label}_da_dN_vs_{suffix}.png';fig.savefig(out/name,dpi=180);plt.close(fig);files.append(name)
        for ykey,name,ylabel,log in [('interval_da_dN_m_per_cycle','event_rates','Event da/dN'),('delta_N','waiting_cycles','Waiting cycles'),('committed_delta_a_m','event_lengths','Event length (m)')]:
            fig,ax=plt.subplots(figsize=(8,5))
            for fraction in GRID:
                rows=[r for r in ev if r['fraction']==fraction and r.get(ykey) is not None]
                if rows: ax.plot([r['event_index'] for r in rows],[r[ykey] for r in rows],marker='o',label=f'{fraction:.3f}')
            if log:ax.set_yscale('log')
            ax.set(xlabel='Event index',ylabel=ylabel,title=label);ax.grid(True,which='both',alpha=.25);ax.legend(ncol=2,fontsize=7);fig.tight_layout();fn=f'{label}_{name}.png';fig.savefig(out/fn,dpi=180);plt.close(fig);files.append(fn)
    for xkey,suffix in [('absolute_deltaK_MPa_sqrt_m','absolute'),('fraction','fraction')]:
        fig,ax=plt.subplots(figsize=(9,6))
        for label in dense.q.OPTIONS:
            rows=[r for r in cases if r['class']==label and r['late_50um_da_dN_m_per_cycle'] is not None]
            ax.plot([r[xkey] for r in rows],[r['late_50um_da_dN_m_per_cycle'] for r in rows],'o-',label=label)
        ax.set_yscale('log');ax.set(xlabel=xkey,ylabel='Late-50-um da/dN (m/cycle)');ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fn=f'four_class_da_dN_vs_{suffix}.png';fig.savefig(out/fn,dpi=180);plt.close(fig);files.append(fn)
    return files

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('dense',type=Path);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--developed',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args(argv)
    root=a.dense.resolve();out=(a.out or root/'final_analysis').resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=load(root/'dense_deltaK_matrix.json');family_hash=manifest.get('family_hash');cases=[];events=[]
    for label,(option,seed,critical) in dense.q.OPTIONS.items():
        for fraction in GRID:
            case,source=source_case(label,seed,fraction,a.qualification.resolve(),root,a.developed.resolve())
            row,case_events=analyze_case(label,option,seed,critical,fraction,case,source,family_hash);cases.append(row);events.extend(case_events)
    write_csv(out/'complete_event_intervals.csv',events);write_csv(out/'case_summary.csv',cases);write_csv(out/'combined_curve.csv',cases)
    write_csv(out/'censor_table.csv',[r for r in cases if r['censor_flag']]);write_csv(out/'failure_table.csv',[r for r in cases if r['terminal_classification'] in {'failed','blocked_before_launch'}])
    files=plots(cases,events,out); validation={'all_40_rows':len(cases)==40,'unique_rows':len({(r['class'],r['fraction']) for r in cases})==40,'no_zero_censor_rates':all(r['late_50um_da_dN_m_per_cycle'] is None for r in cases if r['censor_flag']),'all_launch_tasks_terminal':all(r['terminal_classification'] in {'completed_growth','right_censored_no_growth','right_censored_after_growth'} for r in cases if .75<=r['fraction']<.95)}
    (out/'validation_report.json').write_text(json.dumps(validation,indent=2,sort_keys=True)+'\n');(out/'production_summary.json').write_text(json.dumps({'schema':'v10.2.30_dense_deltaK_analysis_v1','cases':cases,'plots':files,'validation':validation,'empirical_Paris_law_fit':False},indent=2,sort_keys=True)+'\n');(out/'provenance_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    write_csv(out/'full_launch_matrix.csv',manifest.get('cases',[]));write_csv(out/'restart_history.csv',[{**r,**load(root/r['case']/dense.SOURCE_NAME)} for r in manifest.get('cases',[]) if r['mode']=='resumed'])
    print(json.dumps(validation,sort_keys=True));return 0 if all(validation.values()) else 1
if __name__=='__main__':raise SystemExit(main())
