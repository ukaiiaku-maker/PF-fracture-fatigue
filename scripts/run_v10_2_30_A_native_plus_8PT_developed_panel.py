#!/usr/bin/env python3
"""Registry-driven A-native+8PT developed-fatigue panel runner/analyzer."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

LOADS=(10.8,13.5,16.2,21.6)
HEAD=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
BRANCH="codex/v10.2.30-A-native-TP-panel"
PY="/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
FAMILY="/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"

def atomic_csv(rows,path):
    tmp=path.with_suffix(path.suffix+'.tmp'); pd.DataFrame(rows).to_csv(tmp,index=False); os.replace(tmp,path)

def make_registry(root):
    options=pd.read_csv(root/'A_native_plus_8PT_registry.csv').option_key.tolist(); rows=[]
    for o in options:
      for dk in LOADS:
        out=root/'developed'/'n80'/o/f'DK_{dk:g}'
        rows.append({'composite_id':f'{o}__DK_{dk:g}__R0p1__n80__seed1720','parameter_option':o,'deltaK_MPa_sqrt_m':dk,
          'R':.1,'n_bins':80,'seed':1720,'status':'PENDING','physical_censor':False,'watchdog_nonterminal':False,
          'result_path':str(out.resolve()),'solver_head':HEAD,'material_hash':'from_registry_manifest','common_physics_hash':'from_registry_manifest',
          'exit_code':None,'wall_seconds':None})
    atomic_csv(rows,root/'A_native_plus_8PT_developed_job_registry.csv'); return rows

def run_one(row,root,registry_path):
    out=Path(row["result_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update({'PYTHON_BIN':PY,'CONDA_ENV':'arrhenius-sharp-front-v10-codex','CONDA_DEFAULT_ENV':'arrhenius-sharp-front-v10-codex',
      'EXPECTED_BRANCH':BRANCH,'EXPECTED_HEAD':HEAD,'FAMILY_JSON':FAMILY,
      'V10230_ENTRY_MODULE':'arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK',
      'V10230_CANDIDATE_REGISTRY':str((root/'A_native_plus_8PT_registry.csv').resolve()),
      'V10230_CANDIDATE_SELECTION':str((root/'A_native_plus_8PT_selection.json').resolve()),
      'TARGET_FRACTION':'developed_panel','TARGET_EXT_UM':'100','CYCLES_MAX':'1000000000000','HAZARD_SEED':'1720','MAX_WALL_SECONDS':'43200',
      'PARAMETER_OPTION':row['parameter_option'],'TARGET_DELTAK':str(row['deltaK_MPa_sqrt_m']),'R_RATIO':'0.1',
      'RUN_LABEL':row['composite_id'],'OUTROOT':str(out)})
    t=time.time(); p=subprocess.run(['bash','scripts/run_v10_2_30_weakt_high_cycle_1e12.sh'],cwd=Path.cwd(),env=env)
    row=dict(row); row['exit_code']=p.returncode; row['wall_seconds']=time.time()-t
    result=out/'developed_fatigue_growth_summary.json'; checkpoint=out/'high_cycle_live_checkpoint.json'
    if p.returncode==0 and result.exists():
      d=json.load(open(result)); row['status']='COMPLETE' if d.get('target_reached') else ('PHYSICAL_CENSOR' if d.get('status')=='cycle_censor' else 'COMPLETE_PARTIAL_GROWTH')
      row['physical_censor']=row['status']=='PHYSICAL_CENSOR'
    elif checkpoint.exists(): row['status']='WALL_LIMIT_NONTERMINAL'; row['watchdog_nonterminal']=True
    elif p.returncode == 2 and not checkpoint.exists(): row["status"]="LAUNCH_PREFLIGHT_FAILURE"
    else: row["status"]="NUMERICAL_FAILURE"
    return row

def run_panel(root,workers):
    path=root/'A_native_plus_8PT_developed_job_registry.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else make_registry(root)
    pending=[r for r in rows if r['status']=='PENDING']; byid={r['composite_id']:r for r in rows}
    with ThreadPoolExecutor(max_workers=workers) as ex:
      futures={ex.submit(run_one,r,root,path):r for r in pending}
      for f in as_completed(futures):
        r=f.result(); byid[r['composite_id']]=r; atomic_csv(list(byid.values()),path)
        print(json.dumps({'completed':r['composite_id'],'status':r['status'],'exit_code':r['exit_code']}),flush=True)

def analyze(root):
    jobs=pd.read_csv(root/'A_native_plus_8PT_developed_job_registry.csv'); points=[]
    for _,j in jobs.iterrows():
      p=Path(j.result_path)/'developed_fatigue_growth_summary.json'
      if not p.exists(): continue
      d=json.load(open(p)); dev=d.get('developed_interval') or {}
      points.append({'parameter_option':j.parameter_option,'deltaK_MPa_sqrt_m':j.deltaK_MPa_sqrt_m,'R':j.R,'n_bins':j.n_bins,'seed':j.seed,
        'status':j.status,'target_reached':d.get('target_reached'),'stable_growth':d.get('stable_growth_provisional'),
        'event_count':d.get('event_count'),'final_extension_um':d.get('final_projected_extension_um'),'developed_da_dN_m_per_cycle':dev.get('da_dN'),
        'developed_event_count':dev.get('event_count'),'censor_or_failure_reason':d.get('censor_or_failure_reason'),'result_path':j.result_path})
    pts=pd.DataFrame(points); pts.to_csv(root/'A_native_plus_8PT_developed_fatigue_points.csv',index=False)
    fits=[]; slopes=[]
    for opt,g in pts.groupby('parameter_option'):
      a=g[(g.developed_da_dN_m_per_cycle>0)&g.developed_da_dN_m_per_cycle.notna()].sort_values('deltaK_MPa_sqrt_m')
      for (_,x),(_,y) in zip(a.iloc[:-1].iterrows(),a.iloc[1:].iterrows()):
        m=np.log(y.developed_da_dN_m_per_cycle/x.developed_da_dN_m_per_cycle)/np.log(y.deltaK_MPa_sqrt_m/x.deltaK_MPa_sqrt_m)
        slopes.append({'parameter_option':opt,'deltaK_low':x.deltaK_MPa_sqrt_m,'deltaK_high':y.deltaK_MPa_sqrt_m,'local_m':m})
      if len(a)>=3:
        coef=np.polyfit(np.log(a.deltaK_MPa_sqrt_m),np.log(a.developed_da_dN_m_per_cycle),1); pred=np.polyval(coef,np.log(a.deltaK_MPa_sqrt_m)); yy=np.log(a.developed_da_dN_m_per_cycle)
        r2=1-float(np.sum((yy-pred)**2))/max(float(np.sum((yy-yy.mean())**2)),1e-300)
        ls=[x['local_m'] for x in slopes if x['parameter_option']==opt]
        fits.append({'parameter_option':opt,'m':coef[0],'C':np.exp(coef[1]),'R2':r2,'admissible_points':len(a),'rate_span':a.developed_da_dN_m_per_cycle.max()/a.developed_da_dN_m_per_cycle.min(),
          'curvature_local_slope_range':max(ls)-min(ls) if ls else None,'adaptive_loads_required':len(a)<3})
      else: fits.append({'parameter_option':opt,'m':None,'C':None,'R2':None,'admissible_points':len(a),'rate_span':None,'curvature_local_slope_range':None,'adaptive_loads_required':True})
    pd.DataFrame(fits).to_csv(root/'A_native_plus_8PT_paris_window_analysis.csv',index=False)
    pd.DataFrame(slopes).to_csv(root/'A_native_plus_8PT_local_slopes.csv',index=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['init','run','analyze']);p.add_argument('--root',type=Path,default=Path('runs/A_native_plus_8PT_fatigue_v1'));p.add_argument('--workers',type=int,default=3);a=p.parse_args();root=a.root.resolve()
    if a.command=='init': make_registry(root)
    elif a.command=='run':
      if not (root/'A_native_plus_8PT_developed_job_registry.csv').exists(): make_registry(root)
      run_panel(root,a.workers)
    else: analyze(root)
if __name__=='__main__': main()
