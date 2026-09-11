"""Figures from saved forward tables; plotted coordinates are audited sidecars."""
from pathlib import Path
import sys,csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ART=ROOT/'artifacts/retained_controls_monotonic_forward';FIG=ART/'figures'
COLORS={'A_NATIVE':'#305d8c','P25_TRANSFER_V1_RANK1':'#157a6e','P40_TRANSFER_CALIBRATED_GEN2':'#b25c00','P55_TRANSFER_V1_RANK1':'#7645a8'}
LABELS={'A_NATIVE':'A_NATIVE','P25_TRANSFER_V1_RANK1':'P25','P40_TRANSFER_CALIBRATED_GEN2':'P40 GEN2','P55_TRANSFER_V1_RANK1':'P55 boundary'}

def main():
 FIG.mkdir(exist_ok=True);plots=[]
 def read(name):return list(csv.DictReader((ART/name).open()))
 curves=read('monotonic_first_passage_vs_temperature.csv');surface=read('barrier_temperature_surfaces.csv');cross=read('fatigue_fracture_forward_crosswalk.csv')
 def figure(name,title):
  fig,ax=plt.subplots(figsize=(10,6));ax.set_title(title);return fig,ax,dict(name=name,series=[])
 def line(ax,p,table,rows,x,y,label,**kw):
  vals=[(float(r[x]),float(r[y])) for r in rows if r.get(x) not in ('',None) and r.get(y) not in ('',None)]
  xx=[v[0] for v in vals]; yy=[v[1] for v in vals]
  ax.plot(xx,yy,label=label,**kw);p['series'].append(dict(table=table,x_column=x,y_column=y,x=xx,y=yy,label=label))
 def finish(fig,ax,p,xlabel,ylabel,log=False,legend=True):
  ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
  if log:ax.set_yscale('log')
  ax.grid(alpha=.2)
  if legend:ax.legend(fontsize=8)
  fig.text(.5,.012,'Conditional F1 reduction; full signed F2 unavailable. Saturated/collapsed values do not define a material archetype.',ha='center',fontsize=8)
  fig.tight_layout(rect=(0,.035,1,1));fig.savefig(FIG/(p['name']+'.png'),dpi=150);plt.close(fig)
  p['source_hashes']={n:hashlib.sha256((ART/n).read_bytes()).hexdigest() for n in {s['table'] for s in p['series']}}
  plots.append(p)
 def subset(cid,tier='BEST_CURRENT_MONOTONIC_FORWARD',rate=.005):return [r for r in curves if r['candidate_id']==cid and r['tier']==tier and float(r['Kdot'])==rate]
 table='monotonic_first_passage_vs_temperature.csv'
 fig,ax,p=figure('01_NEW_CANDIDATE_BARRIER_SURFACES_VS_T','Exact cleavage surfaces: inherited explicit thermal coefficients are zero')
 for cid,color in COLORS.items():
  for T,ls in [(300,'-'),(1200,'--')]:
   rr=[r for r in surface if r['candidate_id']==cid and r['barrier']=='cleavage' and float(r['temperature_K'])==T]
   line(ax,p,'barrier_temperature_surfaces.csv',rr,'stress_Pa','G',LABELS[cid]+f' {T} K',color=color,ls=ls)
 finish(fig,ax,p,'Surface stress (Pa)','Cleavage barrier (eV)')
 for name,title,col,log in [('02_ABSOLUTE_KFP_VS_T','Primary: absolute opening first-passage loads','K_FP',True),('03_NORMALIZED_KFP_VS_T','Normalized shape only: absolute scales are in Figure 02','K_over_300K',True)]:
  fig,ax,p=figure(name,title)
  for cid,color in COLORS.items():
   for rate,ls in [(.0005,':'),(.005,'-'),(.05,'--')]:
    rr=subset(cid,rate=rate)
    line(ax,p,table,rr,'temperature_K',col,LABELS[cid]+f' rate {rate:g}',color=color,ls=ls)
  finish(fig,ax,p,'Temperature (K)','KFP (MPa √m)' if col=='K_FP' else 'KFP(T) / KFP(300 K)',log)
 fig,ax,p=figure('04_INTRINSIC_VS_STATE_AWARE_KFP','Intrinsic and conditional transient-blunting predictions')
 for cid,color in COLORS.items():
  for tier,ls in [('F0_INTRINSIC_OPENING','-'),('F1_EMISSION_BLUNTING','--')]:
   rr=subset(cid,tier)
   # Keep unavailable F1 gaps; no interpolation across failed state branches.
   for r in rr:
    if not r.get('K_FP'):r=dict(r)
   good=[r for r in rr if r.get('K_FP')]
   line(ax,p,table,good,'temperature_K','K_FP',LABELS[cid]+' '+tier[:2],color=color,ls='none',marker='o' if tier.startswith('F0') else 'x',ms=3)
 finish(fig,ax,p,'Temperature (K)','KFP (MPa √m)',True)
 action=read('monotonic_cumulative_action.csv')
 fig,ax,p=figure('05_CUMULATIVE_MONOTONIC_ACTION','Integrated action on absolute load axis at 300 K')
 for cid,color in COLORS.items():
  rr=[r for r in action if r['candidate_id']==cid and float(r['temperature_K'])==300 and float(r['Kdot'])==.005 and r['tier']=='F0_INTRINSIC_OPENING' and float(r['K'])>0]
  line(ax,p,'monotonic_cumulative_action.csv',rr,'K','action_over_threshold',LABELS[cid],color=color)
 ax.set_xscale('log');finish(fig,ax,p,'Absolute K (MPa √m)','Bmono / Xi')
 fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True);p=dict(name='06_AK_AT_AND_DLN_KFP_DT',series=[])
 for ax,col in zip(axes,['AK','AT','thermal_derivative']):
  for cid,color in COLORS.items():line(ax,p,table,subset(cid),'temperature_K',col,LABELS[cid],color=color)
  ax.set_ylabel(col);ax.grid(alpha=.2)
 finish(fig,axes[-1],p,'Temperature (K)','d ln KFP / dT (1/K)')
 desc=read('barrier_derivative_descriptors.csv')
 fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True);p=dict(name='07_BARRIER_FIRST_DERIVATIVE_AND_CURVATURE',series=[])
 for ax,col in zip(axes,['D1','D2']):
  for cid,color in COLORS.items():
   rr=[r for r in desc if r['candidate_id']==cid and r['tier']=='F0_INTRINSIC_OPENING' and r['barrier']=='cleavage' and r['evaluation']=='fixed_K18_intrinsic' and float(r['Kdot'])==.005]
   line(ax,p,'barrier_derivative_descriptors.csv',rr,'temperature_K',col,LABELS[cid],color=color)
  ax.set_ylabel(col+' (eV)');ax.grid(alpha=.2)
 finish(fig,axes[-1],p,'Temperature (K)','D2 (eV)')
 fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True);p=dict(name='08_ACCESSIBILITY_AND_SATURATION',series=[])
 for ax,col in zip(axes,['zero_load_action_fraction','renewal_ceiling_ramp_fraction']):
  for cid,color in COLORS.items():line(ax,p,table,subset(cid),'temperature_K',col,LABELS[cid],color=color)
  ax.set_ylim(-.02,1.02);ax.set_ylabel(col);ax.grid(alpha=.2)
 finish(fig,axes[-1],p,'Temperature (K)','Renewal-ceiling ramp fraction')
 fig,ax,p=figure('09_FATIGUE_SLOPE_VS_PREDICTED_FRACTURE_RESPONSE','Fatigue slopes and absolute monotonic load at 300 K')
 for r in cross:
  if r['fatigue_global_slope']:line(ax,p,'fatigue_fracture_forward_crosswalk.csv',[r],'fatigue_global_slope','K_FP_300K',LABELS[r['candidate_id']],marker='o',ls='none',color=COLORS[r['candidate_id']])
 finish(fig,ax,p,'Physical fatigue global slope (P55: pilot only)','KFP at 300 K (MPa √m)',True)
 fig,ax=plt.subplots(figsize=(12,5));ax.axis('off');ax.set_title('Final forward classifications — full signed state remains unavailable')
 cells=[[LABELS[r['candidate_id']],f"{float(r['K_FP_300K']):.4g}",f"{float(r['K_FP_1200K']):.4g}",'Inadmissible over full T interval','F2 unavailable'] for r in cross]
 t=ax.table(cellText=cells,colLabels=['Row','KFP 300 K','KFP 1200 K','Available forward hierarchy','Full state'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,2)
 p=dict(name='10_FINAL_FORWARD_CLASSIFICATION',series=[],classification_rows=cross,source_hashes={'fatigue_fracture_forward_crosswalk.csv':hashlib.sha256((ART/'fatigue_fracture_forward_crosswalk.csv').read_bytes()).hexdigest()})
 fig.tight_layout();fig.savefig(FIG/(p['name']+'.png'),dpi=150);plt.close(fig);plots.append(p)
 (ART/'figure_data_manifest.json').write_text(json.dumps(plots,indent=2)+'\n')
 print('Wrote',len(plots),'figures')
if __name__=='__main__':main()
