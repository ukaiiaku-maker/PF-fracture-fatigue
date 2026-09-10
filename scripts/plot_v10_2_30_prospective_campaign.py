"""Focused campaign figures; targets, frozen predictions and qualified physics."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_v10_2_30_prospective_campaign import read,slopes,target_rate
from arrhenius_fracture.prospective_paris_transfer_engine_v10230 import build_transfer_manifest
ART=ROOT/'artifacts/prospective_paris_candidates';WORK=ROOT/'runs/prospective_paris_transfer_v1/analysis_work'
COLORS={'P25':'#16786d','P40':'#ae5f10','P55':'#754ba0'}


def main():
    frozen=read(ART/'transfer_candidate_freeze_v1.json');allrows=read(WORK/'harvested_results.json');out=WORK/'figures';out.mkdir(exist_ok=True)
    rows=[r for r in allrows if r['physical_rate'] is not None]
    primary=[r for r in rows if r['R']==.1 and r['seed']==1720]
    def save(fig,name):fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
    def base():return plt.subplots(figsize=(6.6,4.4),constrained_layout=True)
    for name,ylabel,local in [('target_prediction_physical_rates.png','Developed da/dN (m/cycle)',False),('target_prediction_physical_local_slopes.png','Adjacent local slope',True)]:
        fig,ax=base()
        for c in frozen['candidates']:
            target=c['target'];cid=c['candidate_id'];color=COLORS[target]
            expected=sorted([j for j in frozen['jobs'] if j['candidate_id']==cid and j['R']==.1 and j['seed']==1720],key=lambda j:j['Kmax'])
            physical=sorted([r for r in primary if r['candidate_id']==cid],key=lambda r:r['Kmax'])
            ks=np.array([j['Kmax'] for j in expected]);tar=np.array([target_rate(target,k) for k in ks]);pred=np.array([j['predicted_rate'] for j in expected])
            if local:
                xm=np.sqrt(ks[:-1]*ks[1:]);ax.plot(xm,np.diff(np.log(tar))/np.diff(np.log(ks)),':',color=color,label=target+' target')
                ax.plot(xm,np.diff(np.log(pred))/np.diff(np.log(ks)),'--',color=color,label=target+' calibrated prediction')
                measured=slopes(physical);ax.plot([np.sqrt(r['Klo']*r['Khi']) for r in measured],[r['physical_slope'] for r in measured],'o-',color=color,label=target+' physical'+(' GEN2' if target=='P40' else ''))
            else:
                ax.plot(ks,tar,':',color=color,label=target+' target');ax.plot(ks,pred,'--',color=color,label=target+' calibrated prediction')
                ax.plot([r['Kmax'] for r in physical],[r['physical_rate'] for r in physical],'o-',color=color,label=target+' physical'+(' GEN2' if target=='P40' else ''))
        ax.set(xlabel='Kmax (MPa √m)',ylabel=ylabel);ax.legend(fontsize=7,ncol=2)
        if not local:ax.set_yscale('log')
        fig.text(.5,-.02,'P40 GEN2: Kmax 13.5, 18 and 21 are calibration loads.',ha='center',fontsize=7)
        save(fig,name)
    fig,ax=base();sigma=np.linspace(0,15e9,300)
    for c in frozen['candidates']:
        m,_=build_transfer_manifest(c['candidate_id']);ax.plot(sigma/1e9,m.cleavage.values_eV(sigma,300),color=COLORS[c['target']],label=c['candidate_id'])
    ax.set(xlabel='Cleavage stress (GPa)',ylabel='Barrier (eV)',title='Frozen single-EXP barriers, 300 K');ax.legend(fontsize=7);save(fig,'barrier_profiles.png')
    fig,ax=base()
    for t,color in COLORS.items():
        selected=sorted([r for r in primary if r['target']==t],key=lambda r:r['Kmax']);ax.plot([r['Kmax'] for r in selected],[r['prediction_residual_decade'] for r in selected],'o-',color=color,label=t)
    ax.axhline(.2,color='gray',ls=':');ax.axhline(-.2,color='gray',ls=':');ax.set(xlabel='Kmax (MPa √m)',ylabel='log10(physical / frozen prediction)');ax.legend();save(fig,'prediction_residuals.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for t,color in COLORS.items():
        selected=sorted([r for r in primary if r['target']==t],key=lambda r:r['Kmax']);ks=[r['Kmax'] for r in selected]
        axes[0].plot(ks,[r['terminal_radius_m']*1e6 for r in selected],'o-',color=color,label=t)
        axes[1].plot(ks,[r['max_recorded_stress_Pa']/(r['Kmax']*1e6/np.sqrt(2*np.pi*1e-6)) for r in selected],'o-',color=color)
    axes[0].set(xlabel='Kmax (MPa √m)',ylabel='Terminal r_eff (µm)');axes[0].legend();axes[1].set(xlabel='Kmax (MPa √m)',ylabel='Recorded peak stress / nominal r0 stress');save(fig,'state_stress_transmission.png')
    for name,selector in [('seed_transfer.png','seed'),('R_transfer.png','R')]:
        fig,ax=base();anydata=False
        values=[1720,1001723] if selector=='seed' else [-.95,.1,.5]
        for t,color in COLORS.items():
            for value,style in zip(values,['o-','s--','^:']):
                selected=sorted([r for r in rows if r['target']==t and r[selector]==value and r['Kmax'] in (15,18,21) and (r['R']==.1 if selector=='seed' else r['seed']==1720)],key=lambda r:r['Kmax'])
                if selected:anydata=True;ax.plot([r['Kmax'] for r in selected],[r['physical_rate'] for r in selected],style,color=color,label=f'{t} {selector}={value}')
        ax.set(xlabel='Kmax (MPa √m)',ylabel='Developed da/dN (m/cycle)');ax.set_yscale('log')
        if anydata:ax.legend(fontsize=7,ncol=2)
        else:ax.text(.5,.5,'No qualified transfer trajectories',ha='center',transform=ax.transAxes)
        save(fig,name)
    fig,ax=base()
    for t,color in COLORS.items():
        selected=sorted([r for r in primary if r['target']==t],key=lambda r:r['Kmax']);ax.plot([r['Kmax'] for r in selected],[r['physical_rate'] for r in selected],'o-',color=color,label=t+(' GEN2' if t=='P40' else ''))
    ax.set(xlabel='Kmax (MPa √m)',ylabel='Physical developed da/dN (m/cycle)');ax.set_yscale('log');ax.legend();save(fig,'P25_P40_P55_comparison.png')
    mono=list(csv.DictReader((WORK/'monotonic_side_effect_check_all_frozen.csv').open()));fig,ax=base()
    for cid in dict.fromkeys(r['candidate_id'] for r in mono):
        selected=[r for r in mono if r['candidate_id']==cid];ax.plot([float(r['temperature_K']) for r in selected],[float(r['K_first_MPa_sqrt_m']) for r in selected],'o-',label=cid)
    ax.set(xlabel='Temperature (K)',ylabel='No-feedback first-passage K (MPa √m)',title='Reduced monotonic side check — analysis only');ax.set_yscale('log');ax.legend(fontsize=7);save(fig,'monotonic_side_effects.png')
    original=list(csv.DictReader((ART/'p40_pilot_prediction_comparison.csv').open()))
    original_local=list(csv.DictReader((ART/'p40_pilot_local_slopes.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for field,label,style in [('target_rate','Frozen target',':'),('predicted_rate','Pre-pilot sensitivity prediction','--'),('physical_rate','Physical original P40','o-')]:
        axes[0].plot([float(r['Kmax']) for r in original],[float(r[field]) for r in original],style,label=label)
    for field,label,style in [('target_slope','Frozen target',':'),('predicted_slope','Pre-pilot sensitivity prediction','--'),('physical_slope','Physical original P40','o-')]:
        axes[1].plot([np.sqrt(float(r['Klo'])*float(r['Khi'])) for r in original_local],[float(r[field]) for r in original_local],style,label=label)
    axes[0].set(xlabel='Kmax (MPa √m)',ylabel='Developed da/dN (m/cycle)',yscale='log');axes[1].set(xlabel='Interval geometric mean Kmax (MPa √m)',ylabel='Adjacent local slope')
    axes[0].legend(fontsize=7);fig.suptitle('Original P40: immutable pre-pilot prediction versus physical pilots')
    save(fig,'original_P40_sensitivity_pilot_comparison.png')
    print(json.dumps({'figures':[str(p) for p in sorted(out.glob('*.png'))]},indent=2))


if __name__=='__main__':main()
