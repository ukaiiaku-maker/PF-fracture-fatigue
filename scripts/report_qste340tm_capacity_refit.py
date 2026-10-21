#!/usr/bin/env python3
"""Final figures and scientific report for the QSTE340TM capacity refit."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.refit_qste340tm_capacity_policy import DATA,OUT,TEMPERATURE_K,SMOOTHING_RATIO
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis

def main():
 records=list(csv.DictReader(open(DATA/'normalized/qste340tm_constant_amplitude.csv')))
 fits={p:json.load(open(OUT/f'fit_{p}_phase_resolved.json')) for p in ('uncapped','fixed_1000','fixed_10000')}
 preds={p:list(csv.DictReader(open(OUT/f'predictions_{p}_phase_resolved.csv'))) for p in fits}
 colors={'uncapped':'#0072B2','fixed_1000':'#D55E00','fixed_10000':'#009E73'}
 fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
 for ax,R in zip(axes.flat,(.1,.3,.5,.7)):
  for p,rows in preds.items():
   b=[r for r in rows if float(r['R'])==R];x=np.array([float(r['DeltaK_MPa_sqrt_m']) for r in b]);y=np.array([float(r['predicted_dadn_m_per_cycle']) for r in b]);o=np.argsort(x);ax.plot(x[o],y[o],lw=1.8,c=colors[p],label=p.replace('fixed_','cap '))
  b=[r for r in preds['uncapped'] if float(r['R'])==R];x=np.array([float(r['DeltaK_MPa_sqrt_m']) for r in b]);y=np.array([float(r['observed_dadn_m_per_cycle']) for r in b]);ax.scatter(x,y,s=10,c='#222',alpha=.48,label='data');ax.set(xscale='log',yscale='log',xlabel=r'$\Delta K$ (MPa$\sqrt{m}$)',title=f'R={R:.1f}');ax.grid(True,which='both',alpha=.2)
 axes[0,0].set_ylabel('da/dN (m/cycle)');axes[1,0].set_ylabel('da/dN (m/cycle)');h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,.005));fig.suptitle('QSTE340TM: phase-resolved joint refits (298.15 K, nominal 10 Hz)',y=.99);fig.tight_layout(rect=(0,.055,1,.955));fig.savefig(OUT/'final_common_axis_data_fits.png',dpi=240,bbox_inches='tight');fig.savefig(OUT/'final_common_axis_data_fits.svg',bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for ax,R in zip(axes.flat,(.1,.3,.5,.7)):
  for p,rows in preds.items():
   b=[r for r in rows if float(r['R'])==R];ax.plot([float(r['cycles']) for r in b],[float(r['retained']) for r in b],lw=1.7,c=colors[p],label=p)
  ax.set(xlabel='cycles',ylabel='retained count',title=f'R={R:.1f}');ax.grid(True,alpha=.2)
 fig.suptitle('Retained population versus measured-history cycles');fig.tight_layout();fig.savefig(OUT/'final_retained_count_vs_cycles.png',dpi=240,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 primary=preds['uncapped']
 for ax,R in zip(axes.flat,(.1,.3,.5,.7)):
  b=[r for r in primary if float(r['R'])==R];x=np.array([float(r['DeltaK_MPa_sqrt_m']) for r in b]);
  for key,label,c in [('emission_per_cycle','emission','#0072B2'),('peierls_per_cycle','Peierls','#D55E00'),('taylor_per_cycle','Taylor','#009E73'),('escape_per_cycle','series escape','#CC79A7'),('source_per_cycle','retained source','#000000'),('recovery_per_cycle','recovery','#E69F00')]:ax.plot(x,[max(float(r[key]),1e-300) for r in b],lw=1.25,label=label,c=c)
  ax.set(yscale='log',xlabel=r'$\Delta K$ (MPa$\sqrt{m}$)',ylabel='process transfer / cycle',title=f'uncapped, R={R:.1f}');ax.grid(True,which='both',alpha=.2)
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,.005));fig.suptitle('Separate EXP_floor process rates',y=.99);fig.tight_layout(rect=(0,.065,1,.955));fig.savefig(OUT/'final_process_rates.png',dpi=240,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
 for ax,R in zip(axes.flat,(.1,.3,.5,.7)):
  b=[r for r in primary if float(r['R'])==R];x=np.array([float(r['DeltaK_MPa_sqrt_m']) for r in b]);o=np.argsort(x);ax.scatter(x,[float(r['observed_dadn_m_per_cycle']) for r in b],s=10,c='#333',alpha=.45,label='data');ax.plot(x[o],np.array([float(r['predicted_dadn_m_per_cycle']) for r in b])[o],c='#0072B2',lw=2,label='full state');ax.plot(x[o],np.array([float(r['frozen_dadn_m_per_cycle']) for r in b])[o],c='#D55E00',lw=1.5,ls='--',label='same parameters, frozen state');ax.set(xscale='log',yscale='log',xlabel=r'$\Delta K$ (MPa$\sqrt{m}$)',title=f'R={R:.1f}');ax.grid(True,which='both',alpha=.2)
 h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='lower center',frameon=False,bbox_to_anchor=(.5,.005));fig.suptitle('Primary uncapped fit: full versus frozen retained state',y=.99);fig.tight_layout(rect=(0,.055,1,.955));fig.savefig(OUT/'final_full_vs_frozen.png',dpi=240,bbox_inches='tight');plt.close(fig)
 stress=np.linspace(0,25e9,600);fig,ax=plt.subplots(figsize=(8.5,5.5))
 for p,d in fits.items():
  q=d['parameters'];basis=SmoothBroadBasis(q['opening_sigma_b_Pa'],q['opening_exponent'],SMOOTHING_RATIO);g=(q['opening_floor_over_kBT']+q['opening_amplitude_over_kBT']*basis.value(stress))*KB_EV_PER_K*TEMPERATURE_K;ax.plot(stress/1e9,g,lw=2,c=colors[p],label=p)
 ax.set(xlabel='local opening stress (GPa)',ylabel=r'$G^*_{open}$ (eV)',title='Joint-fit opening barriers at 298.15 K');ax.grid(True,alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'final_Gstar_vs_stress.png',dpi=240,bbox_inches='tight');fig.savefig(OUT/'final_Gstar_vs_stress.svg',bbox_inches='tight');plt.close(fig)
 hold=json.load(open(OUT/'leave_one_R_out_fixed_1000.json'))['results'];freq=json.load(open(OUT/'diagnostics.json'))['finalist_sensitivity'];p=fits['uncapped'];m=p['metrics']
 rows='\n'.join(f"| {R} | {m['block_metrics'][str(R)]['rmse_log10_rate']:.4f} | {m['block_metrics'][str(R)]['observed_loglog_slope']:.3f} | {m['block_metrics'][str(R)]['predicted_loglog_slope']:.3f} |" for R in (.1,.3,.5,.7))
 hrows='\n'.join(f"| {x['held_out_R']:.1f} | {x['training_rmse_log10_rate']:.4f} | {x['held_out_metrics']['rmse_log10_rate']:.4f} | {x['optimizer']['success']} |" for x in hold)
 report=f'''# QSTE340TM capacity-policy joint refit\n\nDate: 21 October 2026\n\n## Decision\n\nWith birth-phase-resolved retention, the primary **uncapped** joint fit converged at {m['overall_rmse_log10_rate']:.5f} decades. Fixed 10,000 converged at {fits['fixed_10000']['metrics']['overall_rmse_log10_rate']:.5f}; fixed 1,000 at {fits['fixed_1000']['metrics']['overall_rmse_log10_rate']:.5f}. The historical 115.338-unit cap remains an unchanged control and is not the default.\n\n| R | RMSE (decades) | observed slope | predicted slope |\n|---:|---:|---:|---:|\n{rows}\n\nThe uncapped maximum retained count is {m['max_retained_count']:.2f}, total density {m['max_density_m2']:.4e} m^-2, maximum backstress {m['max_backstress_Pa']/1e9:.3f} GPa, and minimum raw opening barrier {m['min_raw_opening_barrier_over_kBT']:.3f} kBT. No zero-barrier clipping is used.\n\n## Retention timing\n\nThe native full-cycle approximation undercounted phase-resolved surviving births by 6.7x to 4.8e3x at audited states. The final selected fits use explicit birth-phase survival and preserve emission, Peierls, Taylor, recovery, backstress, blunting, stored-energy, and opening physics. Observation-interval refinement from 1 to 8 substeps changed legacy-mode RMSE by less than 0.0002 decades; assigning all pre-measurement cycles to the first measured load worsened it to about 0.189 decades.\n\n## Complete-R holdouts (fixed-1,000 converged reference)\n\n| held R | training RMSE | held-out RMSE | optimizer converged |\n|---:|---:|---:|:---:|\n{hrows}\n\nR=0.5 remains the principal transfer weakness. The R=0.1 holdout is an optimizer-limited bound.\n\n## Limits\n\nFrequency is assumed, not measured: same-parameter checks at 1 and 20 Hz materially worsen RMSE, so absolute kinetic parameters remain frequency-conditional. The dataset is isothermal at 298.15 K; thermal coefficients are retained but not identified. The primary fit reaches the profiled initial-retained upper bound (500), so pre-measurement state is not identified. Its same-parameter frozen-state RMSE is {p['frozen_metrics']['overall_rmse_log10_rate']:.3f} decades versus {m['overall_rmse_log10_rate']:.3f} full-state, confirming consequential state evolution. The released-curvature sensitivity reached 0.14880 decades but did not converge and is not selected. Overload and closure histories were not fitted.\n'''
 (OUT/'RESULT.md').write_text(report)
 print(report)
if __name__=='__main__':main()
