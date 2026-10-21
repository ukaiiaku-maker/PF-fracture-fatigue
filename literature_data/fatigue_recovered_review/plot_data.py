#!/usr/bin/env python3
"""Regenerate data-only plots. No statistical or constitutive fit is performed."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
def read(name):
    with (ROOT/'normalized'/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def finish(fig, ax, name):
    ax.tick_params(labelsize=11);ax.grid(False,which='both');ax.legend(frameon=False,fontsize=10)
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=220);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
growth=read('qste340tm_growth_rates.csv')
fig,ax=plt.subplots(figsize=(8.2,5.6))
for R in [.1,.3,.5,.7]:
    a=[r for r in growth if r['protocol']=='CONSTANT_AMPLITUDE' and float(r['R'])==R]
    ax.loglog([float(r['reported_DeltaK_MPa_sqrt_m']) for r in a],[float(r['dadn_m_per_cycle']) for r in a],
              'o',markersize=3.2,label=f'R = {R:g}')
ax.set_xlabel(r'Reported $\Delta K$ (MPa$\sqrt{\mathrm{m}}$)',fontsize=13)
ax.set_ylabel(r'Reported $da/dN$ (m/cycle)',fontsize=13)
finish(fig,ax,'qste340tm_constant_amplitude')
fig,ax=plt.subplots(figsize=(8.2,5.6))
for proto,ol,label in [('CONSTANT_AMPLITUDE',None,'Constant amplitude'),('SINGLE_OVERLOAD',1.5,'Overload ratio 1.5'),('SINGLE_OVERLOAD',2.,'Overload ratio 2.0')]:
    a=[r for r in growth if float(r['R'])==.1 and r['protocol']==proto and (ol is None or float(r['overload_ratio_label'])==ol)]
    ax.loglog([float(r['reported_DeltaK_MPa_sqrt_m']) for r in a],[float(r['dadn_m_per_cycle']) for r in a],
              'o',markersize=3.2,label=label)
ax.set_xlabel(r'Reported $\Delta K$ (MPa$\sqrt{\mathrm{m}}$)',fontsize=13)
ax.set_ylabel(r'Reported $da/dN$ (m/cycle)',fontsize=13)
finish(fig,ax,'qste340tm_R01_overload_comparison')
aN=read('steel316ln_crack_histories.csv')
fig,ax=plt.subplots(figsize=(8.2,5.6))
for spec in ['A15','B3','B7','B42']:
    a=[r for r in aN if r['specimen_id']==spec]
    ax.plot([float(r['cycles']) for r in a],[float(r['reported_crack_length_mm']) for r in a],
            '--' if spec=='B42' else '-',linewidth=1.7,
            label=spec+' (author-labeled outlier)' if spec=='B42' else spec)
ax.set_xlabel('Elapsed cycles',fontsize=13);ax.set_ylabel('Reported crack length (mm)',fontsize=13)
finish(fig,ax,'steel316ln_unirradiated_histories')
