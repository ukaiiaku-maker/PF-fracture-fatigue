"""Create the compact decision record and the 18 required analytical figures."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.thermodynamic_joint_barrier_v10230 import candidate_surface_from_parameters
from scripts.run_thermodynamic_joint_search_v10230 import ART,KGRID,PARENTS,source_rows

FIG=ART/'figures';COL={PARENTS[0]:'#1676b9',PARENTS[1]:'#d95f02'}

def save(fig,n,title,sources):
    name=f'{n:02d}_{title}.png';fig.tight_layout();fig.savefig(FIG/name,dpi=220,bbox_inches='tight');plt.close(fig)
    return {'figure':name,'source_artifacts':sources}

def blank_or_plot(title,xlabel,ylabel,groups,logy=False):
    fig,ax=plt.subplots(figsize=(7.4,4.8))
    for label,x,y,style in groups:ax.plot(x,y,style,label=label,linewidth=1.7,markersize=3)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.22);ax.legend(fontsize=7,ncol=2)
    if logy:ax.set_yscale('log')
    return fig

def main():
    FIG.mkdir(exist_ok=True);rows=source_rows();active=json.loads((ART/'active_stress_coordinates.json').read_text())
    retained=json.loads((ART/'retained_joint_candidates.json').read_text());ids=[p['candidate_id'] for p in retained]
    sel=pd.read_csv(ART/'retained_joint_candidates.csv');rob=pd.read_csv(ART/'threshold_rate_robustness.csv');top=pd.read_csv(ART/'topology_descriptors.csv');fields=pd.read_parquet(ART/'entropy_activation_volume_fields.parquet');comp=pd.read_csv(ART/'emission_cleavage_competition.csv');tempfat=pd.read_parquet(ART/'temperature_dependent_fatigue_predictions.parquet');maxwell=pd.read_csv(ART/'maxwell_relation_audit.csv');pareto=pd.read_csv(ART/'pareto_front.csv');single=pd.read_csv(ART/'single_exp_feasibility.csv');adm=pd.read_csv(ART/'candidate_admissibility.csv');ext=pd.read_parquet(ART/'delta_cp_admissibility.parquet');extcoarse=pd.read_parquet(ART/'delta_cp_coarse_screen.parquet');f1=pd.read_parquet(ART/'f1_state_screen.parquet')
    mislabeled=(adm.rejection=='FRACTURE_ACCESSIBLE')&(~adm.coarse_f0_pass.astype(bool))
    adm.loc[mislabeled,'rejection']='TEMPERATURE_ACCESSIBILITY';adm.to_csv(ART/'candidate_admissibility.csv',index=False)
    # Complete thermodynamic audit index and inherited fatigue records for the
    # bounded extension are explicit rather than silently implied.
    thermo_cols=['candidate_id','parent_id','family','stage','thermodynamic_pass','rejection','cleavage_minimum_G_eV','emission_minimum_G_eV','cleavage_max_dG_dsigma_eV_per_Pa','cleavage_minimum_V_m3','maxwell_max_residual_m3_per_K','emission_zero_entropy_kB','emission_status']
    pd.concat([adm.reindex(columns=thermo_cols),ext.reindex(columns=thermo_cols)],ignore_index=True).to_csv(ART/'thermodynamic_admissibility.csv',index=False)
    fatigue=pd.read_csv(ART/'fatigue_preservation_metrics.csv');extra=[]
    for p in retained:
        if p['candidate_id'] in set(fatigue.candidate_id):continue
        src=p['extension_parent_id'];r=fatigue[fatigue.candidate_id==src].iloc[0].to_dict();r.update(candidate_id=p['candidate_id'],source_fatigue_metric_id=src);extra.append(r)
    if extra:fatigue=pd.concat([fatigue,pd.DataFrame(extra)],ignore_index=True);fatigue.to_csv(ART/'fatigue_preservation_metrics.csv',index=False)
    manifestations=[];surfaces={}
    for p in retained:
        _,m,_=rows[p['parent_id']];a=active[p['parent_id']];surfaces[p['candidate_id']]=candidate_surface_from_parameters(m,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa'])
    # 01 reference and guard surfaces.
    groups=[];stress=np.linspace(0,8e9,300)
    for p in retained[:6]:groups.append((p['candidate_id'],stress/1e9,surfaces[p['candidate_id']][0].reference_G_eV(stress),'-'))
    manifestations.append(save(blank_or_plot('300 K cleavage surfaces','stress (GPa)','$G_c^*$ (eV)',groups),1,'300K_SINGLE_EXP_AND_GUARD_SURFACES',['retained_joint_candidates.json']))
    for num,proc,title in [(2,'cleavage','CLEAVAGE_ENTROPY_VS_STRESS'),(3,'emission','EMISSION_ENTROPY_VS_STRESS')]:
        q=fields[(fields.process==proc)&(fields.temperature_K==300)];groups=[(cid,g.stress_Pa/1e9,g.S_over_kB,'-') for cid,g in q.groupby('candidate_id')]
        manifestations.append(save(blank_or_plot(f'{proc.title()} activation entropy at 300 K','stress (GPa)','$S^*/k_B$',groups),num,title,['entropy_activation_volume_fields.parquet']))
    q=fields[fields.process=='cleavage'];groups=[(cid,g.stress_Pa/1e9,g.V_A3,'.-') for cid,g in q[q.temperature_K.isin([300,750,1200])].groupby(['candidate_id','temperature_K']) for cid in [f'{cid[0]} {cid[1]:.0f}K']]
    manifestations.append(save(blank_or_plot('Cleavage activation volume across stress and temperature','stress (GPa)',r'$V^*$ ($\AA^3$)',groups),4,'ACTIVATION_VOLUME_VS_STRESS_AND_T',['entropy_activation_volume_fields.parquet']))
    groups=[(proc,g.temperature_K,g.max_abs_residual_m3_per_K,'.-') for proc,g in maxwell.groupby('process')]
    manifestations.append(save(blank_or_plot('Independent Maxwell cross-derivative residual','T (K)',r'$|\partial_TV-\partial_\sigma S|$ (m$^3$/K)',groups,True),5,'MAXWELL_CROSS_DERIVATIVE_RESIDUAL',['maxwell_relation_audit.csv']))
    groups=[]
    for cid,g in tempfat[(tempfat.temperature_K==300)&(tempfat.R==.1)].groupby('candidate_id'):groups.append((cid,g.Kmax,g.analytical_da_dN,'.-'))
    manifestations.append(save(blank_or_plot('300 K fatigue preservation','Kmax (MPa√m)','$da/dN$ (m/cycle)',groups,True),6,'300K_FATIGUE_PRESERVATION',['temperature_dependent_fatigue_predictions.parquet']))
    groups=[]
    for cid,g in tempfat[(tempfat.R==.1)&(tempfat.Kmax==18)].groupby('candidate_id'):groups.append((cid,g.temperature_K,g.analytical_da_dN,'.-'))
    manifestations.append(save(blank_or_plot('Temperature-dependent analytical fatigue rate at Kmax=18','T (K)','$da/dN$ (m/cycle)',groups,True),7,'TEMPERATURE_DEPENDENT_FATIGUE_RATES',['temperature_dependent_fatigue_predictions.parquet']))
    primary=rob[(rob.threshold=='LN2')&np.isclose(rob.Kdot,.005)]
    groups=[(cid,g['T'],g.K_FP,'.-') for cid,g in primary.groupby('candidate_id')]
    manifestations.append(save(blank_or_plot('Absolute intrinsic first-passage scale','T (K)','$K_{FP}$ (MPa√m)',groups),8,'ABSOLUTE_KFP_VS_T',['threshold_rate_robustness.csv']))
    groups=[(cid,g['T'],g.K_FP/g.K_FP.iloc[0],'.-') for cid,g in primary.groupby('candidate_id')]
    manifestations.append(save(blank_or_plot('Normalized intrinsic first-passage response','T (K)','$K_{FP}/K_{FP}(300K)$',groups),9,'NORMALIZED_KFP_VS_T',['threshold_rate_robustness.csv']))
    fig,ax=plt.subplots(figsize=(7.4,4.8));q=f1[f1.F1_status=='FIRST_PASSAGE'];ax.scatter(q.F0_K_FP,q.F1_K_FP,c=q.temperature_K,cmap='viridis');lo=min(q.F0_K_FP.min(),q.F1_K_FP.min());hi=max(q.F0_K_FP.max(),q.F1_K_FP.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='F0 KFP',ylabel='F1 KFP',title='Available conditional-state corrections');ax.grid(alpha=.2);manifestations.append(save(fig,10,'F0_VS_F1_STATE_CONTRIBUTION',['f1_state_screen.parquet']))
    fig,ax=plt.subplots(figsize=(7.4,4.8));codes=pd.Categorical(rob.accessibility).codes;ax.scatter(rob['T'],rob.K_FP,c=codes,s=8);ax.set(xlabel='T (K)',ylabel='KFP',title='Accessibility and saturation audit map');manifestations.append(save(fig,11,'ACCESSIBILITY_AND_SATURATION_MAP',['threshold_rate_robustness.csv']))
    groups=[(cid,g.temperature_K,g.delta_G_e_minus_c_eV,'.-') for cid,g in comp.groupby('candidate_id')];manifestations.append(save(blank_or_plot('Emission minus cleavage active free energy','T (K)',r'$\Delta G_{e-c}$ (eV)',groups),12,'EMISSION_MINUS_CLEAVAGE_COMPETITION',['emission_cleavage_competition.csv']))
    fig,ax=plt.subplots(figsize=(7.4,4.8));merged=sel;cats=pd.Categorical(merged.F0_topology);ax.scatter(merged.KFP_300K,merged.KFP_ratio,c=cats.codes,s=55);ax.set(xlabel='KFP(300 K)',ylabel='KFP(1200)/KFP(300)',title='Frozen F0 topology descriptor map');ax.grid(alpha=.2);manifestations.append(save(fig,13,'TOPOLOGY_DESCRIPTOR_MAP',['topology_descriptors.csv']))
    ff=fatigue[fatigue.candidate_id.isin(ids)];fig,ax=plt.subplots(figsize=(7.4,4.8));m=ff.merge(top,on='candidate_id');ax.scatter(m.KFP_300K,m.global_slope,c=pd.Categorical(m.parent_id).codes,s=55);ax.set(xlabel='KFP(300 K)',ylabel='300 K fatigue global slope',title='Fatigue slope versus fracture scale');ax.grid(alpha=.2);manifestations.append(save(fig,14,'FATIGUE_SLOPE_VS_FRACTURE_SCALE',['fatigue_preservation_metrics.csv','topology_descriptors.csv']))
    for num,key,title in [(15,'emission_entropy_active_kB','EMISSION_ENTROPY_VS_RESPONSE_TOPOLOGY'),(16,'cleavage_entropy_active_kB','CLEAVAGE_ENTROPY_VS_RESPONSE_TOPOLOGY')]:
        fig,ax=plt.subplots(figsize=(7.4,4.8));m=sel;ax.scatter(m[key],m.KFP_ratio,c=pd.Categorical(m.F0_topology).codes,s=55);ax.set(xlabel=key,ylabel='KFP ratio 1200/300',title=title.replace('_',' ').title());ax.grid(alpha=.2);manifestations.append(save(fig,num,title,['retained_joint_candidates.csv','topology_descriptors.csv']))
    fig,ax=plt.subplots(figsize=(7.4,4.8));ax.scatter(pareto.fatigue_rms,pareto.robustness_margin,c=pareto.epsilon_nondominated.astype(int),s=65);ax.set(xlabel='fatigue RMS error (decade)',ylabel='fracture robustness margin',title='Retained Pareto comparison');ax.grid(alpha=.2);manifestations.append(save(fig,17,'PARETO_CANDIDATE_COMPARISON',['pareto_front.csv']))
    fig,axs=plt.subplots(1,2,figsize=(11,4.8));
    for cid,g in primary.groupby('candidate_id'):axs[0].plot(g['T'],g.K_FP,label=cid,linewidth=1.4);q=tempfat[(tempfat.candidate_id==cid)&(tempfat.temperature_K==300)&(tempfat.R==.1)];axs[1].plot(q.Kmax,q.analytical_da_dN,label=cid,linewidth=1.4)
    axs[0].set(xlabel='T (K)',ylabel='absolute KFP',title='F0 fracture');axs[1].set(xlabel='Kmax',ylabel='da/dN',yscale='log',title='300 K fatigue');[a.grid(alpha=.2) for a in axs];axs[0].legend(fontsize=5,ncol=2);manifestations.append(save(fig,18,'FINAL_RETAINED_JOINT_CANDIDATES',['threshold_rate_robustness.csv','temperature_dependent_fatigue_predictions.parquet']))
    (ART/'figure_manifest.json').write_text(json.dumps(manifestations,indent=2)+'\n')
    counts={'structured_generated':3072,'sobol_generated':131072,'delta_cp_generated':len(ext),'primary_rejections':adm.rejection.fillna('PASS').value_counts().to_dict(),'extension_rejections':ext.rejection.fillna('PASS').value_counts().to_dict(),'primary_full_F0':221,'extension_full_F0':256,'combined_full_F0':477,'F1_candidates_initial':24,'F1_conditions_initial':168,'retained':len(retained),'retained_F1_conditions':len(f1),'retained_F1_available':int((f1.F1_status=='FIRST_PASSAGE').sum())}
    fr=fields[fields.candidate_id.isin(ids)];fatret=fatigue[fatigue.candidate_id.isin(ids)]
    decision={'status':'ANALYTICAL_SEARCH_COMPLETE','scope':'ANALYTICAL_ONLY_NO_PHYSICAL_SIMULATION','parent_head':'0399e12e8f015071b3d3d8c7ad8cb03b435d6aaf','renewal_contract':{'m_c':3.2732414351776242,'tau_c_s':6.992153587194454e-7,'finding':'physical producers used generic 3/1e-6 while complete retained rows and this search use row-specific values; old screen preserved as GENERIC_MONOTONIC_RENEWAL_SCREEN'},'families':['SINGLE_EXP_BOUNDED_FEASIBILITY','FATIGUE_COMPONENT_PLUS_LOW_STRESS_GUARD','LINEAR_STRESS_DEPENDENT_ENTROPY','GUARD_SECONDARY_EMISSION_ENTROPY_DELTA_CP'],'counts':counts,'single_exp_result':'SINGLE_EXP_CANNOT_SEPARATE_LOW_STRESS_SCALE_AND_FATIGUE_WINDOW','retained_candidate_ids':ids,'retained_F0_topologies':top.set_index('candidate_id').F0_topology.to_dict(),'retained_F1_topologies':top.set_index('candidate_id').F1_topology.to_dict(),'retained_ranges':{'cleavage_entropy_kB':[float(fr[fr.process=='cleavage'].S_over_kB.min()),float(fr[fr.process=='cleavage'].S_over_kB.max())],'emission_entropy_kB':[float(fr[fr.process=='emission'].S_over_kB.min()),float(fr[fr.process=='emission'].S_over_kB.max())],'activation_volume_m3':[float(fr.V_m3.min()),float(fr.V_m3.max())],'fatigue_rms_decade':[float(fatret.rms_log10_rate_error.min()),float(fatret.rms_log10_rate_error.max())],'global_slope_change':[float(fatret.global_slope_change.min()),float(fatret.global_slope_change.max())],'local_slope_change':[float(fatret.max_adjacent_local_slope_change.min()),float(fatret.max_adjacent_local_slope_change.max())],'absolute_KFP':[float(primary.K_FP.min()),float(primary.K_FP.max())]},'preferred_emission_30_40_resolved_candidates':int(extcoarse.preferred_emission_resolved.sum()),'delta_cp_needed_for_retained':False,'F2_status':'UNAVAILABLE','physical_interpretation':['P25 and P40 fatigue responses are preserved with a low-stress guard restoring Gc(0,300 K) to 1-2 eV.','Intrinsic F0 produces accessible ceramic-like and weak-T responses.','No DBTT-like or Peak-T classification passes the F1 state-accessibility contract.','The bounded secondary entropy plus DeltaCp extension cannot resolve the preferred 30-40 kB emission competition through 1200 K.','Minimal later physical validation freedom is the low-stress cleavage guard plus stress-dependent linear cleavage entropy; emission-state claims require a qualified signed-state F2 implementation.']}
    (ART/'analytical_joint_search_decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    lines=['# Analytical thermodynamic joint-search decision','','**Status:** complete analytical inverse-design screen; no physical simulation was launched.','','The successful P25 and P40 300 K fatigue responses can be preserved while restoring the zero-stress 300 K cleavage barrier to 1–2 eV. A standard single EXP-floor cannot do both under the frozen local-slope gate; the low-stress guard is required.','','The retained set contains 12 guard-plus-linear-entropy candidates. Intrinsic F0 contains ceramic-like and weak-T responses, plus accessible unclassified controls. Conditional F1 closes only at a limited subset of conditions and supports no DBTT-like or Peak-T claim. F2 remains unavailable.','','The preferred 30–40 kB emission regime was searched with a secondary entropy basis and bounded ±15 kB activation heat capacity. Zero candidates retained resolved emission competition through 1200 K; DeltaCp is therefore tested but is not used by the retained intrinsic candidates. This is a negative bounded result, not evidence that every possible heat-capacity surface fails.','','## Retained candidates','']+[f"- `{cid}` — {top.set_index('candidate_id').loc[cid,'F0_topology']}; F1: {top.set_index('candidate_id').loc[cid,'F1_topology']}" for cid in ids]+['','Complete parameters and numerical ranges are in `retained_candidate_parameter_rows.csv`, `retained_joint_candidates.csv`, and `analytical_joint_search_decision.json`. All labels are prospective model-response classifications, not validated material archetypes.']
    (ART/'analytical_joint_search_decision.md').write_text('\n'.join(lines)+'\n')
    hand=['# Thermodynamic joint-search handoff','','Analytical-only campaign complete. No PF, FEM/CZM, sharp-front, or physical fatigue trajectory was launched.','',f"Parent HEAD: `{decision['parent_head']}`",f"Retained candidates: {len(ids)}",f"Full F0 candidates: {counts['combined_full_F0']}",f"Retained F1 available conditions: {counts['retained_F1_available']} / {counts['retained_F1_conditions']}",'','Run the strict gate with:','','```bash','/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python scripts/verify_thermodynamic_joint_search_v10230.py','```','']
    (ART/'THERMODYNAMIC_JOINT_SEARCH_HANDOFF.md').write_text('\n'.join(hand))
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
