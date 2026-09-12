"""Freeze source provenance and prospective gates before candidate generation."""
from pathlib import Path
import csv, hashlib, json, subprocess, sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_forward_temperature_v10230 import source_rows
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest

ART=ROOT/'artifacts/thermodynamic_joint_barrier_search'
PARENT='0399e12e8f015071b3d3d8c7ad8cb03b435d6aaf'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ART.mkdir(parents=True,exist_ok=True)
    rows=source_rows();registry=[]
    for cid,(row,manifest,audit) in rows.items():
        registry.append(dict(candidate_id=cid,role=('PRIMARY_PARENT' if cid.startswith(('P25_','P40_')) else 'BOUNDARY' if cid.startswith('P55_') else 'BASELINE'),complete_row_sha256=digest(row),material_manifest_sha256=audit['material_manifest_sha256'],complete_parameter_vector=row))
    (ART/'source_fatigue_controls_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    with (ART/'source_fatigue_controls_registry.csv').open('w',newline='') as f:
        fieldnames=['candidate_id','role','complete_row_sha256','material_manifest_sha256','complete_parameter_vector'];w=csv.DictWriter(f,fieldnames=fieldnames,lineterminator='\n');w.writeheader();w.writerows([dict(r,complete_parameter_vector=json.dumps(r['complete_parameter_vector'],sort_keys=True)) for r in registry])
    sources=['artifacts/prospective_paris_candidates/final_candidate_parameter_rows.csv','artifacts/prospective_paris_candidates/physical_developed_rates.csv','artifacts/prospective_paris_candidates/physical_local_slopes.csv','artifacts/prospective_paris_candidates/prospective_paris_candidate_decision.json','artifacts/row_renewal_monotonic_forward/renewal_contract_audit.json','artifacts/row_renewal_monotonic_forward/retained_fatigue_controls_registry.json','scripts/thermodynamic_joint_barrier_v10230.py','scripts/freeze_thermodynamic_joint_search_v10230.py']
    (ART/'source_parameter_hashes.json').write_text(json.dumps({p:sha(ROOT/p) for p in sources},indent=2,sort_keys=True)+'\n')
    renewal=json.loads((ROOT/'artifacts/row_renewal_monotonic_forward/renewal_contract_audit.json').read_text())
    result=dict(authoritative_choice='COMPLETE_ROW_RENEWAL_FOR_NEW_ANALYTICAL_SEARCH',row_hits=renewal['row_m'],row_tau_s=renewal['row_tau_s'],physical_P25_P40_P55_executed_hits=renewal['old_m'],physical_P25_P40_P55_executed_tau_s=renewal['old_tau_s'],mismatch=True,interpretation='The physical trajectories executed generic launcher values, while their immutable complete rows specify noninteger renewal fields. The prior bounded audit established no intentional cross-contract distinction. The present inverse design follows the mission-defined complete physical-row contract and does not reinterpret the existing trajectories.',generic_screen_preserved='artifacts/retained_controls_monotonic_forward',corrected_baseline='artifacts/row_renewal_monotonic_forward',physical_producer_table='artifacts/row_renewal_monotonic_forward/executed_candidate_campaign_renewal.csv',constructed_engine_audit='artifacts/row_renewal_monotonic_forward/renewal_contract_audit.json')
    (ART/'renewal_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (ART/'renewal_contract_audit.md').write_text('# Cooperative-renewal contract audit\n\nThe 42 P25/P40/P55 physical trajectory manifests record m=3 and tau=1e-6 s, while every retained complete row records m=3.2732414351776242 and tau=6.992153587194454e-7 s. The previous generic monotonic result is preserved as `GENERIC_MONOTONIC_RENEWAL_SCREEN`; the corrected baseline is preserved separately. No intentional cross-contract distinction was established. Under this mission, new candidates inherit the exact complete-row values. Existing physical trajectories are unchanged and are not retrospectively relabeled.\n')
    bounds=dict(reference_temperature_K=300,primary_parents=['P25_TRANSFER_V1_RANK1','P40_TRANSFER_CALIBRATED_GEN2'],boundary_parent='P55_TRANSFER_V1_RANK1',structured_rows_target=3072,sobol_rows_per_primary_parent=65536,sobol_seed=260911,cleavage_zero_target_eV=[1,2],guard_log10_q_low=[-8,-3],guard_exponent=[1.5,8],cleavage_entropy_zero_kB=[-50,50],cleavage_entropy_active_kB=[-40,40],cleavage_entropy_infinity_kB=[-20,20],emission_entropy_active_preferred_kB=[20,60],emission_entropy_active_control_kB=[0,20],emission_entropy_infinity_kB=[-5,15],emission_zero_entropy_admissible_kB=[-20,100],heat_capacity_kB=[-15,15],single_exp_bounds=dict(G00_eV=[1,2],sigc_GPa=[.2,8],alpha=[.02,5],exponent=[.25,12],floor_fraction=[.001,.8]))
    (ART/'search_bounds.json').write_text(json.dumps(bounds,indent=2)+'\n')
    protocol=dict(schema='thermodynamic_joint_search_protocol_v1',parent_head=PARENT,branch='codex/v10.2.30-thermodynamic-joint-barrier-search',analytical_only=True,new_physical_trajectories=0,workers_maximum=4,search_data_excludes_historical_fracture_rows=True,canonical_comparison_gating=False,thresholds=dict(primary='LN2',LN2=0.6931471805599453,UNIT_ACTION=1.0,SEED_1720=0.07509316036236147),temperatures_full_K=list(range(300,1201,25)),temperatures_coarse_K=[300,450,600,750,900,1050,1200],ramp_rates=[.0005,.005,.05],fatigue_Kmax=[12,12.75,13.5,15,16.5,18,19.5,21,24.3],fatigue_gate=dict(rms_log10_rate_error_max=.05,max_log10_rate_error_max=.10,global_slope_change_max=.15,adjacent_local_slope_change_max=.30,guard_contribution_diagnostic_eV=.005),accessibility_primary=dict(KFP_300K=[1,50],zero_load_action_fraction_max=1e-3,AK_min=0),topology_accessible_fraction_min=.8,topology=dict(ceramic_ratio_max=.70,weak_ratio_max=1.25,dbtt_transition_width_K_min=100,dbtt_ratio_min=1.5,peak_prominence_min=.15),derivative_tolerances=dict(G_eV_abs=2e-8,S_kB_abs=2e-6,V_m3_abs=1e-34,Maxwell_m3_per_K_abs=1e-34,relative=2e-5),stage_limits=dict(full_F0=5000,F1=500,deltaCp_parents=128,deltaCp_variants=32768),F2='STATE_CLOSURE_UNAVAILABLE')
    (ART/'search_protocol_freeze.json').write_text(json.dumps(protocol,indent=2)+'\n')
    definition=dict(reference_temperature_K=300,cleavage_reference='exact parent EXP-floor plus low-stress guard',emission_reference='exact A_NATIVE 300 K emission surface',G='G_r-(T-Tr)S_r+DeltaCp*((T-Tr)-T*ln(T/Tr))',S='S_r+DeltaCp*ln(T/Tr)',H='G+T*S',V='-partial G/partial sigma',Maxwell='partial V/partial T=partial S/partial sigma',entropy_units='k_B',activation_volume_units=['m^3','nm^3','angstrom^3'],attempt_frequencies_fixed=True,no_extra_entropy_prefactor=True)
    (ART/'thermodynamic_surface_definition.json').write_text(json.dumps(definition,indent=2)+'\n')
    (ART/'thermodynamic_surface_definition.md').write_text('# Thermodynamic surface definition\n\nAt `T_r=300 K`, cleavage is the exact P25 or P40 EXP-floor surface plus one low-stress guard; emission is the exact A_NATIVE surface. The extension is `G=G_r-(T-T_r)S_r+DeltaCp[(T-T_r)-T ln(T/T_r)]`, `S=S_r+DeltaCp ln(T/T_r)`, `H=G+TS`, and `V=-dG/dsigma`. Entropy and volume are derived from this single surface. Attempt frequencies remain fixed and no entropy prefactor is added.\n')
    print(json.dumps(dict(frozen=True,parent=PARENT,rows=len(registry)),indent=2))

if __name__=='__main__':main()
