"""Terminal-only independent P40 pilot analysis (development scratch).

No partial trajectory is accepted, and no censored rate is made finite.
Event sums use only events wholly after the first 20 micrometres.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
ART = ROOT / 'artifacts/prospective_paris_candidates'
RUN = ROOT / 'runs/prospective_paris_p40_pilot_v1'


def read(path):
    return json.loads(path.read_text())


def interval(events):
    da = sum(e['projected_advance_m'] for e in events)
    dn = sum(e['cycles_between_events'] for e in events)
    assert da > 0 and dn > 0
    return dict(event_count=len(events), da_m=da, dN=dn, rate=da/dn,
                mean_length_m=da/len(events), mean_wait_cycles=dn/len(events))


def analyze(job, attempt=2):
    p = Path(job['result_path'] + f'__attempt{attempt}')
    manifest = read(p/'high_cycle_run_manifest.json')
    launch = read(p.parent/f'{p.name}__launch.json')
    args = read(p/'run_args.json')
    summary = read(p/'developed_fatigue_growth_summary.json')
    assert int((p/'exit_code.txt').read_text()) == 0
    assert manifest['git_head'] == launch['launch_head']
    assert launch['result_path_virgin_at_launch'] and launch['resume'] is False
    assert launch['launch_time_unix'] <= (p/'high_cycle_run_manifest.json').stat().st_birthtime
    assert args['temperatures'] == [300.0] and args['R'] == job['R']
    assert args['frequency_Hz'] == 1000 and args['mpz_n_bins'] == 80
    assert args['cycles_max'] == 1e12 and args['crack_backend'] == 'sharp_wake'
    assert args['da_phys'] == 5e-6 and args['target_crack_extension_um'] == 100
    assert manifest['hazard_seed'] == job['seed']
    audit = manifest['prospective_candidate']
    assert audit['candidate_row_sha256'] == 'eb1373bd5a124b296e8f12dcc2a2c07fd0bb9af42d1dd0f28f36179d2eda8281'
    actual = MaterialManifest.from_csv(p/'selected_material_manifest_v10_2_22.csv')
    assert digest(actual.as_dict()) == audit['material_manifest_sha256']
    control = read(p/'v10_2_30_fixed_deltaK_control.json')
    assert math.isclose(control['target_Kmax_MPa_sqrt_m'],job['Kmax_MPa_sqrt_m'],rel_tol=1e-12)
    assert summary['target_reached'], 'physical censor must be reported separately without a finite rate'
    kinetic = read(p/'kinetic_tip_cell_audit_v101.json')['records']
    gates = read(p/'hazard_energy_gated_events_v10_2_30.json')
    assert gates and all(not e['athermal_Gc_used'] and not e['paris_law_used'] for e in gates)
    assert all(not e['independent_toughness_floor_used'] for e in gates)
    events = summary['event_measurements']
    selected = [e for e in events if e['projected_extension_pre_m'] >= 20e-6]
    stats = interval(selected)
    final = events[-1]['projected_extension_post_m']
    start = max(final-50e-6,20e-6); mid=(start+final)/2
    early = interval([e for e in selected if e['projected_extension_post_m']>start and e['projected_extension_pre_m']<mid])
    late = interval([e for e in selected if e['projected_extension_post_m']>mid])
    ratio = late['rate']/early['rate']
    qualified = stats['event_count']>=10 and stats['da_m']>=50e-6 and .5<=ratio<=2
    last = kinetic[-1]
    result = dict(candidate_id=job['parameter_option'],Kmax=job['Kmax_MPa_sqrt_m'],R=job['R'],seed=job['seed'],
        result_path=str(p),producer_head=manifest['git_head'],developed_qualified=qualified,
        late_early_ratio=ratio,physical_rate=stats['rate'],predicted_rate=job['predicted_da_dN_RADIUS_SCALED_B1'],
        target_rate=job['target_da_dN'],terminal_radius_m=last['persistent_tip_radius_m'],
        mobile_count=last['state_mobile_count'],retained_count=last['state_retained_count'],
        K_shield=last['state_active_K_shield_signed_Pa_sqrt_m'],sigma_back=last['persistent_sigma_back_Pa'],
        **stats)
    result['repository_boundary_overlap_rate'] = summary['developed_interval']['da_dN']
    result['development_boundary_rule'] = 'whole_events_with_pre_extension_at_least_20um'
    result['source_row_sha256'] = audit['candidate_row_sha256']
    result['material_manifest_sha256'] = audit['material_manifest_sha256']
    result['max_recorded_cleavage_stress_Pa'] = max(float(e['sigma_cleave_eff_Pa']) for e in csv.DictReader((p/'steps_0300K.csv').open()))
    result['stress_cap_Pa'] = args['sigma_cap_GPa'] * 1e9
    result['recorded_stress_cap_active'] = result['max_recorded_cleavage_stress_Pa'] >= result['stress_cap_Pa']
    result['floor_eV'] = actual.cleavage.G00_eV * actual.cleavage.floor_fraction
    result['minimum_event_barrier_eV'] = min(e['hazard_barrier_J']/1.602176634e-19 for e in gates)
    result['event_barrier_at_floor'] = result['minimum_event_barrier_eV'] <= result['floor_eV'] * (1+1e-10)
    result['maximum_recorded_renewal_fraction'] = max(float(e['lambda_c']) for e in csv.DictReader((p/'steps_0300K.csv').open()))*args['multihit_tau']
    result['renewal_ceiling_active'] = result['maximum_recorded_renewal_fraction'] >= .1
    result['diagnostic_scope'] = 'recorded event states; not a reconstruction of unsaved phasewise maxima'
    result['prediction_residual_decade']=math.log10(result['physical_rate']/result['predicted_rate'])
    result['target_residual_decade']=math.log10(result['physical_rate']/result['target_rate'])
    return result


def main():
    frozen=read(ART/'prediction_freeze_manifest.json')
    rows=[analyze(job) for job in frozen['physical_jobs']]
    slopes=[]
    for a,b in zip(rows,rows[1:]):
        width=math.log(b['Kmax']/a['Kmax'])
        s=dict(Klo=a['Kmax'],Khi=b['Kmax'])
        for prefix in ('physical','predicted','target'):
            s[prefix+'_slope']=math.log(b[prefix+'_rate']/a[prefix+'_rate'])/width
        s['event_size_contribution']=math.log(b['mean_length_m']/a['mean_length_m'])/width
        s['waiting_contribution']=-math.log(b['mean_wait_cycles']/a['mean_wait_cycles'])/width
        assert math.isclose(s['physical_slope'],s['event_size_contribution']+s['waiting_contribution'],abs_tol=1e-10)
        slopes.append(s)
    out=ART
    for name,data in [('p40_pilot_physical_results.csv',rows),('p40_pilot_local_slopes.csv',slopes)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    print(json.dumps(dict(rows=rows,slopes=slopes),indent=2))


if __name__=='__main__':main()
