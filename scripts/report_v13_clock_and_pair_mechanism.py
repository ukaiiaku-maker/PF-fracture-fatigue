#!/usr/bin/env python3
"""Physical interpretation and revised ensemble gate; no simulation launch."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json


def ensemble_gate(evidence, *, common_model_physically_defensible=False,
                  exact_parent_parity=False, exact_pair_and_fallback=False,
                  stable_ordering=False):
    reasons=[]
    if not common_model_physically_defensible:
        reasons.append('common_pair_model_kinetic_scales_not_physically_established')
    probabilities=[r['probability'] for r in evidence if r.get('probability') is not None]
    if not any(.05<=p<=.95 for p in probabilities):
        reasons.append('no_established_intermediate_probability_case')
    if len(probabilities)<2 or max(probabilities)-min(probabilities)<.01:
        reasons.append('measurable_material_temperature_or_evolved_state_response_not_established')
    if not stable_ordering:
        reasons.append('stable_ordering_under_modest_parameter_variation_not_established')
    if not exact_parent_parity or not exact_pair_and_fallback:
        reasons.append('new_model_physical_parent_pair_fallback_gate_not_complete')
    return {'status':'PASS_SHORT_ENSEMBLE_WARRANTED' if not reasons else 'NOT_PASSED_NO_ENSEMBLE',
        'reasons':reasons,'all_eight_intermediate_probabilities_required':False,
        'minimum_intermediate_cases':1,'short_ensemble_launched':False,
        'boundary':'BRANCHING_KINETICS_MODEL_UNCALIBRATED'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    audit=json.loads((args.root/'clock_audit.json').read_text())
    later=json.loads((args.root/'later_companions/summary.json').read_text())
    gate=ensemble_gate([])
    atomic_json(args.root/'revised_ensemble_gate.json',gate)
    lines=['# V13 inherited-clock audit and cooperative-pair decision','',
        '**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        '## Decision','',
        'The first-event nonwinner is far from completion in all eight clean parents. Retain the fresh independent order-three mark as a sequential-null model. A separate default-off cooperative-pair transition is implemented for that regime. Later snapshots reveal short inherited residual times in some states, so a default-off inherited-residual/commitment race is also implemented for the near-completion regime. Neither is registered into production. No branch parameter or seed was tuned. No short branching ensemble was launched.','',
        'The earlier phrase is corrected to `no_stochastic_companion_selected_at_the_preregistered_reference_mark`. All eight original exact pair trials were admissible. Record `d10c3eb`, its original compact archive and all parent checkpoint bytes remain preserved.','',
        '## Existing clocks','',
        '| Case | Nonwinner H/eta at primary crossing | Residual post-primary time (s) | Raw balance | Effective balance |',
        '|---|---:|---:|---:|---:|']
    for case in audit['cases']:
        row=next(r for r in case['candidates'] if not r['winner'])
        lines.append(f"| {case['case']} | {row['normalized_progress_at_crossing']:.8g} | {row['predicted_remaining_post_s_from_raw_crossing']:.8g} | {case['raw_rate_balance_pre']:.8g} | {case['effective_rate_balance_pre']:.8g} |")
    lines+=['',audit['time_interpretation'],'',
        'The candidate table in `clock_audit.json` includes both directions, fixed parent-event thresholds (not the winner’s renewed threshold), actions at the last accepted state/raw crossing/accepted endpoint, ordinals, threshold RNG keys, full process RNG hashes, raw/effective rates and exact pair margins. The primary event is already consumed post-cleavage: its residual completion time is zero; no unsupported next-primary rate is invented. Nonwinner post-primary rates are taken from the already evaluated candidate-specific marginal mechanics.','',
        'Source proof: `draw_branch_mark` uses `gammaincinv(channel.multihit_order, u)/rate` with a separate branch key. `CompanionChannel` has no inherited H or eta input. This is a fresh waiting-time construction, although the preserved baseline clock continues to exist outside the mark.','',
        '## Inverse requirements, not fitted parameters','',
        '| Case | tau for 1% (µs) | tau for 5% (µs) | tau for 50% (µs) | Barrier reduction for 5% (eV) | Raw-arrival multiplier for 5% |',
        '|---|---:|---:|---:|---:|---:|']
    for case in audit['cases']:
        a,b,c=case['inverse_requirements']
        lines.append(f"| {case['case']} | {a['required_tau_B_s']*1e6:.6g} | {b['required_tau_B_s']*1e6:.6g} | {c['required_tau_B_s']*1e6:.6g} | {b['barrier_reduction_eV']:.6g} | {b['raw_arrival_multiplicity_factor']:.6g} |")
    lines+=['',
        'The JSON separately reports rate-prefactor/site multiplication and independent complete-nucleation-site multiplication. The latter obeys 1−(1−P_site)^N and is not interchangeable with multiplying raw arrivals inside one cooperative gamma process. Order-one/two/three probabilities and inverse requirements are all retained. Nothing was applied to production.','',
        '## New pair model','',
        'The unregistered, default-off module implements k_pair = attempt_rate × B × exp(−Q_pair/(k_B T)), competing against explicit primary-commitment and embryo-arrest rates. There is one combined pair barrier. There is no separate junction/overlap fit, no conversion of macroscopic pair energy to atomistic activation work, and no identification of the hit-memory tau_c with a commitment lifetime. Embryo arrest retains the exact accepted single arm; it cannot undo canonical fracture.','',
        'Enabling the module requires explicit kinetic inputs. No physically justified combined barrier, attempt/commitment ratio, or arrest scale was established by this audit, and raw versus effective co-criticality remains an explicit modeling choice. The callable has exact fallback and parent-invariant guards, but is not registered into the production step loop. Unit fixtures are software tests, not physical pair calibration.','',
        '## Later branch-disabled process states','',
        '| Material case | Snapshot | Forward extension (µm) | Radius (µm) | Shielding (MPa√m) | Raw companion rate (s⁻¹) | Exact pair | Null P maximum |',
        '|---|---|---:|---:|---:|---:|---|---:|']
    for r in later['cases']:
        if not r.get('companions'):
            lines.append(f"| {r['material_case']} | {r['case']} | — | — | — | — | {r.get('reason',r['status'])} | — |")
            continue
        for o in r['companions']:
            state=o.get('process_state',{})
            pmax=max((x['P_committed'] for x in r.get('sensitivity_surface',[])),default=None)
            fmt=lambda x:'unevaluated' if x is None else f'{x:.8g}'
            lines.append(f"| {r['material_case']} | {r['case']} | {r['clean_parent_record']['forward_extension_um']:.6g} | {fmt(state.get('radius_m',0)*1e6) if state else '—'} | {fmt(state.get('shielding_Pa_sqrt_m',0)/1e6) if state else '—'} | {fmt(o.get('raw_arrival_per_s'))} | {o.get('exact_pair_admissible',o.get('reason','unevaluated'))} | {fmt(pmax)} |")
    lines+=['','Only Peak/1000 K and weak-T/1000 K were extended, from their certified first-event checkpoints, with branching disabled and unchanged physical parameters. Sparse captures target the next event and 25/50/100 µm. Full source-process profiles, backstress, persistent-source state, geometry, local tensors, fallback identities and pair costs accompany the frozen records. An inadmissible observation is not assigned a fabricated rate or probability.','',
        'Both runs reached 101.8321817 µm with 29 canonical single-arm events and unchanged loading parameters. The first-to-final interval is about 14.403 µs for Peak and 14.072 µs for weak-T, at essentially unchanged opening. These are branch-disabled capability trajectories, not observed physical branching ensembles.','',
        '### Later co-criticality and mechanism change','',
        '| Case | Snapshot | Native companion K (MPa√m) | Raw / effective pre-event balance | Inherited residual time (µs) | Reference null mark |',
        '|---|---|---:|---|---:|---|']
    for r in later['cases']:
        if not r.get('companions'):
            continue
        o=r['companions'][0]
        factors=r.get('inherited_clock_and_pair_driving_factors',{})
        residual=r.get('later_nonwinner_residual',{}).get('post_primary_frozen_residual_s')
        fmt=lambda v:'unevaluated' if v is None else f'{v:.8g}'
        lines.append(f"| {r['material_case']} | {r['case']} | {fmt(o.get('K_companion_discrete_Pa_sqrt_m',0)/1e6)} | {fmt(factors.get('raw_rate_balance_pre'))} / {fmt(factors.get('effective_rate_balance_pre'))} | {fmt(None if residual is None else residual*1e6)} | {o.get('single_reference_overlay_disposition','unevaluated')} |")
    lines+=['',
        'All eight later exact pair trials are admissible. At about 27/53/102 µm, all six unchanged reference marks select and accept the pair. Their full analytic null-model surfaces include both low and high probabilities. Thus negligible first-event branching does not imply negligible later branching under the same null model. No parameter retuning produced this change.','',
        'At approximately 27 and 102 µm the inherited nonwinner residual completes in about 0.12–0.24 µs under the frozen post-primary effective hazard; near 53 µm it takes about 2.73 µs. These later states are not interchangeable with the eight first-event states. The residual implementation integrates supplied effective-hazard segments toward the preserved eta−H, never draws a new cleavage threshold, and races only private commitment/arrest processes. Exact baseline single fallback, parent identities, RNG objects and pair energy acceptance remain guarded. Commitment/arrest scales and any evolving subgrid state closure are still unspecified; this is a default-off implementation, not a qualified stateful multifront production model.','',
        'The radius remains 1 µm. The largest measured direct shielding/blunting raw-rate correction among later captures is about 2.12 parts per million. The strong propensity change tracks the native single-to-pair marginal mechanical drive and resulting raw barrier/rate, not a demonstrated large direct shielding/blunting effect. This frozen attribution does not exclude indirect effects of earlier process evolution, nor does it validate the sharp-wake discrete drive as continuum G or applied remote K.','',
        '## Revised ensemble gate','',
        '`NOT_PASSED_NO_ENSEMBLE`','',
        'The revised gate needs at least one intermediate-probability case, measurable variation with material/temperature/evolved state, stable ordering, a physically defensible common model and exact parent/pair/fallback behavior. It does not require all eight cases to be intermediate. For clarity, the original implementation required two robust material classes, not all eight intermediate cases; the new rule reduces that to at least one qualifying case.','',
        'The new cooperative-pair and inherited-residual models have no physically established commitment/arrest scales. No physically justified pair attempt-rate/barrier ratio or evolving subgrid closure was selected here. Therefore later-state diversity and positive frozen null marks do not alone qualify a common intended co-critical branch model. This is not a requirement that all eight probabilities be intermediate, nor a claim that physical calibration must precede every capability experiment. No large Monte Carlo run, first-parent rerun, long field atlas, or branching ensemble was performed.','']
    (args.root/'V13_CLOCK_AND_PAIR_MECHANISM.md').write_text('\n'.join(lines))
    print(gate['status'])


if __name__=='__main__':
    main()
