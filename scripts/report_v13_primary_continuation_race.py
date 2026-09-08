"""Report actual frozen evidence and existing short-run records, never launch."""
import json
from pathlib import Path

from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json, sha256
from scripts.run_v13_primary_race_short_ensemble import OUT, CASES, case_folder


def main():
    frozen=json.loads((OUT/'frozen_summary.json').read_text())
    lines=['# V13 inherited companion versus renewed primary','',
        '**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        'The inherited-clock/primary-continuation frozen pattern passes. The older d325b95 audit and its trajectories are immutable. No pair barrier, commitment rate, arrest rate, material change, or branch RNG is introduced. The option is default-off (`--v13-inherited-primary-race`).','',
        '## Clock semantics','',
        'Both clocks are read at the canonical accepted endpoint. Thresholds are cumulative: remaining primary action is the next cumulative threshold minus current action, not the absolute threshold. Pending completed events have zero remaining time without consuming or renewing them. A companion must strictly beat both the primary and tau_c; exact pair admissibility remains required. With multiple candidates, the earliest admissible completion wins; exact admissible ties fall back to the canonical single arm.','',
        'The raw first-passage reconstruction is reported separately, using both clocks at that common time origin. It is not mixed with endpoint action. The raw reconstruction has the archived absolute-time roundoff bound. Both origins give the same eight outcomes. No subgrid time or extra emission update is applied.','',
        '## Frozen physical states','',
        '| Case | Reach (µm) | Primary next (µs), endpoint | Companion (µs), endpoint | Primary next (µs), raw origin | Companion (µs), raw origin | Pair margin (J/m) | Outcome |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in frozen['cases']:
        lines.append(f"| {r['case']} | {r['forward_extension_um']:.5f} | {r['T_i_next_s']*1e6:.8g} | {r['T_j_s']*1e6:.8g} | {r['raw_crossing_T_i_next_s']*1e6:.8g} | {r['raw_crossing_T_j_s']*1e6:.8g} | {r['exact_pair_margin_J_per_m']:.8g} | {'branch opportunity' if r['outcome']=='PAIR_ACCEPTED' else 'single'} |")
    lines+=['','The correlation window is 1 µs. All eight original first-event controls also lose to expiry independently of the primary rate. Every candidate clock, action, threshold, ordinal, seed identity, full process/RNG fingerprint, raw/effective balance and source hash is in the per-event JSON.','',
        'Primary kinetics follow the actual production selection: valid local J when admissible, otherwise the independently evaluated same-plane marginal energy. The companion uses the previously qualified single-to-pair marginal energy. These are model-native/discrete mechanics, not applied remote K or continuum-qualified G. Fixed-state primary-increment solves were performed; no parent trajectory was regenerated. Existing pair mechanics were reused.','',
        'The source callback was exercised on all eight exact archived states with hash-verified cached FEM results and exact request-topology checks. Accepted marks preserve baseline clock/RNG/process/event-counter objects; losing states retain the exact canonical single-state object. An AST regression verifies that the default-off canonical parent body is unchanged apart from explicit observer/overlay hooks and the already-qualified diagnostic aliases.','',
        '## Ideal exponential ensemble identity','',
        'For independent exponential residual clocks at constant rates, P_branch = lambda_j/(lambda_i+lambda_j) × [1−exp(−(lambda_i+lambda_j)tau_c)] × A_pair. Deterministic quadrature verifies the identity. Equal saturated rates give 0.43233235838169365; primary-first has the same probability, and neither completes with probability exp(−2). This ideal identity is not an estimated probability for conditioned archived histories or a calibrated material probability.','',
        '## Short physical gate','',
        'Preregistered: Peak/weak-T at 300/1000 K; common seeds 3621–3624; at most two workers. Each case stops at first branch plus 20 µm additional growth of a daughter (25 µm total daughter length), 125 µm maximum forward reach, or an existing legitimate gate. No automatic retry, seed screen, parameter tuning, or long atlas. The V13 mark ledger is distinct from the unchanged baseline event counters.','']
    records=[]
    for case in CASES:
        path=case_folder(case)/'terminal.json'
        if path.exists():
            records.append(dict(json.loads(path.read_text()),case=case,terminal_sha256=sha256(path)))
    complete=len(records)==len(CASES)
    lines.append(f"Terminal records available: {len(records)}/{len(CASES)}. " + ('All requested cases have terminal records.' if complete else 'The short ensemble is not complete; no final incidence conclusion is claimed.'))
    if records:
        lines+=['','| Case | Branches | Forward reach (µm) | First branch reach (µm) | Terminal reason |','|---|---:|---:|---:|---|']
        for r in records:
            birth=r.get('first_branch')
            reach=r.get('forward_extension_um')
            lines.append(f"| {r['case']} | {r['branch_count']} | {reach if reach is not None else 'unavailable'} | {birth['parent_forward_extension_um'] if birth else 'none'} | {r['reason']} |")
    lines+=['','The frozen evidence supports history-dependent, nonmonotonic branch opportunities, not calibrated recursive branching physics. Earlier attribution found constant 1 µm radius and at most 2.12 ppm direct shielding/blunting correction: the later rate increase tracks geometry-dependent native mechanics, not demonstrated strong direct dislocation-state control. Four seeds per material/temperature support descriptive incidence and onset spacing only.','']
    if (OUT/'snapshot_output_recovery_registry.json').exists():
        lines+=['## Execution provenance qualifications','',
            'The two seed-3621 Peak cases saved accepted pair checkpoints, then encountered a missing required `latest_action` argument in the new snapshot call. The original attempts are preserved as software-interruption evidence. The recovery registry seals those exact pair checkpoint hashes; recovery continues from them without replaying loading, regenerating the parent, changing parameters, or reseeding. They are not classified as physical terminal results.','',
            'Independent fresh-launch full process/RNG hashes are not identical to the earlier parent campaign: its launcher imported the legacy compatibility clock before the campaign environment, whereas the new launcher imports afterward. Directional cleavage clocks are unit-exponential with the same specified seed in both. The unchanged production hook disables the legacy topology clock with threshold 1e300. A direct frozen process-step comparison found differences only in compatibility mode/seed and its unused generator state; all other serialized fields were exact in that fixture. This is a bounded source-state check, not a universal proof. No legacy RNG state was silently replaced. Within each accepted mark and checkpoint continuation, the running baseline clocks and RNG remain exact.','']
    (OUT/'V13_PRIMARY_CONTINUATION_RACE.md').write_text('\n'.join(lines))
    atomic_json(OUT/'short_ensemble_summary.json',{'records':records,'expected_cases':CASES,'complete':complete,
        'calibrated_probability_claimed':False,'boundary':'BRANCHING_KINETICS_MODEL_UNCALIBRATED'})


if __name__=='__main__':
    main()
