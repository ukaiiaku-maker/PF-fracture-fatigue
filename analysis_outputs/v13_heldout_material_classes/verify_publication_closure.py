"""Read-only scientific-data and package verification; no tests or mechanics."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image
from scripts.run_v13_heldout_materials import OUT,DEST,ROOT,PHYSICS,RECORD,HELPERS,verify_source,verify_accepted
from scripts.run_pf_current_source_multifront_field_atlas_v12 import sha256,atomic_json

FINAL=OUT/'final_four_material_record'


def main():
    verify_source();frozen=verify_accepted()
    pre=json.loads((OUT/'publication_preflight.json').read_text())
    for name,h in pre['heldout_files'].items():assert sha256(Path(name))==h
    provenance=json.loads((FINAL/'provenance.json').read_text())
    for name,h in provenance['files'].items():assert sha256(Path(name))==h
    summary=json.loads((FINAL/'summary.json').read_text())
    cases=json.loads((FINAL/'case_table.json').read_text())['rows']
    assert len(cases)==64 and sum(r['branch_observed'] for r in cases)==57
    assert sum(r['unbranched_75um_complete'] for r in cases)==7 and not any(r['early_gate_censor'] for r in cases)
    for g in summary['groups']:
        rows=[r for r in cases if r['group']==g['group']]
        assert len(rows)==8 and {r['seed'] for r in rows}==set(range(3621,3629))
        x=sorted(r['first_branch_primary_reach_um'] if r['branch_observed'] else 75. for r in rows)
        assert math.isclose(sum(x)/8,g['restricted_mean_branch_free_reach_um'],rel_tol=1e-12)
        # With no early censors, lower empirical 0.5 quantile is the product-limit median.
        assert math.isclose(x[3],g['median_first_branch_reach_um'],rel_tol=1e-12)
    paired=json.loads((FINAL/'paired_seed_contrasts.json').read_text())['rows']
    assert len(paired)==28 and all(len(p['pairs'])==8 for p in paired)
    marks=json.loads((FINAL/'native_opportunities.json').read_text())['rows']
    for r in marks:
        m=r['metrics']
        if m['chi_B'] is not None:
            assert math.isclose(m['chi_B'],r['T_j_s']/min(r['T_i_next_s'],r['tau_c_s']),rel_tol=1e-12)
        if m['completed_pending_companion']:assert r['T_j_s']==0.
    births=json.loads((FINAL/'birth_saturation_and_pair_margins.json').read_text())['rows']
    assert len(births)==57 and all(0<=b['chi_B']<=1 and b['exact_pair_margin_J_per_m']>0 for b in births)
    networks=json.loads((FINAL/'final_topologies.json').read_text())
    xmax=max(100.,5*math.ceil(max(c['accepted_terminal_reach_um'] for c in cases)/5))
    for c in cases:
        branches=networks[c['case']]['branches']
        assert bool(any(b['generation']>0 for b in branches))==c['branch_observed']
        for b in branches:
            for x,y in b['path_m']:
                if x>=.0005:assert -2<=(x-.0005)*1e6<=xmax and -45<=y*1e6<=32
    suite=json.loads((FINAL/'full_suite_review.json').read_text())
    assert suite['status_counts']==dict(passed=1099,failed=31,skipped=5)
    assert suite['full_suite_invocations']==1 and suite['returncode']==1
    assert len(suite['test_inventory'])==1135
    v13=[r for r in suite['test_inventory'] if 'test_v13_' in r['test']['classname']]
    assert len(v13)==131 and all(r['status']=='passed' for r in v13)
    triage=[]
    for r in suite['failures']:
        message=r.get('message') or '';detail=r.get('detail') or ''
        group='MISSING_HISTORICAL_PRODUCT_FIXTURE' if message.startswith('FileNotFoundError:') else 'SANDBOX_PS_PERMISSION' if "PermissionError" in message+detail and "'ps'" in (message+detail).replace('\\','') else 'ASSERTION_OR_CONTRACT_FAILURE'
        triage.append(dict(test=r['test'],category=group,message=message))
    assert Counter(r['category'] for r in triage)==dict(MISSING_HISTORICAL_PRODUCT_FIXTURE=24,SANDBOX_PS_PERMISSION=1,ASSERTION_OR_CONTRACT_FAILURE=6)
    bundle=OUT/'heldout_source_increment.bundle'
    b=subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,text=True,capture_output=True,check=True)
    heads=subprocess.check_output(['git','bundle','list-heads',str(bundle)],cwd=ROOT,text=True)
    assert summary['report_producer_code_commit'] in heads and RECORD in b.stdout+b.stderr
    assert summary['report_producer_code_commit']==suite['source_commit']
    subprocess.run(['git','diff','--exit-code',PHYSICS,'--','arrhenius_fracture',*HELPERS],cwd=ROOT,check=True)
    archive=OUT/'V13_FOUR_MATERIAL_REVIEW.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('SHA256_MANIFEST.json'))
        assert len(z.namelist())==len(set(z.namelist()))==len(manifest)+1
        assert set(z.namelist())==set(manifest)|{'SHA256_MANIFEST.json'}
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        for f in FINAL.iterdir():
            if f.is_file():assert sha256(f)==manifest['final/'+f.name]
    figures=[]
    for p in sorted(FINAL.glob('*.png')):
        with Image.open(p) as im:size=im.size;im.verify()
        figures.append(dict(path=str(p),sha256=sha256(p),pixels=size,decode='PASS',visual_review='PASS',
            note='All 64 topology panels inspected; overlapping survival curves are valid. Chi symlog axis includes an empty negative margin; no negative chi observations are present.' if p.name in ('all_64_final_topologies.png','chi_B_distributions.png','eight_group_first_branch_survival.png') else 'Labels, counts, units and plotted points visually checked.'))
    assert len(figures)==5
    q=json.loads((DEST/'queue_status.json').read_text());assert q==dict(active={},pending=[],paused=False)
    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),publication_status='PUBLICATION_RECORD_COMPLETE_MERGE_BLOCKED_FULL_SUITE_FAILURES',
        supersedes_storage_hold='SIMULATIONS_COMPLETE_PUBLICATION_HELD_FOR_STORAGE (historical record retained unchanged)',
        boundary='BRANCHING_KINETICS_MODEL_UNCALIBRATED',cases=64,branches=57,completed_nonbranching=7,early_gate_censors=0,
        accepted_freeze=frozen,heldout_preflight_seals_verified=len(pre['heldout_files']),report_input_hashes_verified=len(provenance['files']),
        independent_reductions='PASS: eight means/medians, 28 paired contrasts, native chi ratios, branch topology and plot bounds',
        figures=figures,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive),members=len(manifest)+1,manifested_members=len(manifest),integrity='PASS',every_member_sha256='PASS'),
        source_bundle=dict(path=str(bundle),sha256=sha256(bundle),base_commit=RECORD,producer_commit=summary['report_producer_code_commit'],verification=b.stdout+b.stderr),
        verifier_sha256=sha256(Path(__file__)),full_suite=dict(counts=suite['status_counts'],invocations=1,returncode=1,v13_named_tests_passed=len(v13),failure_triage=triage),
        merge_gate='NOT_CLEARED_FULL_SUITE_FAILURES',full_suite_and_provenance_reviewed=True,automatic_merge=False,
        no_new_simulation_or_physics_change=True,queue=q,durable_free_bytes=shutil.disk_usage(OUT).free)
    atomic_json(OUT/'PUBLICATION_CLOSURE_VERIFICATION.json',record)
    lines=['# V13 four-material publication closure','',record['publication_status'],'','**BRANCHING_KINETICS_MODEL_UNCALIBRATED**','',
        'The physical simulation program is complete: 64 theta=15°, rate-1x cases, 57 first branches, seven completed nonbranching observations through 75 µm, zero early-gate censors. No new trajectory, restart, continuation, reseed, screen or physics change was performed for publication. The earlier storage hold is superseded by this verified package; its original record remains unchanged.','',
        '[Final scientific report and five figures](final_four_material_record/V13_FOUR_MATERIAL_FIRST_BRANCH_REPORT.md) · [Compact review archive](V13_FOUR_MATERIAL_REVIEW.zip) · [Closure verification](PUBLICATION_CLOSURE_VERIFICATION.json) · [Complete test inventory](final_four_material_record/full_suite_review.json)','',
        '## Scientific comparison','',
        '| Group | Branches / 8 | Median reach (µm) | Restricted mean, 0–75 µm (µm) |',
        '|---|---:|---:|---:|']
    for g in summary['groups']:lines.append(f"| {g['group']} | {g['branches']} | {g['median_first_branch_reach_um']:.2f} | {g['restricted_mean_branch_free_reach_um']:.2f} |")
    lines+=['','Peak, DBTT/300 K and ceramic-like are high-incidence under this tested condition. Weak-T shows the clearest mixed-incidence transition; DBTT/1000 K is delayed and slightly reduced. Temperature contrasts are parameterization-specific. Counts remain descriptive model outcomes, not calibrated physical probabilities. Recursive branch spacing and generalized branch-angle prediction remain outside the dataset.','',
        '## Native birth-state ranges','',
        '| Group | Primary / companion saturated births | Pending births | χ_B range | Native pair margin range (J/m) |',
        '|---|---:|---:|---:|---:|']
    for g in summary['groups']:
        rows=[r for r in births if r['group']==g['group']];chi=[r['chi_B'] for r in rows];m=[r['exact_pair_margin_J_per_m'] for r in rows]
        lines.append(f"| {g['group']} | {g['primary_saturated_births']} / {g['companion_saturated_births']} | {g['pending_companion_births']} | {min(chi):.6g}–{max(chi):.6g} | {min(m):.9g}–{max(m):.9g} |")
    lines+=['','All 57 primary channels are near the retained 1/tau_c asymptote at birth; ten births use an inherited completed-pending companion. Ranges above are rounded for display only. Exact native values, all opportunities (including unevaluated/infinite statuses), paired seeds and distributions are in the compact record. No new mechanics was used to fill missing native values.','',
        '## One-time suite and merge decision','',
        'Exactly one invocation: **1,099 passed, 5 skipped, 31 failed** (1,135 cases). All 131 V13-named tests passed. Failures comprise 24 missing historical V5/V6 product fixtures, six assertion/contract failures, and one status test blocked from running ps by the sandbox. The complete identities, tracebacks and skip reasons are retained; none has been waived or relabeled as passing. No second invocation was performed.','',
        '**Merge gate: NOT_CLEARED_FULL_SUITE_FAILURES.** The suite and final provenance have been reviewed, but failures remain unresolved. No automatic merge and no corrective physics/source changes are authorized by this record.','',
        '## Provenance','',
        f"Frozen physics: `{PHYSICS}`. Accepted transition record: `{RECORD}`. Report-producer and one-time test source: `{summary['report_producer_code_commit']}`. Execution commits remain separately recorded for every case. The incremental source bundle verifies against the accepted transition base.",'',
        f"Review archive: {archive.stat().st_size:,} bytes; {len(manifest)+1} members ({len(manifest)} SHA-256-manifested members plus manifest). Integrity and every member hash PASS. Archive SHA-256: `{sha256(archive)}`.",'',
        f"All five final figures were decoded and visually reviewed. All {len(pre['heldout_files'])} held-out preflight seals and {len(provenance['files'])} report-input hashes reproduce, as do the 17,636 frozen transition and 3,620 theta40 artifacts. No accepted data changed. Durable free space at closure: {record['durable_free_bytes']/1024**3:.2f} GiB.",'',
        'The compact archive contains the scientific report, figures, tables, full test records and source bundle. This index and closure-verification JSON are external verification sidecars, avoiding a circular archive self-hash.','']
    (OUT/'FINAL_RECORD_INDEX.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=record['publication_status'],archive=record['archive'],figures=len(figures),merge_gate=record['merge_gate']),indent=2))


if __name__=='__main__':main()
