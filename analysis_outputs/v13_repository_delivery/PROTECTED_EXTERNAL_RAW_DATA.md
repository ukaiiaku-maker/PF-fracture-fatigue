# Protected external V13 raw data

The PR contains the complete compact publication record.

Native simulation checkpoints, accepted-interval records, opportunity and branch
ledgers, and portable field inputs remain protected external scientific data and
are not tracked in Git. Their identities are bound by committed provenance and
SHA-256 manifests. A clean Git checkout is therefore **not** a complete copy of
the native raw simulation dataset.

## Canonical retained data

Canonical root:

`/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_theta0_pf_parity_claude/.campaign_execution/PF-fracture-fatigue_v12_field_atlas`

Retention status: **PROTECTED_RETAINED_ON_DATA_DRIVE**.

The authoritative final-report provenance contains **12,913 files**
(**1,290,856,937 bytes**): **12,905 external files**
(**1,288,391,086 bytes**) plus these eight tracked auxiliary inputs:

- `analysis_outputs/v13_heldout_material_classes/full_suite.log`
- `analysis_outputs/v13_heldout_material_classes/full_suite.xml`
- `analysis_outputs/v13_heldout_material_classes/full_suite_result.json`
- `analysis_outputs/v13_heldout_material_classes/heldout_plan.json`
- `analysis_outputs/v13_heldout_material_classes/heldout_source_increment.bundle`
- `analysis_outputs/v13_transition_regime_screen/final_transition_record/case_table.json`
- `analysis_outputs/v13_transition_regime_screen/final_transition_record/final_topologies.json`
- `analysis_outputs/v13_transition_regime_screen/final_transition_record/native_mark_opportunities.json`

This resolves the earlier count discrepancy: 12,905 is the native/untracked
subset; 12,913 is the complete report-input manifest.

The final 64-case cohort comprises eight seeds in each of Peak/300 K,
Peak/1000 K, weak-T/300 K, weak-T/1000 K, DBTT/300 K, DBTT/1000 K,
ceramic-like/300 K, and ceramic-like/1000 K. It contains 57 first branches,
seven completed nonbranching observations through 75 µm, and zero early-gate
censors.

- The 32 DBTT/ceramic held-out cases are rooted under
  `analysis_outputs/v13_heldout_material_classes/ensemble`.
- The 32 Peak/weak-T transition cases and their parent inputs are rooted under
  `analysis_outputs/v13_transition_regime_screen`.
- Earlier theta40 reference evidence is retained under
  `analysis_outputs/v13_primary_continuation_race`; it is supporting evidence,
  not an additional member of the final 64-case theta15 cohort.

Exact counts, byte counts, commit identities, manifest hashes, and mapping to the
compact products are in [external_raw_data_manifest.json](external_raw_data_manifest.json).

## Scientific and source identity

- Frozen V13 physics:
  `35f4a03c622112cd5bc7cf9bc08c5b7fe2387e46`.
- Accepted transition record:
  `b2e0e2d85e048553ad2593d895a98974356213e2`.
- Four-material publication record:
  `22cb426415126241fbc7e6806dc4619796e19e20`.
- Merge-review record:
  `76be5d83456fb6a61394ea0ddffde65f2bee4a23`.
- Canonical family SHA-256:
  `2eac89aa8fa2658dcae791036287c26c7f34e92d971409b0aad84581b9f5e6b7`.

The compact review archive is tracked at
`analysis_outputs/v13_heldout_material_classes/V13_FOUR_MATERIAL_REVIEW.zip`
with SHA-256
`35234f3e78afcbd6679b523aacf402ea025ad9b6ec48f982140729ddcb83e519`.
It contains the scientific report, figures, tables, test records, and source
bundle. It does not contain every native raw input.

## Retention, backup, and cleanup

The current Data-drive copy and all three authoritative manifests verified on
2026-09-09. No independent second durable copy of the complete native dataset
has been verified, so none is claimed. A separate long-term archival plan is
recommended before manuscript publication; that follow-up does not change the
source-PR result.

Cleanup code must treat every path and SHA-256 referenced by
`provenance.json`, `accepted_record_freeze.json`, and
`accepted_ensemble_freeze.json` as protected. These data must not be deleted,
rewritten, relocated, or deduplicated without a separate provenance-preserving
qualification. Superseded accepted physical trajectories remain review-gated.

Before any separately authorized regeneration, verify every manifest entry,
then verify the compact archive and all 32 members against its internal
`SHA256_MANIFEST.json`. This publication task performed only read-only
verification; it did not regenerate reports or figures.
