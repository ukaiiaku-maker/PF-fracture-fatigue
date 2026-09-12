# V5 final physics-closure campaign at `d36f62e`

This directory is the bounded, Git-safe record of the single final physical
A/B campaign and its postprocess-only recovery. Physical execution used exact
implementation `d36f62e8f4f7350c67e2edcd800748ccb63e1fb9`. Recovery run
[`34372970094`](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/actions/runs/34372970094)
reused the immutable 35-phase artifacts from source run
[`34301599004`](https://github.com/ukaiiaku-maker/PF-fracture-fatigue/actions/runs/34301599004);
it did not rerun a physical worker.

The complete 6,353,047,046-byte publication is stored as GitHub Actions
artifact `10122532888`, digest
`sha256:e24ac5848009a388fd11c335f3cd19b404f3d494c97f3b999684c64c1fc95815`.
It contains 76 bounded gzip parts totaling 6,352,695,895 bytes. No copy of that
large archive is added to ordinary Git history. `publication.json` and
`publication_sha256_manifest.json` provide the reconstruction inventory and
part hashes.

`exact_head_audit.json` records a successful clean-worker reconstruction by
the frozen implementation. The final hardened ledger adds one complete
source-bound ontology field and matching gate that the frozen ledger schema
does not know. The audit therefore proves a remove-only temporary projection:
both projected ledgers are independently derived by the frozen implementation,
its full validator returns zero for both A and B, the original extended ledgers
remain untouched, and no scientific predicate or tolerance is relaxed.

Final science is blocked. Static V1 is 0/33 families and 8/12 derivatives;
controlled V2 is 6/12; accepted continued-front kinetics have no causal
`r_tip` edge; and independent source-native replay is bit-exact for only 10 of
160 natural rows even though all 160 same-worker partition/restart comparisons
pass. The row ontology records 132 PASS and 150 BLOCKED across all 282 rows.
Historical raw full-state neutrality remains FAIL. `r_tip != R_void` is
preserved.

The captured scientific ledger keeps `mission_terminal=false` because branch-
head CI and the external PR ledger follow this immutable evidence commit.
After those checks terminate, the controller and draft PR bodies may record
`V5_FINAL_PHYSICS_CLOSURE_COMPLETE_BUT_BLOCKED`; the evidence itself is not
rewritten retroactively.
