# Legacy ZIP independence test for campaign V2

Runtime verdict: **PASS**. The 288-row V2 plan, family resolution, seed/rate
validation, current-status regeneration, completed-case verification, and
pending-case dry-run validation do not read `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/1_backupdatatouse.zip`.

Deletion verdict: **NOT_READY_TO_DELETE**.

The ZIP is no longer a runtime dependency, but it remains the retained
historical record for legacy theta/rate products not proven to have a second
content-equivalent archive. Runtime independence alone is insufficient to
authorize destructive deletion. Keep the ZIP until a complete member-level
replacement archive is verified.

Runtime ZIP-reference findings: 0. Any listed references are
fail-closed audit/history references, not V2 launcher dependencies.
