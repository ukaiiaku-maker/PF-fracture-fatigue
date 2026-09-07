# Crack-Rebonding Ablation — Approved Plan Provenance

- Approved plan file: `docs/v10_2_30_crack_rebonding_approved_plan.md`
- SHA-256: `8d56ce3dc5d25cd90fb6fee22d42daefb59e980cfedb520a25ec9c2c681654b9`
- Approved (UTC): 2026-08-30T19:15:13Z
- Branch: `codex/v10.2.30-optional-crack-rebonding`
- Base commit: `b7bd38b97da551aaef1b43d6d28c9ea44a06655c`
- Review history: two rounds of technical review (round 1: twelve points on
  `round(cycles)`, coupling order, `m_h=1`, disabled-path parity, double install,
  unaudited hazard-only assumption; round 2: nine points on phase-resolved coupling,
  the generic exact propagator, the rebonding-coupled event-time root-finder, the
  two-stage fail-closed block limiter, exactly-once-commit, fresh-patch isolation,
  phase provenance, the feedback-mode scope contradiction, and named regression
  tests). Both rounds' corrections are folded into the approved plan text.
- Status: approved by user, software-only pass (Part X physical campaign explicitly
  deferred). Implementation must be checked against this exact plan version — if the
  plan is revised during implementation, update this file's hash and note the change
  and reason here rather than silently drifting from the approved design.

Verify at any time with:

```
shasum -a 256 docs/v10_2_30_crack_rebonding_approved_plan.md
```
