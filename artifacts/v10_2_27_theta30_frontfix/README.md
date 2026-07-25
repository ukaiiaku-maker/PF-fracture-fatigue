# v10.2.27 theta=30° front-fixed signed-kernel artifacts

This directory is the durable source of truth for the compact mechanics artifacts required by the v10.2.27 paper campaign.

## Storage policy

Tracked in Git:

- the canonical six-state frozen-geometry snapshot archive;
- the state-table input used to select the six states;
- mechanically derived normalization data;
- accepted load-invariance response tables and audits;
- the final active-only signed-kernel family JSON;
- coverage/workflow manifests and SHA-256 fingerprints;
- restoration and deterministic rebuild scripts.

Not tracked in Git:

- complete simulation run directories;
- step-by-step histories, images, videos, and temporary solver files;
- duplicated extracted copies of the tracked archives;
- analysis plots that can be regenerated from tracked summaries or campaign outputs.

No production runner may require a file that exists only below `runs/` without either:

1. a tracked canonical copy and restoration path, or
2. a deterministic rebuild path from tracked canonical inputs.

## Canonical snapshot archive

Expected path:

`artifacts/v10_2_27_theta30_frontfix/v10_2_27_frozen_geometry_snapshots_700K_theta30_frontfix_E000_E1200_v2.zip`

Canonical archive SHA-256:

`8a4bc221447aa98e8b56b3a1797f42224b6c1bda4da124c37ab2448fd8e4b5ae`

The canonical archive is a cleaned deterministic form of the original uploaded archive. It excludes macOS metadata, uses sorted members and fixed timestamps, and contains these six physical states:

- `E000`
- `E200`
- `E500`
- `E800`
- `E1000`
- `E1200`

Each state contains `snapshot.json` and `state_arrays.npz`. The archive also contains the completed capture manifests and reachable-state trace.

The original uploaded ZIP had SHA-256:

`bd21814144b8d8523e0419eb01022242a841c294a6e779b4cd2868850c158263`

## Remaining canonical artifacts

The snapshot archive is necessary but not sufficient to reproduce the accepted family fingerprint

`35710f0c2f003bea5367d101f0ad27bc93625b0a631dc3f139c6af6a6cfaafbb`.

The following must also be tracked after deterministic reconstruction:

- six-state load-invariance response CSV files and audit JSON files;
- the mechanically derived normalization JSON;
- `v10_2_27_theta30_active_only_campaign_family_frontfix_E1200_v2.json`;
- `v10_2_27_signed_kernel_coverage_audit.json`;
- `v10_2_27_theta30_signed_kernel_workflow.json`.

The final family JSON is a compact production input and must not remain exclusively under `runs/`.
