# v10.2.27 mechanical-kernel artifacts and resolver

This directory is the durable source of truth for compact mechanics artifacts used by the v10.2.27 paper campaign. Production runs request a **mechanical configuration** and required crack-growth coverage; they do not require a manually preserved `FAMILY_JSON` path.

## Resolver policy

The official runners call `scripts/ensure_v10_2_27_signed_kernel.py`. The resolver:

1. canonicalizes the mechanical configuration and computes a SHA-256 fingerprint;
2. validates an explicit override when one is intentionally supplied;
3. reuses a matching tracked or local cached family with sufficient coverage;
4. restores/builds a family from registered portable mechanics artifacts;
5. invokes a registered configuration-specific builder when no reusable family exists;
6. caches the result under `runs/v10_2_27_kernel_cache/<configuration-fingerprint>/`;
7. refuses to substitute a single-front atlas for branching topology.

A stale missing `FAMILY_JSON` is ignored by default and triggers mechanical resolution. Set `KERNEL_STRICT_FAMILY_OVERRIDE=1` only when a missing explicit override must remain fatal.

## Mechanical identity

The fingerprint includes mechanics that can change the signed FEM response, including orientation, specimen/BC profile, initial crack geometry, mesh/process-zone policies, active-station convention, elasticity policy, interaction length, and topology mode.

It deliberately excludes material kinetic parameters, hazard seeds, plotting/snapshot settings, and requested target extension. Target extension is a **coverage requirement**: an existing longer atlas may serve a shorter campaign.

Temperature enters the fingerprint only when `TEMPERATURE_DEPENDENT_MECHANICS=1`.

## Branching

The current tracked atlas provider is single-front only. Branching requests cannot fall back to it. A branching configuration must register a `topology_cached` or `direct_fem` builder. This is a model-capability gate, not a filesystem error.

## Storage policy

Tracked in Git:

- the canonical six-state frozen-geometry snapshot archive;
- accepted load-invariance response tables and audits;
- kernel configuration/registry records;
- final compact family JSON and coverage/build manifests when available;
- deterministic restoration/build scripts.

Not tracked in Git:

- complete simulation run directories;
- step histories, figures, videos, and temporary solver files;
- extracted duplicate copies of tracked archives;
- local kernel caches, which are reproducible from the registry/build protocol.

No production runner may require a file that exists only below `runs/` without either a tracked portable source or a registered builder.

## Canonical snapshot archive

Expected path:

`artifacts/v10_2_27_theta30_frontfix/v10_2_27_frozen_geometry_snapshots_700K_theta30_frontfix_E000_E1200_v2.zip`

Canonical archive SHA-256:

`8a4bc221447aa98e8b56b3a1797f42224b6c1bda4da124c37ab2448fd8e4b5ae`

It contains completed states `E000`, `E200`, `E500`, `E800`, `E1000`, and `E1200`. Each state contains `snapshot.json` and `state_arrays.npz`, including the serialized engine configuration used to derive mechanics normalization automatically.

## Canonical load-invariance archive

Expected path:

`artifacts/v10_2_27_theta30_frontfix/v10_2_27_active_load_invariance_700K_theta30_frontfix_E000_E1200_v2.zip`

Canonical archive SHA-256:

`e71d9dcae52129a175100fa56f3f0445105536598bc21d160624c75b5b52b919`

For every state it contains measured responses at load scales 0.5, 1.0, and 1.5, response audits, the combined load sweep, and the passed frozen-geometry load-invariance report.

The portable builder rewrites only serialized absolute provenance paths into current validated local paths; response data and audits remain unchanged.

## Automatic construction for a new configuration

For a new orientation or other single-front mechanics configuration, provide a builder command rather than a reference family:

```bash
KERNEL_BUILD_COMMAND="bash scripts/build_v10_2_27_kernel_for_configuration.sh"
KERNEL_CAPTURE_COMMAND='<command that writes frozen states to $V10227_KERNEL_CAPTURE_OUTROOT>'
```

The builder automatically evaluates load invariance for the generated states, derives normalization from the snapshot engine configuration, builds and validates the family, and registers it in the local cache.

The capture command is configuration-specific because crack geometry and topology are part of the mechanics. It must generate at least two states and sufficient extension coverage. The resolver prevents accidental reuse across incompatible configurations.

## Registry

`artifacts/v10_2_27_kernel_registry.json` contains tracked families and rebuild recipes. The accepted unbranched θ=30° mechanics are registered as a portable-artifact recipe. The legacy raw family SHA was:

`35710f0c2f003bea5367d101f0ad27bc93625b0a631dc3f139c6af6a6cfaafbb`

That raw hash included run-location provenance. New builds record both file integrity and a path-independent physics fingerprint so equivalent mechanics remain portable across machines and cache locations.
