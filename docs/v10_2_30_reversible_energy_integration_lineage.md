# Authoritative reversible energy-gated fatigue integration

## Frozen provenance

- Integration base: `codex/v10.2.30-fatigue-da-dN`, commit
  `54aa2f14b43be5421d8a536a5ab583effa21048b`.
- Four-class qualification: tag `v10.2.30-four-class-qualification-passed`,
  commit `1c82a2532a0dd863825c90cf1262fa4c6dee27b4`.
- Qualified simulation code: `24b63a5bfd86a8ea249d457750b14b8c19488973`.
- Reversible source lineage: `codex/v9.14-intrinsic-reverse-glide-v7`, commit
  `06ac695`.
- Updated oneD/PF evidence: `codex/oneD-v2-terminal-predictive-program`, commit
  `5045feed8c87440676c580fe7f74989be4c4a995`; producer commit
  `cddc51605aee93d8ecbaa0dda76c12085e08f9de`.
- Claude audit evidence is preserved separately at branch
  `codex/v9.14-oneD-v2-mechanics-audit`, commit `273c9f3`. Its campaign
  controller and conclusions are not production authority.

## Executable production call graph

The authoritative fatigue entry is
`sharp_front_v10_2_30_energy_gated_fatigue.main`. It installs
`CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine`, delegates physical
waiting-cycle integration to
`persistent_site_high_cycle_engine_v10230_v5`, and retains the v10.2.30
transaction in this order:

1. persistent-site signed state evolution;
2. stochastic cleavage first passage;
3. exact cycle-number localization;
4. threshold-correlated, mean-preserving event-length proposal;
5. hazard-derived fixed-opening energy admissibility search;
6. checked sharp-wake geometry commit;
7. atomic MPZ translation by the same admitted distance;
8. geometry-specific high-cycle cache invalidation and rebuild.

The stable accelerator alias is
`v10.2.30_production_event_to_event_high_cycle_v5_rate_separated_positive_state_dmd`.
It projects active constitutive state and integrates monotone cumulative ledgers
from separate rates. It is retained, subject to new signed-state parity gates.

Key SHA-256 values at the integration base:

| Component | SHA-256 |
|---|---|
| production entry | `40aa9bee976c716b672335c83658a846168fe7f672808dd5d542ac8bf628783d` |
| corrected energy transaction | `62eade2ed25c784269e34c389e394f391d2caec4915a910ececee548677b45a2` |
| first-passage localizer | `ec0d5a9007d21b7242f3c7c6d3a5f2dc4d3eafbaf65597dc1bc329c2026f3215` |
| energy gate | `c7d6bbe0185fa7660bfe240a7461a3f1246e5c66523b02d889f65517913f054f` |
| high-cycle v5 | `2bc6dd6912dea1d9da572c3a98cde3db599a85e83da2ec789f0e31cd8da9d517` |
| rate-separated DMD v5 | `0770ae997d4ef011febe5397c2330580a41e25e2350432c4f8f77f350566981e` |
| persistent source state | `a9a073806c00fc709f0782110d83be89a51a271d5a42744be439fef7922f8794` |
| shared signed-Burgers state | `b96c1812e8ec59e076e1e8f35b29461dade973561dd8adde6a15b578bd6c39f9` |

## Updated fracture/PF mapping

The v10.2.30 solver already owns the qualified 2-D-derived active shielding
kernel, reliable signed tensor emission drive, finite-tip/blunting state,
persistent source closure, stochastic event, energy gate, and atomic geometry
transaction. The oneD V2 work provides useful source-drive and monotonic
screening evidence, but its `PredictiveLifecycleReduction`, renewal rules,
surrogate/cache controls, and monotonic translation closures are not physical
fatigue machinery and are excluded.

The v7 result contributes a narrower missing capability: intrinsic signed
transport of already-mobile populations, true-reverse-drive classification,
surface-return fate accounting, and bounded cancellation of source-linked
blunting. It must not replace the v10.2.30 event transaction or its 2-D-derived
emission mechanics.

## Integration decision

Architecture **A** is selected: extend the qualified v10.2.30 persistent-site
state with v7's signed reversible mobile-transport and return semantics.

This is preferable to adding the v10 transaction to the standalone v7 runner
because v10.2.30 already has the qualified forward PF mechanics, persistent
source, stochastic threshold lifecycle, energy gate, sharp-wake transaction,
checkpoint contract, and DMD state machine. Porting those into v7 would recreate
the production solver around a less authoritative base. Only the independently
qualified reversible transport behavior will be adapted, using v10 state
objects and v10 tensor/source mechanics.

Emission/source nucleation and already-mobile glide remain separate interfaces.
The v10 reliable 2-D tensor projection remains authoritative for emission. The
v7 signed applied-plus-local-internal-stress rule is considered only for mobile
glide and must pass frozen-state parity before activation. Claude's provisional
Taylor change is rejected: its direct A/B failure did not isolate Taylor, so
Taylor is unchanged pending an independent mechanical test.

## Canonical immutable material rows

The authoritative registry is
`arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv`.
Hashes below are SHA-256 over the complete named CSV row serialized with sorted
keys, so identity is independent of CSV column order.

| Role | Candidate | Row SHA-256 | Qualification provenance |
|---|---|---|---|
| Peak | `v913_zeroD_sobol_0242980` | `83245aa3a01450f08d13820f6a042d1c01518748fbf790b6082879a9ed6fdeb1` | v10.2.25 selection; v10.2.30 four-class gate |
| DBTT | `v913_zeroD_sobol_0202500` | `6d2d454e3c79e2171b8547c895a0ee2d42fa4897af90354efe906e7b27d044d4` | v10.2.24/25 selection; v10.2.30 four-class gate |
| weak-T | `v913_zeroD_sobol_0129902` | `cfc86642687cac968223e4d3ac202c9ff14648c37982d3f188326f0ff25da1e5` | v10.2.27 100-um selection; v10.2.30 four-class gate |
| ceramic-like | `v913_zeroD_sobol_0077080` | `d6abae369475fa80722bc3232603815a73f5ebc19433339dd73a2a84c0a1c3db` | v10.2.27 100-um selection; v10.2.30 four-class gate |

The later `oneD_v2_focused_weak_T_0016` and
`oneD_v2_focused_ceramic_like_0018` rows are screening candidates, not canonical
replacements. They may be tested after solver qualification without changing
the immutable four-row table.

## Claude audit disposition

Retain as evidence:

- separation of source drive from mobile-glide drive;
- the 24-point source-drive discrepancy table;
- inheritance-chain tracing;
- distinction between moving-window loss and physical surface return;
- finding that the R=0.01 failure occurs with both Taylor variants;
- orchestration diagnostics, provided they do not launch production work.

Reject as authority:

- the v7 runner as the production solver;
- Paris-slope or event-count-only parity;
- provisional Taylor sign clipping;
- controller-derived material replacements;
- any surrogate lifecycle closure as constitutive fatigue physics.
