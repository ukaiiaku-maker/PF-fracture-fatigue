# R10 alternative-G* joint search

## Decision

The explicitly identified alternative crack-opening family is no longer comparison-only. Four complete, same-mechanics materials pass the joint screen: one each with Weak-T, Peak-T, DBTT, and ceramic-like fracture-temperature topology. Every material also has a five-point native developed-fatigue curve whose four adjacent local slopes remain in the 3.4–4.6 band.

This conclusion is specific to the alternative thermodynamic `G*` family bound to the accepted V3.1/R9 current 1-D backend. It does not alter or replace the EXP_floor bank, the dislocation surface, mechanics, stochastic event law, source geometry, or the separately scoped CZM-hazard work.

## Reused evidence and new work

- Reused without replay: the accepted V2/V3 banks, the V3.1 `V3_ceramic_V2_P25_007__V3_1_TRUST1` lineage, and the R9 common-window workflow.
- New analytical acquisition: 4,096 Sobol vectors jointly varying stress scale/exponent, entropy constant/shape, and heat-capacity constant/shape. The 300 K barrier was preserved at three stress anchors, and the emission/plasticity surface was unchanged.
- 891 vectors passed positivity/monotonicity plus the analytical fatigue screen. The qualified role counts were Weak-T 3, Peak-T 64, DBTT 77, ceramic-like 9, other 136, and thermally censored 602.
- New current full-state fracture work: 28 cases, four materials at 300–1200 K in 150 K increments. Every case completed 11 events and 55 micrometres without a censor.
- New current full-state fatigue work: 20 cases, four materials at five amplitudes. Every case completed 20 events and 117.797 micrometres. Rates use the common developed 20–100 micrometre window, not startup secants.

## Joint results

| Role | Material | Full-state fracture K-init at 300/450/600/750/900/1050/1200 K (MPa sqrt(m)) | Native local slopes | Connected rate width (decades) |
|---|---|---|---|---:|
| Weak-T | `R10_ALT_GSTAR_02143` | 3.5949 / 3.8877 / 4.0043 / 4.0508 / 4.0689 / 4.0763 / 4.0772 | 4.0609 / 4.1722 / 4.3609 / 4.2282 | 4.2871 |
| Peak-T | `R10_ALT_GSTAR_02506` | 3.5956 / 11.0747 / 7.4507 / 6.3686 / 5.9165 / 5.6665 / 5.5082 | 3.9853 / 4.2789 / 4.5665 / 4.2754 | 4.3594 |
| DBTT | `R10_ALT_GSTAR_00348` | 3.5287 / 8.3404 / 9.4013 / 9.8873 / 10.1632 / 10.3376 / 10.4559 | 3.5084 / 3.8561 / 4.3456 / 4.2422 | 4.0654 |
| Ceramic-like | `R10_ALT_GSTAR_02460` | 3.5502 / 3.4032 / 3.0835 / 2.7911 / 2.4997 / 2.1572 / 1.6685 | 3.6864 / 3.9458 / 4.3329 / 4.2431 | 4.1306 |

The native state-correction contributions to the four log-slopes are small for these selected rows (largest magnitude 0.0562). These are therefore opening-controlled fatigue examples under the retained EXP_floor plasticity background; they are not evidence that plasticity was removed or that all alternative-family candidates are opening-controlled.

## Scope and claim limits

- Temperature behavior comes from the single thermodynamic surface of each material, including its entropy and heat-capacity terms. No cold/hot endpoint was fitted independently.
- The four labels describe the topology of the native fracture-temperature curve. In particular, the DBTT row is an opening-barrier DBTT role, not a claim that its transition was created by plasticity.
- The calculation establishes internally consistent joint capability in the retained model. It is not experimental validation or uniqueness of these parameter vectors.
- CZM-hazard results remain a separate comparator with their previously declared mechanical-scope limits. Nothing here converts an unresolved CZM case into a material failure, and no CZM result gated this search.

## Evidence map

- `alternative_gstar_registry.json`: complete selected material vectors and hashes.
- `analytical_summary.json`, `selected_roles.csv`, `analytical_qualified.csv`: acquisition population and analytical predictions.
- `full_state/*/panel_summary.json`: native thermal-fracture cases.
- `fatigue/*/campaign_summary.json`: accepted cycles, events, extensions, and native result paths.
- `analysis/*`: developed-rate, bootstrap, state, local-slope, and connected-width tables and plots.
- The delivery archive `R10_ALTERNATIVE_GSTAR_JOINT_EXECUTED.zip` contains the complete run tree, including live restart/checkpoint state.
