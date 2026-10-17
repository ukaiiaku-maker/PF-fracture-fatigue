# R10 mechanical/mixed cohesive opening comparison — initial information batch

## Decision

The additive family is implemented and executed without changing the legacy cohesive–Arrhenius family or the EXP_floor plasticity inputs. The initial batch does **not** identify a strict Paris-window pass. Its useful result is a resolved shape/mechanism contrast: matched bilinear laws frequently cross into abrupt mechanical advance while smooth exponential laws are commonly inactive or limited to a single-excursion burst under the same retained dislocation material.

## Completed work

- Analytical: 8,192 tuples allocated; 8,141 domain-valid; 32,564 matched arm records.
- State-informed: 48/48 complete materials received fresh native retained-state exposure.
- Native 1-D: 12 materials × 4 arms × 3 loads = 144 cases.
- Developed follow-up: R10D_COH_27231, 4 arms × 6 loads, one-micrometre geometry-owned cells.
- Same-material 2-D: 12 nonlinear cohesive/plastic transfer cases at 300, 750, and 1200 K.

## Fatigue disposition

The refined R10D_COH_27231 bilinear and high-load exponential responses reached the extension target as a single-excursion cascade. They therefore remain startup-burst censors, not developed rates. No local slope is reported from those bursts, and the strict 3.4–4.6 / two-rate-decade objective has zero passes in this batch. Lower loads are true fixed-domain non-advance censors over their declared exposure.

## Mechanism and thermal observations

The initial 12-material block contains 14 paired load points where smooth and bilinear event counts differ despite identical plasticity hashes. In 2-D, athermal arms ended at a resolved FE branch-loss boundary, while mixed arms reached qualified thermal escape; the temperature-specific trigger coordinates are retained in `results_summary.json`. Low-barrier crossover intervals were recorded as unsupported first-passage censors and were not evaluated with the old cooperative clock.

## Interpretation limits

The 2-D block is a selected thermal-fracture transfer, not a developed spatial fatigue curve. The native 1-D screen is intentionally separated from the developed follow-up. Numerical renewal caps were removed before the reported follow-up, so its burst classification is physical within the declared one-micrometre interface-cell discretization rather than a capped-rate artifact.
