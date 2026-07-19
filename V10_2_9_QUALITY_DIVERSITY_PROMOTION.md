# v10.2.9 quality-diversity promotion

## Purpose

A scalar objective is useful for ranking, but pure top-N promotion can collapse a
large Sobol or cross-entropy population into one narrow parameter basin.  That is
particularly risky in this staged campaign because analytical and first-passage
screens cannot yet observe every behavior used later, including R-curve shape,
signed population history, wake retention, and mechanism sensitivity.

v10.2.9 changes only the downselection policy.  It does not alter any material
parameter, barrier, mechanical artifact, state equation, or 2-D calculation.

## Selection policy

For each material class and each gate:

1. Stage-pass candidates are used exclusively whenever enough passers exist to
   fill the promotion budget.
2. The best 25% of the promotion budget is reserved by pure objective rank.
3. At the analytical DBTT stage, the best candidate from each available dominant
   historical-anchor lineage is retained when that lineage is not already
   represented.
4. Remaining slots are selected greedily by a weighted quality-diversity score:

   ```text
   utility = 0.35 * objective-quality percentile
           + 0.65 * combined novelty
   ```

5. Combined novelty uses robustly normalized distances in two independent spaces:

   ```text
   combined novelty = 0.45 * parameter-space novelty
                    + 0.55 * response-space novelty
   ```

The quality reserve and weights are configurable, but the defaults intentionally
favor response diversity slightly more than raw parameter distance.

## Parameter-space diversity

The selector compares the complete searched material definition:

- cleavage and emission EXP-floor barriers;
- Peierls and Taylor transport barriers;
- activation entropies;
- Taylor correlation scale and density;
- source capacity;
- encounter, recovery, refresh, and blunting parameters;
- historical anchor weights.

Positive scale parameters spanning orders of magnitude are compared in log space.
Signed slopes and activation entropies remain linear.  Every feature is robustly
scaled using population quantiles, and the parameter block is normalized by its
number of active dimensions.

## Response-space diversity

### Analytical promotion

The selector uses the complete 300--1200 K trajectory at 100 K increments for:

- no-feedback cleavage first passage;
- first emission;
- emission advantage;
- linearized signed source-bin shielding;
- expected source activations and signed line content;
- retained-fraction indicator.

This is deliberately broader than the scalar analytical objective.  Candidates
with similar objective values but different transition temperatures, shielding
signatures, source use, or retention trends can therefore survive to the exact
first-passage stage.

### First-passage promotion

DBTT diversity uses the exact 300, 700, 900, and 1200 K first-passage trajectory.
FCC-like weakT diversity uses 300, 700, and 1200 K.  This preserves different
transition shapes rather than keeping only candidates with the same endpoint
ratio.

### R-curve promotion

The final reduced downselection uses initiation and final toughness at every
class-specific temperature together with R-rise and endpoint-ablation metrics.
The four candidates promoted to 2-D therefore include the best-scoring system and
three additional high-quality systems chosen to represent distinct R-curve and
mechanism responses.

## Protection against low-quality novelty

Diversity does not override the stage gates.  When at least N candidates pass a
gate, every promoted candidate is selected from the passing population.  When
fewer than N pass, all passers enter the eligible pool before near-pass candidates
are considered.

The candidate pool is also limited to the best `pool_factor * N` systems by the
stage objective, with a default factor of 12.  This prevents very poor outliers
from being promoted merely because they are far away.

## Audit outputs

Every analytical, first-passage, and R-curve stage writes:

```text
quality_diversity_selection.json
```

The audit records:

- selected candidate IDs and selection reasons;
- objective ranks and quality percentiles;
- dominant anchor lineage;
- parameter and response features used;
- robust feature scaling;
- nearest-neighbor distances among selected candidates;
- pairwise diversity of the selected set;
- pairwise diversity of a pure top-N reference set;
- whether diversity changed the promotion set.

Promoted CSVs also contain per-candidate selection annotations.  Manifest CSVs
remain clean because only physical material fields are written.

## Defaults by stage

With the planned campaign sizes:

```text
analytical -> first passage: 256/class
    pure quality reserve: 64/class

first passage -> R-curve: 48/class
    pure quality reserve: 12/class

R-curve -> 2-D: 4/class
    pure quality reserve: 1/class
```

The remaining slots balance quality and novelty.  These defaults can be changed
through:

```text
QD_QUALITY_RESERVE_FRACTION
QD_QUALITY_WEIGHT
QD_PARAMETER_WEIGHT
QD_RESPONSE_WEIGHT
QD_POOL_FACTOR
QD_PRESERVE_ANCHOR_LINEAGES
```

The physical signed shielding-kernel family and signed drive family remain hard
prerequisites.  This selection update does not fabricate or authorize either
mechanical artifact.
