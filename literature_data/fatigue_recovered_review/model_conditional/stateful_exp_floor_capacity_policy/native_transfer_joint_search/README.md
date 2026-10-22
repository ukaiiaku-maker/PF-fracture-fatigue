# QSTE340TM native-transfer joint-search continuation

## Decision

This executed continuation does **not** establish an accepted single-material
calibration. It supplies a stronger bounded near-frontier and identifies the
remaining failure precisely: the retained EXP_floor/opening reduction cannot
simultaneously reproduce the measured within-R curve shapes and the decrease in
rate with increasing R at fixed Kmax.

The best absolute-rate candidate has RMSE 0.443206 log10(m/cycle), improved from
the preserved `3c840232` faithful-reduction fit (0.569635), but its predicted
broad slopes are only 0.086, 0.488, 0.020 and 0.196 for R=0.1, 0.3, 0.5 and 0.7.
The best shape-oriented Pareto point has shape RMSE 0.121350 and corresponding
slopes 2.429, 2.359, 2.277 and 1.185, but its absolute RMSE worsens to 0.567770.
It is retained as a Pareto comparator, not promoted over the best absolute fit.

At the existing rows nearest Kmax=45 MPa sqrt(m), the measured rates decrease
from 5.868e-7 at R=0.1 to 1.868e-8 m/cycle at R=0.7. The best-absolute
predictions are respectively 7.114e-8 and 6.956e-8; the shape predictions are
7.981e-8 and 1.529e-7. The 48 and 50 MPa sqrt(m) summaries show the same failure.
These contrasts reuse the same measured rows and were not counted again in the
objective.

## Actual native-transfer qualification

`native_transfer_correspondence.json` compares the aggregate reduction against
the actual `UnifiedMPZState` source/activity, exchange, recovery, transport and
moving-frame owners at 32, 128, 512 and 2048 phase panels. Selected and
sustained-active states have close final inventories and opening action under
the deliberately matched one-system, four-bin, uniform-stress geometry. In the
sustained-active 2048-panel case, relative errors are about 1.1e-6 in mobile
inventory, -3.1e-8 in retained inventory, -4.9e-5 in source activity and 1.6e-4
in integrated emission. The integrated escape difference is -0.385, although
the resulting inventories remain close.

The strong-escape control fails transfer: at 2048 panels, retained inventory,
source activity, emission, escape and opening-action relative differences are
approximately 33, 5.8, 33, 26 and 7.6. Refining the phase interval does not
remove the mismatch. This is a structural spatial-versus-aggregate limitation,
not a new material negative. The native moving-frame probe also transfers
mobile and retained content into the wake and refreshes source activity; the
aggregate replay has no equivalent crack-remap owner.

The selected-finalist mid-range checks likewise do not establish process-by-
process identity. Their mobile and emission errors remain small through R=0.7,
but aggregate/native integrated escape differs by 92--99%. Retained-inventory
error reaches -27% for the best-absolute finalist and -58% for the shape
finalist at R=0.7. Opening action is unchanged in these particular checks only
because both fitted cleavage-backstress fractions collapse to effectively zero;
it is not evidence that the unresolved transport is generally irrelevant.

Consequently, the calibration is supported only as a conditional measured-
history replay inside the selected/sustained-active correspondence envelope.
Strong-escape candidates and post-advance predictions remain unresolved. The
aggregate model does not represent native spatial stress decay, persistent
accumulated-slip radius, wake shielding, signed history, crack remap or source
refresh.

## Search and parameters

The continuation evaluated 377 unique profiled plastic candidates, including
historical starts, accepted-manifest recovery scales and activity-stratified
starts. Each of the absolute and shape differential-evolution searches used 168
outer evaluations and two active-basin refinements. Both optimizer success
flags are false because the declared iteration budget was exhausted; the
opening Jacobian applies only to the five conditional opening coordinates.

The best absolute vector is:

- opening floor 17.7545627 kBT, amplitude 3.33284e-5 kBT,
  scale 6.65988e7 Pa, exponent 1.755688;
- cleavage backstress fraction 1.00975e-5;
- emission G0(298 K) 2.524057 eV, plastic critical stress 0.991437 GPa;
- transport G0(298 K) 1.784394 eV, recovery 9.27418e-8 /s;
- initial mobile 2940.875 and retained 77.542 counts in the one-micrometre map.

Its stateful RMSE is 0.443206 versus 0.600443 with state frozen, so the state is
consequential even though the fitted opening response is nearly stress
independent. The best shape vector and every Pareto vector are preserved in
`joint_search.json` rather than summarized incompletely here.

The calculation uses 298.15 K and a provisional 10 Hz frequency. Recovery was
searched from 1e-10 to 1e-2 /s, based on the original accepted-manifest range of
0 to 6.9475e-3 /s; neither finalist is on that bound. No empirical retained
capacity, endpoint-survivor retained source, empirical R multiplier, thermal
fit or modified mechanics was introduced.

## Files and replay

- `joint_fit_DeltaK.png`: common plotted rate axis versus reported DeltaK.
- `joint_fit_Kmax.png`: the same rows and predictions versus derived Kmax.
- `joint_predictions.csv`: all 760 observations in source chronology.
- `fixed_Kmax_contrasts.csv`: exact source-bound fixed-Kmax summaries.
- `joint_search.json`: parameters, Pareto front, convergence, state histories,
  full/frozen results and selected active native comparisons.
- `native_transfer_correspondence.json`: paired native/aggregate states and
  integrated process transfers.
- `execution_manifest.json` and `source_snapshot/`: exact dependencies, hashes,
  runtime versions, fixed assumptions and replay commands.

Run the commands in `execution_manifest.json` from the repository root. The
search command is deterministic for the recorded seed but took several minutes
on the recorded machine. Historical results from `5703aee3`, `1bb0928e` and
`3c840232` remain unchanged.
