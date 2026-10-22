# QSTE340TM faithful retained-state reduction and joint refit

## Decision

The old endpoint-survivor scalar is not a retained-production law. In the
native owner, emission first credits mobile content; encounter/Taylor exchange
then transfers mobile and retained content, recovery removes retained content,
and Peierls transport removes mobile content. The implementation in
`scripts/qste340tm_faithful_reduction.py` preserves that order and uses the
native exact exchange map. The strong-escape audit makes the distinction
quantitative: 3.991250e6 births occur, only 3.683 endpoint mobile survivors are
predicted by the old diagnostic integral, while 3.991230e6 are transferred to
retained content by the actual exchange operator. Inventory closure is
1.1e-9 count (roundoff scale).

The completed finite refit is informative but is **not an accepted
calibration**. A stress-independent opening comparator obtains the smaller
absolute log-rate RMSE (0.4450) by predicting essentially flat curves. The
shape-aware one-scale smooth-broad result consumes an active 0.321--24.012 GPa
stress interval and recovers positive broad slopes of 1.38--2.44, but the
measured slopes are 2.92--3.76 and its aggregate log-rate RMSE is 0.5696.
Consequently neither the old universal plateau nor this near-frontier is a
faithful reproduction of all four measured histories.

## Selected shape-aware result

- Opening: floor 17.3823 kBT, amplitude 1.26086 kBT, scale 9.88948 GPa,
  exponent 1.97594, opening-backstress fraction 2.0.
- Plasticity: emission G0 2.54076 eV, EXP_floor stress scale 21.1834 GPa,
  transport G0 0.878214 eV, retained recovery 2369.09 s-1, shared initial
  mobile/retained counts 46.3858/2.70715.
- The opening barrier actually varies from 17.3634 to 18.6182 kBT over the
  consumed waveforms; unlike the 5703aee3 control, its shape is not wholly
  outside the active stress domain.
- Full/frozen RMSE is 0.5696/0.5836. The fitted trajectory therefore has a
  consequential but small net state effect and does not supply the missing R
  separation.
- Waveform refinement from 32 to 256 panels changes aggregate RMSE by
  5.4e-7. History refinement from one to eight interval panels changes it by
  1.1e-6.

The initial mobile population decays under the fitted high recovery/transport
rates; final retained contents are effectively zero. Peak mobile and retained
counts are 46.39 and 2.71, respectively, and peak mapped density is
5.41e13 m-2. This solution therefore does not rely on a hidden empirical
population cap or an invalid high-density extrapolation.

## Scope and files

`fit_faithful_reduction.json` is the machine-readable fit, search record,
constants, convergence, source hashes, and limitations.
`predictions_faithful_reduction.csv` preserves all 760 records in source order
and adds full, frozen, stress-independent, state, and process columns.
`transfer_audit.json` records low/mid/high loads for all four R histories plus
zero-source and strong-escape continuations. PNG/SVG plots show the common-axis
curves, active G* support, separate mobile/retained histories, and process
transfers.

The comparison is a state-informed aggregate replay, not a spatial native 1-D
calculation. The positive R=0.1--0.7 waveforms do not activate signed return.
The 10 Hz frequency remains the user-authorized provisional assumption. The
finite acquisition does not establish a general failure of EXP_floor
plasticity or broad opening barriers.
