# Import the recovered experimental fatigue data

This package is an actual completed import of the user's supplied files, not the
earlier download starter. Do not repeat unavailable downloads before using it.

## Immediate work
1. Run `python verify_import.py`; preserve the raw files and supplied provenance.
2. Start with `normalized/qste340tm_constant_amplitude.csv`: four measured R
   conditions, 760 author-reduced growth-rate points in m/cycle. The source
   numerical values were mm/cycle and have already been converted.
3. Keep `qste340tm_overload.csv`, `qste340tm_crack_histories.csv` and
   `qste340tm_closure_loads.csv` as paired history/mechanism constraints and
   validation targets. These files contain the same underlying experiments, not
   independent samples to be blindly added to a likelihood.
4. Resolve only the necessary original-paper method details: test T, frequency,
   waveform, geometry/load schedule, OLR definition and overload timing. They
   are explicitly missing, not defaults. Seek the paper locally or at the source.
   Do not infer an absolute Kelvin temperature from a generic room-temperature
   material label. A declared nuisance interval may be appropriate if documented.
5. Once the protocol is adequate, use the EXISTING analytical cycle-integrated
   hazard and existing state-informed model to calibrate the alternative G*
   opening shape with material-specific EXP_floor plasticity. Fit observed
   slope/curvature and rate, not m=4. Do not borrow a synthetic tungsten material
   vector as a supposedly measured steel plasticity calibration.
6. Compare opening-reference and stateful predictions using matched inputs.
   Hold out complete protocol blocks where feasible; do not randomly split
   neighboring correlated points as independent validation. Test overload
   recovery and closure without refitting the barrier separately for each history.

## Scope
- No native simulations or barrier fits have yet been performed on these data.
- No change to mechanics, event reward, clock, source geometry or plasticity
  formulation is authorized by this import package.
- A closure force is not automatically K_open, the reduced blunting radius or an
  additive plasticity fraction. Use a physical observation map; do not fit one
  merely to enforce a desired attribution.
- Do not insert the authors' Wheeler/Alpha/2-PI prediction laws into the existing
  model or calibrate against their predicted curves.
- Preserve source chronology, negative measured increments, and missing values.
- The extra fig15 points and repeated comparisons are explicitly identified;
  resolve those versions before extending the primary measurement set.
- The 316LN data are a separate material. Use its unirradiated controls only after
  resolving CT W and the reported crack-length origin. Preserve B42 as a
  sensitivity/outlier record, and keep irradiated histories separate.
- Keep actual experimental observations distinct from inferred physical quantities,
  model-conditional barriers, regularization and user-imposed priors.

## Deliverables for an eventual fitting task
A source-bound experimental dataset, fitted and held-out curves, barrier and
activation-volume uncertainty over the data-informed local-stress interval,
matched full/reference mechanism contrasts, and explicit unresolved metadata.
Do not claim four-temperature-class experimental validation from these nominally
isothermal datasets.
