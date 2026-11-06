# Codex handoff: Ast 2018 per-temperature joint fit (signed pile-up), 2026-11-04

This continues the Ast 2018 tungsten microcantilever fit made with the 1-D
native Arrhenius crack-tip model. It is a separate workstream from the
v10.2.30 fatigue task in `CODEX_HANDOFF.md`. Read `AGENTS.md` first; its
constraints still apply.

## 1. State at handoff

- Repository: `/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30`
- Branch: `codex/v9.14-v7-paris-physics-informed-continuation`
- HEAD: `30964f2c` (Clarify analytical versus retained Ast report quantities)
- **Nothing from this workstream is committed.** The worktree also holds
  other agents' uncommitted changes (see section 7). Do not `git add -A`.
- No runs are active.

## 2. Scientific objective and user decisions (binding)

Fit Ast 2018 J–Δa, nominal-K and stiffness data jointly at each temperature
(21, 100, 250 and 500 °C) **separately**. Impose no temperature dependence
on any barrier. S* is whatever the per-temperature values imply afterwards.

These are the user's decisions. Do not reopen them without asking.

1. The Arrhenius barrier form is immutable. Fit the barrier values, not new
   laws.
2. **The point of the model is the competition between crack-tip emission
   and opening.** A fit that scores well with tip emission switched off is
   not acceptable as the answer. Always report `tip_activations`,
   `tip_shielding` and the change in tip radius for any best fit.
3. Plasticity may act anywhere in the section (`full_section`). Keep
   tip-only plasticity as an option. The bulk does not nucleate: carriers
   come only from the forest floor (`bulk_nucleation=False`). Most of the
   response should come from the crack tip, with only a little bulk flow.
4. **There is no cap on Taylor φ.** The user's ruling, quoted: "If phi is
   large then the Peierls should dominate. Phi does not apply to dislocations
   of the same sign that pile up. It mainly applies to random forest type
   dislocations which would occur in places where there is mixed signed
   content. A series of emissions from a crack tip on the same plane will not
   interact via the Taylor mechanism." This is implemented as
   `tip_interaction="signed_pileup"`.
5. Do not lift the 2-D production bulk-nucleation block without the user's
   explicit OK.

## 3. What was built (uncommitted files belonging to this workstream)

| File | Role | Category |
|---|---|---|
| `fracture_fit/experimental_fits/ast2018/per_temperature_joint_fit.py` | Isothermal 7-coordinate fit per temperature. Uses a two-sided censored-J objective, linked load and stiffness traces, space filling plus Nelder–Mead with a restart, and a resumable JSONL cache. | fitting tooling |
| `fracture_fit/experimental_fits/ast2018/section_plasticity.py` | Opt-in `full_section` bulk hinge channel used by `spatial.predict_spatial`. | physics option (opt-in) |
| `fracture_fit/experimental_fits/ast2018/spatial.py` | Adds `TIP_INTERACTIONS = ("native", "signed_pileup")` in `_mechanical_state_type`. The native path is unchanged (`super()`). **Other agents also edit this file**; see section 7. | physics option (opt-in) |
| `fracture_fit/experimental_fits/ast2018/implicit_source.py` | Shared with earlier Ast work. This workstream's change (2026-10-28) treats a source root within roundoff of the blocking bound as blocked. Its regression test is `tests/test_ast2018_source_bound_roundoff.py`, which passes. | physics-neutral bug fix |
| `fracture_fit/experimental_fits/ast2018/bulk_flow_fit.py` | Peierls-only bulk fit, now used only to supply a seed. | tooling |
| `fracture_fit/experimental_fits/ast2018/plot_per_temperature_joint_fit.py` | Per-temperature best-fit figure (J, nominal K, stiffness) for every variant. | diagnostics |
| `fracture_fit/tests/test_ast2018_per_temperature_joint_fit.py`, `tests/test_ast2018_signed_pileup.py`, `tests/test_ast2018_section_plasticity.py`, `tests/test_ast2018_implicit_source.py` | Regression tests. | tests |

The summary-figure scripts live with the run outputs:
`runs/ast2018_per_temperature_joint_v5/summary_figures.py`, `seeded/seeded_pileup.py` and
`build_page.py`.

### Fitted coordinates (family A, EXP-floor opening)

The seven coordinates are `G0_open_eV, Gf_open_eV, sigc_open_GPa, G0_emit_eV,
sigc_emit_GPa, G0_peierls_eV, sigc_peierls_GPa`.

- Box: lower `[2.0, .05, .5, .5, .5, .4, .3]`, upper `[6.5, 1.3, 4.0, 3.0, 4.5, 2.6, 5.0]`.
- `--emission-active` caps `G0_emit ≤ 1.6` and `sigc_emit ≤ 1.3`
  (`EMISSION_ACTIVE_HI`).
- Fixed shape parameters:
  - Opening: α = 0.485, n = 2.115.
  - Emission: floor 0.0687·G0, a = 0.278, n = 0.836.
  - Peierls: a = 1.144, n = 1.764, ν0 = 1e12 s⁻¹, stress fraction 1/√3.
  - Taylor: native H0 = 1.532 eV and S = −7.95 kB (not fitted); its σc is tied
    to the Peierls σc at ratio 1.
- Model B (finite-cutoff opening) was dropped after v3, where it collapsed
  onto family A's floor.

The J score stops at `MAX_EXTENSION_UM = 0.98`. The elastic bending table
does not go further, while measured cracks reach 1.4–1.6 µm. Load and
stiffness are scored only until the measured stiffness reaches the elastic
stiffness of a 0.98 µm crack.

## 4. Environment and commands

```bash
cd /Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30/fracture_fit
PY=/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python
export PYTHONPATH=..:. MPLCONFIGDIR=/private/tmp/ast-mpl OMP_NUM_THREADS=1 \
       AST2018_NATIVE_SOURCE=/private/tmp/ast2018_native_source
# fit:   T in {21,100,250,500}; workers; family A|B; optional flags
$PY -m experimental_fits.ast2018.per_temperature_joint_fit 250 8 A --signed-pileup
$PY -m experimental_fits.ast2018.per_temperature_joint_fit 500 8 A --emission-active
# tests
$PY -m pytest -q tests/test_ast2018_per_temperature_joint_fit.py tests/test_ast2018_signed_pileup.py \
    tests/test_ast2018_section_plasticity.py tests/test_ast2018_source_bound_roundoff.py \
    tests/test_ast2018_implicit_source.py
# 2026-11-04: 34 passed, 2 failed (both in test_ast2018_implicit_source.py; see section 6)
```

- **Native source.** `/private/tmp/ast2018_native_source` is a symlink to the
  durable clone `/Volumes/Data/Data/Nanopillar_calculation/ast2018_native_source_ce2bf9d4`
  (pinned commit `ce2bf9d4e868…`). macOS /tmp cleanup emptied the original
  tree on 2026-11-02. Its file hashes were checked against
  `runs/ast2018_21c_rate0p5_bending_fit_v1/source_manifest.json`. If the
  symlink disappears after a reboot, recreate it. The pin check fails closed.
- `OUTPUT` is `runs/ast2018_per_temperature_joint_v5` and `PREVIOUS` is
  `runs/ast2018_per_temperature_joint_v4`. Keep v4: v5 reads it. For any new
  campaign, bump both constants rather than writing into v5.
- `ProcessPoolExecutor` uses spawn, so scripts must be files, not stdin.
- Other Codex jobs share this machine. Never pattern-kill `spawn_main`
  workers; check the parent command first.
- On 2026-11-04 the data volume was 100% full. It now has about 7 GB free.
  Check `df -h /Volumes/Data` before long runs.

## 5. Results (v5; objective = weighted mean squared standardized residual)

| T | Native, emission off | Native, emission on (capped box) | Signed pile-up |
|---|---|---|---|
| 21 °C | 3.13 | 3.16 | **2.77**, essentially no emission |
| 100 °C | **12.68** | 16.59 | 12.68, identical to emission off; crack 0.14 µm |
| 250 °C | 16.9 | 10.91 | **6.92**; 10–14 activations, shielding 0.15–0.19 MPa√m; plateau load right; crack only about 0.3 µm |
| 500 °C | 2.12 | **0.68**; 726 emissions, tip radius +126 nm, interior optimum | 3.97; unstable fracture at flat J ≈ 80; search still drifting (`sigc_peierls` 4.93, at its bound) |

- Best vectors are in `runs/ast2018_per_temperature_joint_v5/<variant>/T{C}_summary.json`.
- The combined table is in `runs/ast2018_per_temperature_joint_v5/fitted_barriers.json`.
- Seeded pile-up reruns are in `.../v5/seeded/model_A_signed_pileup/`:
  - At 100 °C the seeded run returns to 12.68.
  - At 500 °C it sat on a flat no-crack plateau at 30.1.
- Figures (all temperatures) are in
  `/Volumes/Data/Data/Nanopillar_calculation/Ast2018_fit_figures_20261104/`.
  The results page is https://claude.ai/artifact/STEfQ7RUUzCBsoZcGYioXF.

### Interpretation so far

- **Shielding is nearly all-or-nothing under signed pile-up.**
  - Plentiful emission combined with slow Peierls glide pins the local K near
    1.45 MPa√m. That is about 3.5 MPa√m of shielding, and the crack never
    starts.
  - Otherwise shielding stays below about 0.5 MPa√m.
  - The search has no smooth path between the two regimes.
- **Emission is not identified at 21 or 100 °C.** At 2 GPa the emission
  barrier is 56 and 75 kT respectively; both fits sit at the lower bound
  of what the data allow. It is 37 kT at 250 °C and 29 kT at 500 °C.
- **The opening floor Gf sits at 17–18 kT at every temperature** (9 kT at
  500 °C under pile-up). That is ln(ν·t), set by the test time, not a
  material value. The apparent S* of about −16 kB for Gf is that artifact.
- **The Peierls H\* comes out negative** (−0.45 eV under pile-up; native S* ≈
  −37 kB). This is a compensating, unphysical trend. Do not report it as a
  material property.
- **The native emission S\* is about +15 kB.** It is meaningful only where
  emission is active, which here means 250 and 500 °C.
- **Open physics question, for the user to settle:** at 100 and 250 °C the
  crack loses its drive once the hinge yields. The measured J keeps rising
  to about 900–1000 N/m at 100 °C, about 7× the 500 °C value. Emission plus
  blunting alone gives a flat J of about 130.

## 6. Unresolved findings (verify before relying on v5 numbers)

1. **Two tests fail in `tests/test_ast2018_implicit_source.py`**, re-run on
   2026-11-04:
   - `test_warm_reduced_loading_has_no_step_count_emission_floor`: the local
     drive is 8.388 against an expected 8.4 (abs 1e-8).
   - `test_active_warm_plastic_displacement_is_step_stable`: `values[0] > 4e-7`
     does not hold.

   Both use `predict_reduced`, not the spatial or pile-up path. They passed
   when they were written on 2026-10-29. The only known change since then is
   the native source being re-cloned on 2026-11-02, from a /tmp tree whose
   exact state was not recorded. Determine the cause first. Do not loosen the
   tolerances.
2. **Provenance check against the re-cloned source: passed.** On
   2026-11-04, three stored bests were re-evaluated from scratch with the
   current source. All three reproduced their stored objectives to every
   digit:
   - model_A at 21 °C: 3.1310248752312257
   - emission-active at 500 °C: 0.6775619521962631
   - signed pile-up at 250 °C: 6.924545479514462

   The spatial forward path behind v5 is therefore unchanged by the re-clone.
   The check script is `runs/ast2018_per_temperature_joint_v5/rescore_check.py`.
   The failures in item 1 are confined to the `predict_reduced` path or the
   tests' own expectations, and they do not invalidate v5.

## 7. Files edited concurrently by other agents (do not overwrite)

- `config.py`, `spatial.py`, `mechanics.py` and `cantilever_loading.py`
  under `experimental_fits/ast2018/` also carry other sessions' work. Check
  their modification times and diff before editing.
- The `cantilever_*` modules and the many modified `experimental_fits/ast2018`
  and `arrhenius_fracture` files in `git status` belong to other workstreams.
- When committing this workstream, stage only the files in section 3, in
  separate commits:
  1. `implicit_source` fix and its test (physics-neutral bug fix).
  2. `section_plasticity` and the `signed_pileup` option with their tests
     (opt-in physics options). For `spatial.py`, use `git add -p` to take
     only the `TIP_INTERACTIONS` hunks.
  3. Fit and plot tooling.

## 8. Next work, in priority order

1. Resolve the two `implicit_source` test failures (section 6, item 1).
   Find out whether `predict_reduced` changed or the expected values were
   wrong. Add a regression test for whichever defect it turns out to be.
2. **500 °C pile-up.** The best point sits against `sigc_peierls` = 5.0, and
   the seeded starts landed on the over-shielded plateau. Seed from points
   with weaker emission or faster glide, placed between the no-crack plateau
   and the unstable-fracture regime. If the optimum stays at the bound, report
   it as bound-limited (‡). Widening a bound is a modelling decision: ask the
   user first.
3. **The 0.98 µm representation limit.** The elastic bending opening table
   stops before the measured crack lengths of 1.4–1.6 µm. This truncates the
   data that would discriminate the 100 and 250 °C fits. Extending the table,
   using the existing `cantilever_elastic_2d` route, changes only the
   numerical representation, not the physics. Validate it against the
   current table on the overlapping range.
4. Present the hinge-yield and crack-drive question (section 5) to the user,
   with the diagnostics, before changing any physics.
5. Only after emission is active and inside the box at three or more
   temperatures, extract S* per barrier with a line fit of G(T) at fixed
   stress, as in `summary_figures.py`'s table. Flag the Gf ln(ν·t) artifact.
6. Keep each new campaign in a new `runs/ast2018_per_temperature_joint_v6/`
   with a README stating what changed. Mark superseded folders with
   `SUPERSEDED.txt`.

## 9. Completion contract for this workstream

Do not claim a fit result unless all of the following hold:

- The tests in section 4 pass, with no loosened tolerance.
- Each reported best has a complete `T{C}_summary.json` and an evaluations
  JSONL, and its source hash matches the current fit module.
- The tip state is reported, and a fit with emission switched off is not
  presented as the answer.
- Any parameter within 2% of a box limit is flagged.
- `git diff --check` is clean for the files you staged.
