# R10 V2-P25 developed completion and strict-Family-A acquisition

This receipt closes the two accepted V2-P25 checkpoint chains without changing the material, seed, loading, phase resolution, mechanics, event law, or numerical acceptance controls. Both resumed cases reached the declared target of at least 12 events and 70 um extension; each finished with 12 events and 80.545077706 um extension.

## Matched developed fatigue result

All reported rates use the same event window: the first accepted event at or beyond 20 um through the first accepted event at or beyond 60 um. Startup averages and the former 3.339 startup secant are excluded.

| DeltaK (MPa sqrt(m)) | Developed rate (m/cycle) | Final cycles | Final events |
|---:|---:|---:|---:|
| 4.144909057 | 1.106021315e-10 | 739176.022960 | 12 |
| 9.657580495 | 1.773782828e-08 | 4608.030925 | 12 |
| 18.213077452 | 1.462473483e-07 | 558.921756 | 12 |

The developed low-to-middle slope is 6.002768361, the middle-to-high slope is 3.325363634, the endpoint/global slope is 4.855309192, and a three-point log-log regression gives 4.917334012. The developed rate width is 3.121324504 decades. Thus the row supplies a useful amplitude-dependent benchmark, but its two adjacent intervals do not both satisfy the inherited 3.4--4.6 moderate-slope band.

The middle and high phase-exact cases retain failed mean-cycle overlap diagnostics. That is expected evidence for their exact-cycle execution path, not a reason to substitute the mean approximation; their accepted event histories and completion status come from the unchanged phase-exact engine.

## Identity and evidence scope

`P25_TJBSV2_S_026704` remains `B_INHERITED_V2_GUARD_ENTROPY_EXTENSION`, complete-row hash `85f8d9d2ddbe1e7c7c0c34589883b886023507089d0fe423c1ce2c4eb60d0041`. Its fracture span and sealed 300/1200 K 2-D reaches are retained as historical receipts. They are not relabeled current evidence and do not establish strict-Family-A success.

The completed benchmark's A0-to-current offset is nearly constant at the three amplitudes (`log10(current/A0)` = -5.30484, -5.30533, and -5.40542). The fitted acquisition transfer exponent is 0.973775927 with 0.0315202 decade RMS error. This transfer is explicitly a ranking surrogate, not a constitutive replacement.

## Strict-Family-A acquisition

All 16 acquisition rows descend directly from original Weak-T row `v913_zeroD_sobol_0129902`. The P25/P40 designs contribute only their 300 K opening shapes. Their previously projected thermal slopes are discarded and the shapes are re-anchored to the original Weak-T `gT` and `sT`. The original Peierls and Taylor activation entropies and all mechanics/state laws remain unchanged. No cold/hot endpoint is independently selected or fitted.

The leading opening candidate is `R10A_WEAKT_P40_OPEN_P036MEV_V1`. It has corrected A0 global slope 3.901544803, corrected rate width 3.877455798 decades, and frozen 300 K onset 5.22 MPa sqrt(m). The matched dislocation-barrier bracket is:

- `R10A_WEAKT_P40_EMIT_M060MEV_V1`
- `R10A_WEAKT_P40_EMIT_P060MEV_V1`

These rows are executable strict-Family-A hypotheses, not qualified materials. They require same-row full-state fracture and developed fatigue before any promotion; new 2-D work remains gated behind those 1-D results.

## Executed joint gate

The three selected hypotheses were subsequently executed through the decisive 300/450 K full-state fracture screen with the unchanged current backend:

| Candidate | K-init at 300 K (MPa sqrt(m)) | 450 K result | Span lower bound |
|---|---:|---|---:|
| `R10A_WEAKT_P40_OPEN_P036MEV_V1` | 5.237397 | no event through K=80 | 15.2748 |
| `R10A_WEAKT_P40_EMIT_M060MEV_V1` | 5.242302 | no event through K=80 | 15.2605 |
| `R10A_WEAKT_P40_EMIT_P060MEV_V1` | 5.233339 | no event through K=80 | 15.2866 |

All three fail the Weak-T fracture gate. The 450 K terminal hazard actions are approximately `2.22e-8`, leaving essentially the complete unit mark pending; these are physical K-endpoint censors, not work-budget stops. Their remaining 600--1200 K panels were intentionally stopped after this decisive result. Same-row fatigue and 2-D were not run because the established workflow forbids those expensive stages after joint 1-D fracture rejection.

The updated feasible set therefore contains no strict-Family-A Weak-T success. Family-B V2-P25 remains the completed benchmark at its stated scope. The next analytical acquisition must require 450 K opening-action accessibility before another full-state launch and must use lawful coupled thermal/nucleation/motion coordinates rather than independently fitted endpoints.

## Reproduction and contents

The portable tree includes the three complete trajectory results, their final checkpoint state and history, exact source registries, the analytical screen and projection frontier, completed slope tables, strict-Family-A candidate records, source scripts, and focused tests. `MANIFEST.json` records SHA-256 hashes and sizes for the packaged inputs.

Run the focused invariant tests with:

```bash
python -m pytest -q tests/test_r10_family_a_discrepancy_acquisition.py
```

They verify that the acquisition changes preserve original thermal slopes and Peierls/Taylor activation entropies, including the lawful 300 K P40 opening-level shift.
