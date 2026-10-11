# Fracture calculation readiness and first storage-safe batch

> Supersession note (2026-10-10): this file records the earlier readiness
> assessment and mixes retained historical launchers with then-current paths.
> The controlling current-source/historical separation for the bounded science
> pilots is `review_packet/current_science_pilots_6abbf10/EXECUTION_MAP.md` and
> its machine-readable `PILOT_PROTOCOL.json`. In particular, the old V10.2.28
> and `ed53799f` rows below are retained provenance, not the current integrated
> executors selected for the new calculations.

This is an execution map, not campaign authorization. It preserves qualified
paths and separates paper-ready calculations from cyclic and combined-mode
development.

| Paper need | Exact executable/source | Material and controls | Mechanics / numerical policy | Observer, restart, retained output | Status |
|---|---|---|---|---|---|
| Developed single-front propagation and R-curve loading/reloading | `scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh`; `arrhenius_fracture.sharp_front_v10_2_28_audited` | Four registered `v913_paper_*_persistent_sites` rows; `THETA=30`; explicit temperature list and hazard seed | Direct prescribed-geometry full FEM kernel, 1 mm path target, 5 um physical increments, signed evolution; no reduced trajectory | Native step/event histories, `scripts/export_developed_geometry_observations.py`, `scripts/query_k_at_crack_length.py`; incomplete-run restart; 20 selected snapshots | Ready; existing 1.005 mm accepted history retained and must not be rerun for source-label uniformity |
| Temperature/rate/orientation controls | `scripts/run_v10_1_7_4_orientation_temperature_sweep.sh` and `scripts/analyze_v10_1_7_4_orientation_temperature_sweep.py` | Explicit rate, temperature, crystal orientation, class, and seed inputs | Original spatial sharp-front PF/FEM field path at its recorded source identity | Native K/J/event tables and selected field stages; C19 full-PF tables/figures already assembled | Ready for existing qualified grid; any new point is a new authorized case |
| Two-front theta-40 continuation | `arrhenius_fracture.sharp_front_current_source_multifront_physical_v12.main` | 700 K retained theta-40 family; separately owned front clocks | Shared full FEM/stress state, native two-front scheduler and transfer | V12 checkpoint plus byte-exact interrupted/uninterrupted comparison under `review_packet/targeted_completion_b927dac/theta40/` | Ready for its qualified continuation scope |
| Physical cavity/fracture comparison | Historical `scripts/run_paired_monotonic_fracture_v4.py` at launch commit `a2a90cd`; protocol identity `c248c959...` | DBTT, 1200 K, seed 3621, 6.180--6.300 um opening at 1 um/s, matched enabled/suppressed ancestry | Original full coupled FEM + full RT0/P0 transport + topology transactions | Native checkpoint/event owners, K(a), reaction/work, cavity ledgers, C19 Figure 8; 740 MiB retained run | Ready as a finite historical full-order result; not relabeled as new-flux output |
| Corrected-transport single-front cavity path | `arrhenius_fracture.shared_v5_cavity_adapter.main` with `MIXED_RT0_P0_CONSERVATIVE_POSITIVE_FLUX_CORRECTION` | Tip-only, pure-bulk, and tip-plus-bulk source geography remain independent of source lifecycle | Full mechanics and corrected RT0/P0 transport; upper bound fail-closed | Native checkpoint/restart and transport/capture ledgers | Numerically qualified for tested transport scope; retained 1.00002 s pair is a censored no-event result, not the physical cavity figure |
| Original FEM/CZM comparison | Historical repository `Arrhenius_FEM_CZM_MPZ_v9_12_emergent_gnd_dbtt`, bound source commit `ed53799f...`; C19 assembler consumes portable rows | Row-owned rate, temperature, class and seed | Original full FEM/CZM; no replacement by current sharp-front or reduced mechanics | Portable result tables and C19 Figures 9--10 | Reader-facing comparison ready; exact producer commit is not embedded in every old result row and remains an archival limitation |
| Operator-consistent V3.6 cyclic restart | `scripts/run_v3_6_operator_consistent_developed_fatigue_v10230.py` | Registered loads 4.7817625, 8.5987384, 15.4624747 MPa sqrt(m); material/barrier row resolved by the script | Native operator-consistent projection and material-frame patch history | `v10.2.30_v3_6_material_frame_checkpoint_v2` generation + manifest | Restart mechanism ready; only low-activity evolution demonstrated (`max |beta_p|=2.56e-40` in the short high-load phase). Not an active-history qualification |
| Branching plus cavity | No shared executor yet; independent theta-40 multifront and single-front cavity paths remain intact | Branch permission must stay independent of graph/cavity downstream rules | Requires shared full mechanics/transport, separate clocks, event ordering, overlap/remesh transfer | Required adapter test must checkpoint all front/cavity owners | Genuinely new supported composition; independent workstream, not a blocker for current monotonic paper figures |

## First batch after explicit campaign authorization

The first batch should reuse existing kernels and suppress repeated rendered
fields while preserving native mechanics observations and restart:

1. one DBTT theta-30 1 mm propagation/control case using the locked V10.2.28
   launcher, `SAVE_SNAPSHOTS=20`, native K/J-at-length and loading/reloading;
2. one matched registered-material control at the same kernel/loading policy;
3. analysis-only regeneration of the selected original FEM/CZM comparison from
   portable rows (no raw FEM rerun); and
4. reuse, without copying, of the recovered V11 cavity pair and its selected
   full-order field stages.

The retained 1 mm example occupies 65 MiB, the complete V11 pair 740 MiB, the
theta-40 probe 7.4 MiB, and all current positive-transport evidence 73 MiB. A
conservative allowance of 500 MiB for two new long histories plus temporary
checkpoint overlap is small relative to the measured 17 GiB free on
`/Volumes/Data`. Production output should remain under a new run root, retain
only milestone fields plus rolling/native restart generations, and avoid
duplicate image sequences. No existing trajectory is deleted or overwritten.

The read-only `/Volumes/My Passport` is not a writable backup destination.
Campaign launch and external duplication remain pending; neither condition is
treated as deletion authority.

## Executable pending work

- Active V3.6: resume from an actually populated native checkpoint after its
  preceding loading history, then compare interrupted/uninterrupted evolution
  with a non-negligible material-frame state. Do not prescribe slip or alter a
  barrier.
- Combined mode: add one bounded adapter test using an admissible existing
  two-front state and an independently owned cavity state, with shared mechanics
  and transport, deterministic event ordering, remesh transfer, and restart.
  Rare spontaneous branch birth is not required for this compatibility test.
