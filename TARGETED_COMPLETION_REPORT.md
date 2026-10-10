# Targeted physical completion

Numerical source through the native-restart/transport continuation preserves
the accepted theta=40 and long-growth results, implements the two remaining
public restart paths, and identifies the V5 stop as a numerical positivity
failure rather than a physical dilute-concentration limit. No parameters were
refit, no reliability test was weakened, and no scientific data were deleted.

## Results

- The exact theta=40 family (`adb775...` model fingerprint;
  `423bc3...` file hash) now completes the inherited two-front update. The
  defect was a stale process-local tensor-drive serial combined with a
  noncanonical accepted-state fingerprint—not a bad tensor probe. A clean
  committed-source uninterrupted/restarted comparison ends at step 2184,
  13935.60007250726 s, and 410.0000000000013 µm total extension after a 5 µm
  accepted event. All six FEM/stress arrays are byte-exact and every
  restart-owned runtime category is identical.
- The enabled/suppressed 1200 K V5 pair was continued from the common native
  state through 0.52002 s. The attempted 0.50002--0.60002 s step fails the
  lower positivity test in two transport cells: the first linear candidate
  reaches about `-9.76e-15`, while its maximum is `3.54e-9`, far below the
  `1e-2` dilute upper limit. Symmetric equilibration reduces the linear
  residual from about `1.14e-15` to `2.08e-24` but reproduces the negative
  values, identifying a centered mixed-discretization positivity defect rather
  than solver roundoff or a physical upper-domain crossing. A 0.01 s step is
  admissible; 0.01, 0.005, and 0.0025 s paths to 0.52002 s show first-order
  state convergence. At the accepted endpoint there is no cavity
  event and both sites remain `AVAILABLE_SITE`. At the previously reported
  0.50002 s endpoint the maximum birth-clock fraction was
  `3.823191651998431e-6`. Enabled and disabled shared mechanical, plastic,
  damage, reaction, and energy quantities remain paired. This is accepted
  no-event exposure followed by an unresolved numerical transport stop, not a
  physical model-domain limit or a claim of no longer-time cavity influence.
- V10.4 pure-bulk and tip-plus-bulk now use the existing complete combined
  outer/kinetic checkpoint through the public interface. Bounded fresh-process
  comparisons retain identical mesh, damage, displacement, plastic strain,
  density, process-zone populations, hazard and RNG state through restart.
  Pure bulk has nonzero full-field plastic evolution and zero tip population;
  tip-plus-bulk retains both nonzero bulk plastic strain and tip population.
  Material-manifest and source-geography identities now fail closed on restore.
- V3.6 operator-consistent execution now writes and loads an atomic checkpoint
  after every accepted cyclic phase interval. A two-boundary comparison through
  six nonzero phase intervals is exact for the native state, material patches
  and swept-slip ledger, hazard, pending exponential mark, RNG, cycle/phase,
  acceleration history, and cycle snapshots.
- The retained original-model lifecycle fixture grows a cavity from 25 to
  55 micrometres, remeshes twice, conserves vacancy inventory, and survives a
  checkpoint/re-equilibration round trip. It remains explicitly labeled an
  extreme-reservoir functional fixture, not an intact-state nucleation
  prediction. A combined branching-plus-cavity calculation remains unresolved.
- The 300 K trajectory remains unchanged at 2,085 accepted mechanics rows,
  201 events, and 1002.9028827375072 µm projected extension. Its 201 native
  event segments recover 1005.0000000000018 µm of path arclength for all 2,286
  stage-aware observations. Initial/repeated/terminal queries and a 1002 µm
  discrete-jump bracket are retained with a path-coordinate K plot.
- The earlier base/current mechanics comparison is retained with its bounded
  scope: native fallback and one-event histories match, while the intentional
  activated P0 correction is not required to be neutral.
- Unsupported restart families still fail before execution; V10.4 bulk and
  V3.6 have moved from that list to native load-and-continue families.

## Verification and evidence

The complete repository suite passed after the restart/diagnostic changes in the qualified Python
environment. The separate external FEM/CZM 4/4 result was not rerun because
neither its source nor adapter interface changed. The selected sub-3 MiB packet
under `review_packet/targeted_completion_b927dac/` contains the claim index,
receipts, comparisons, compressed numerical tables, plot, and checksums.

Only `/Volumes/Data` was mounted. Newly generated full checkpoints and
histories therefore still have one local copy; the packet records their exact
paths and hashes without claiming an external backup.

## Genuinely pending

The V5 transport operator still needs a positivity-preserving conservative
full-order discretization before exposure beyond the numerical stop can be
interpreted. The combined branching-plus-cavity case was not run. These
limitations do not invalidate the completed V10.4/V3.6 restarts, bounded cavity
fixture, single-front, accepted V5 prefix, or two-front restart evidence.
