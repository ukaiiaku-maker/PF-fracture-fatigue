# Targeted physical completion

Numerical source `b927dac18c58f2fb896fe3520fb4c23ba950ec7e` closes the
accepted-state theta=40 continuation defect, extends the physical V5 pair to a
declared model-validity boundary, and restores native path arclength to the
developed-growth observations. No parameters were refit, no reliability test
was weakened, and no scientific data were deleted.

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
  state to 0.50002 s. The intended 1.00002 s censor could not be reached: the
  next interval hits the native vacancy-concentration bound even after shared
  refinement below `1e-8 s`. At the last accepted endpoint there is no cavity
  event; both sites remain `AVAILABLE_SITE`, with maximum birth-clock fraction
  `3.823191651998431e-6`. Enabled and disabled shared mechanical, plastic,
  damage, vacancy, reaction, and energy quantities are identical at numerical
  precision. This is a censored no-event result, not a claim of no longer-time
  cavity influence.
- The 300 K trajectory remains unchanged at 2,085 accepted mechanics rows,
  201 events, and 1002.9028827375072 µm projected extension. Its 201 native
  event segments recover 1005.0000000000018 µm of path arclength for all 2,286
  stage-aware observations. Initial/repeated/terminal queries and a 1002 µm
  discrete-jump bracket are retained with a path-coordinate K plot.
- The earlier base/current mechanics comparison is retained with its bounded
  scope: native fallback and one-event histories match, while the intentional
  activated P0 correction is not required to be neutral.
- Unsupported V10.4 bulk and V3.6 operator-consistent public restart requests
  now fail before execution rather than silently beginning a fresh run.

## Verification and evidence

The complete repository suite passed 777/777 tests in the qualified Python
environment. The separate external FEM/CZM 4/4 result was not rerun because
neither its source nor adapter interface changed. The selected sub-3 MiB packet
under `review_packet/targeted_completion_b927dac/` contains the claim index,
receipts, comparisons, compressed numerical tables, plot, and checksums.

Only `/Volumes/Data` was mounted. Newly generated full checkpoints and
histories therefore still have one local copy; the packet records their exact
paths and hashes without claiming an external backup.

## Genuinely pending

Native public load-and-continue remains to be implemented for V10.4 bulk and
V3.6 operator-consistent execution. The combined branching-plus-cavity case
was not run, and the new full artifacts still require an external duplicate.
These limitations do not invalidate the completed single-front, V5 paired, or
two-front restart evidence.
