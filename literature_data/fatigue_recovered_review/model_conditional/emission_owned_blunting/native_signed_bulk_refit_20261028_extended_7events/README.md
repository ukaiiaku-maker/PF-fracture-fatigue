# Extended 7-event native matrix (audit continuation, 2026-10-28)

Same material fit as the accepted matrix (native_signed_bulk_refit_20261028/fit.json,
sha256 f93696bb...), same CLI flags, --events 7, new output root.

**Code identity differs from the accepted matrix.** scripts/run_qste340tm_main_result_full_2d.py
was modified at 21:14 by a concurrent session before this launch (sha256 now
3722e958..., handoff binding 22d82fdb...). The change passes --L-pz as the MPZ length
(1 um) instead of the hard-coded 5.557665 um; receipts record driver_L_pz_m=1e-6.
First events reproduce the accepted matrix exactly in 16/16 cases; later events differ.
Treat waits 1-2 here as an L_pz sensitivity of the accepted result, not a replay.

Analysis: scripts/analyze_qste340tm_signed_bulk_extended_events.py -> EXTENDED_ANALYSIS.json,
extended_event_windows.png.
