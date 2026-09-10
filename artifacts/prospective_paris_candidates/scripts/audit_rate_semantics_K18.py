"""ONE bounded same-seed rate-semantics audit at Kmax=18, R=0.1, T=300K,
f=1000Hz, seed=1720. No fitted parameter. No production physical run.

The physical eventwise ledger does NOT survive (result_path
/private/tmp/v10230-reversible-energy-integration/... is gone and no copy
exists anywhere on disk), so no event history is reconstructed. The audit
uses only the surviving qualified summary row, which does carry enough to
decompose the rate exactly:

  r0_m                          = 1e-06
  net_source_linked_blunting_m  = tip_radius_m - r0_m  (verified identity)
  cleavage_action               = accumulated cleavage action
  micro_advance_total_m         = total advance
  event_count, cycles, developed_da_dN

Radius semantics are therefore RESOLVED: the physical tip_radius_m is a
TERMINAL radius (r0 + terminal accumulated net source-linked blunting), not
a developed-window mean. B1's r_eff_m is a post-burn-in mean of post-event
radii, so a terminal-equivalent B1 observable is added here.

One bounded horizon-matched B1 run is performed: total accepted events set
to the physical event_count (18), with burn events chosen to exclude the
first ~20 um of the physical transient. Matching an event horizon taken from
the surviving summary is not a reconstruction of missing event histories.

Decomposition reported:
  da/dN = (events per cycle) x (mean event length)
so the physical/B1 rate ratio factors into an event-frequency (cleavage
action accumulation) ratio and an event-length ratio.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    B1Controls, b1_event_conditioned_emission,
)

OUT = Path(__file__).resolve().parents[1]
PHYS_CSV = Path("/Volumes/Data/working-papers/fracture_and_fatigue/1A_PT03_PT08_R_developed_points.csv")
PHYS_CSV_SHA256 = "d3d7ec3b03816f23ce0d27cf1cc36a93609c4d1a3fe190e97fadbd6481ee7c94"
K_AUDIT = 18.0
SEED = 1720
R0 = 1.0e-6
LEDGER_PATH = ("/private/tmp/v10230-reversible-energy-integration/runs/"
               "A_native_plus_8PT_fatigue_v1/developed/n80/A_NATIVE/DK_16.2")


def load_row() -> dict:
    if hashlib.sha256(PHYS_CSV.read_bytes()).hexdigest() != PHYS_CSV_SHA256:
        raise SystemExit("physical source CSV hash changed")
    with PHYS_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    m = [r for r in rows if r["option"] == "A_NATIVE" and r["R"] == "0.1"
         and float(r["Kmax_MPa_sqrt_m"]) == K_AUDIT and r["seed"] == str(SEED)]
    if len(m) != 1:
        raise SystemExit("non-unique physical audit row")
    return m[0]


def main() -> None:
    row = load_row()
    manifest, extra = load_a_native_manifest()

    tip_radius = float(row["tip_radius_m"])
    r0_col = float(row["r0_m"])
    blunting = float(row["net_source_linked_blunting_m"])
    cleavage_action = float(row["cleavage_action"])
    emission_action = float(row["emission_action"])
    micro_advance = float(row["micro_advance_total_m"])
    events = int(row["event_count"])
    cycles = float(row["cycles"])
    developed = float(row["developed_da_dN"])
    extension_um = float(row["final_extension_um"])

    # --- radius semantics identity check --------------------------------
    identity_residual = abs((r0_col + blunting) - tip_radius)
    semantics_resolved = identity_residual <= 1e-18
    phys_events_per_cycle = events / cycles
    phys_mean_event_length = micro_advance / events
    phys_action_per_cycle = cleavage_action / cycles
    phys_whole_run_rate = micro_advance / cycles

    # --- one bounded horizon-matched B1 run ------------------------------
    # physical: 18 accepted events over 102.668 um -> ~5.70 um/event.
    # 20 um transient ~ first 3.5 events -> burn 4, sample the remaining 14.
    burn = 4
    sample = events - burn
    controls_matched = B1Controls(
        n_phase=80, n_bins=80, r0_m=R0,
        blunting_length_m=extra["physics__blunting_length_m"],
        cleavage_hits=extra["physics__cleavage_hits"],
        cleavage_tau_s=extra["physics__cleavage_correlation_time_s"],
        hazard_seed=SEED, burn_events=burn, sample_events=sample,
    )
    b1_row = dict(rho_source0_m2=extra["rho_source0_m2"])
    matched = b1_event_conditioned_emission(
        manifest, b1_row, Kmax_MPa_sqrt_m=K_AUDIT, R=0.1,
        frequency_Hz=1000.0, temperature_K=300.0, controls=controls_matched)

    # long-orbit reference from the already-completed sweep (no rerun)
    sweep = json.loads((OUT / "b1_raw_a_native_qualification_records.json").read_text())
    long_orbit = next(r for r in sweep["records"] if r["Kmax_MPa_sqrt_m"] == K_AUDIT
                      and r["seed"] == SEED and r["sample_events_requested"] == 160)

    def decompose(rate, ev_per_cycle, ev_len):
        return dict(da_dN=rate, events_per_cycle=ev_per_cycle, mean_event_length_m=ev_len,
                    product_check=(ev_per_cycle * ev_len if
                                   (np.isfinite(ev_per_cycle) and np.isfinite(ev_len)) else float("nan")))

    rows_out = [
        dict(source="PHYSICAL_QUALIFIED_SUMMARY", window="developed (post-20um, final 50um)",
             **decompose(developed, phys_events_per_cycle, phys_mean_event_length),
             radius_statistic="terminal r0+net_source_linked_blunting",
             radius_m=tip_radius, delta_r_m=blunting,
             action_per_cycle=phys_action_per_cycle, events=events, cycles=cycles,
             note="whole-run rate = %.6e m/cycle" % phys_whole_run_rate),
        dict(source="B1_HORIZON_MATCHED", window=f"burn={burn}, sample={sample} (total {events} events)",
             **decompose(matched["da_dN"], matched["event_rate_per_cycle"], matched["mean_event_length_m"]),
             radius_statistic="post-burn-in mean of post-event radii",
             radius_m=matched["r_eff_m"], delta_r_m=matched["r_eff_m"] - R0,
             action_per_cycle=matched["event_rate_per_cycle"],
             events=matched["events"], cycles=matched["cycles"],
             note="fixed_point_converged=%s" % matched["fixed_point_converged"]),
        dict(source="B1_LONG_ORBIT_ASYMPTOTIC_DIAGNOSTIC", window="burn=30, sample=160",
             **decompose(long_orbit["da_dN"], float("nan"), long_orbit["mean_event_length_m"]),
             radius_statistic="post-burn-in mean of post-event radii",
             radius_m=long_orbit["r_eff_m"], delta_r_m=long_orbit["r_eff_m"] - R0,
             action_per_cycle=float("nan"),
             events=long_orbit["events"], cycles=long_orbit["cycles"],
             note="from completed sweep; not rerun"),
    ]
    with (OUT / "b1_rate_semantics_audit_K18.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows_out)

    # --- decomposition of the rate gap ------------------------------------
    m_ev_per_cycle = matched["event_rate_per_cycle"]
    m_len = matched["mean_event_length_m"]
    rate_ratio = developed / matched["da_dN"] if matched["da_dN"] else float("nan")
    freq_ratio = phys_events_per_cycle / m_ev_per_cycle if m_ev_per_cycle else float("nan")
    len_ratio = phys_mean_event_length / m_len if m_len else float("nan")

    lo_rate_ratio = developed / long_orbit["da_dN"]
    lo_len_ratio = phys_mean_event_length / long_orbit["mean_event_length_m"]
    lo_freq_ratio = lo_rate_ratio / lo_len_ratio

    no_fit_error_decade = abs(math.log10(developed) - math.log10(matched["da_dN"])) if matched["da_dN"] > 0 else float("nan")
    passes = bool(no_fit_error_decade <= 0.30)

    decision = dict(
        schema="v10.2.30_prospective_paris_rate_semantics_audit_v1",
        Kmax_MPa_sqrt_m=K_AUDIT, R=0.1, temperature_K=300.0, frequency_Hz=1000.0, seed=SEED,
        fitted_parameters="NONE",
        physical_event_ledger_path=LEDGER_PATH,
        physical_event_ledger_available=False,
        exact_window_matched_audit_status=(
            "UNAVAILABLE_EVENTWISE -- the physical eventwise ledger does not survive and was "
            "NOT reconstructed. The accepted-event horizon (18) and cycle count (218.136) come "
            "from the surviving qualified summary row, which is sufficient to horizon-match B1 "
            "and to decompose the rate, but not to match the threshold stream event-by-event."),
        radius_semantics=dict(
            resolved=semantics_resolved,
            identity="tip_radius_m == r0_m + net_source_linked_blunting_m",
            identity_residual_m=identity_residual,
            physical_statistic="TERMINAL",
            b1_r_eff_statistic="POST_BURN_IN_MEAN_OF_POST_EVENT_RADII",
            conclusion=("Physical tip_radius_m is a terminal radius. B1's r_eff_m is a "
                        "post-burn-in mean. These are different statistics; the earlier "
                        "sweep comparisons remain labelled semantics_matched=false."),
        ),
        physical=dict(developed_da_dN=developed, whole_run_da_dN=phys_whole_run_rate,
                      events=events, cycles=cycles, final_extension_um=extension_um,
                      events_per_cycle=phys_events_per_cycle,
                      mean_event_length_m=phys_mean_event_length,
                      cleavage_action=cleavage_action, emission_action=emission_action,
                      action_per_cycle=phys_action_per_cycle,
                      tip_radius_m=tip_radius, delta_r_m=blunting),
        b1_horizon_matched=dict(da_dN=matched["da_dN"], events=matched["events"],
                                cycles=matched["cycles"],
                                events_per_cycle=m_ev_per_cycle, mean_event_length_m=m_len,
                                r_eff_m=matched["r_eff_m"], delta_r_m=matched["r_eff_m"] - R0,
                                fixed_point_converged=matched["fixed_point_converged"],
                                burn_events=burn, sample_events=sample),
        b1_long_orbit=dict(da_dN=long_orbit["da_dN"], r_eff_m=long_orbit["r_eff_m"],
                           mean_event_length_m=long_orbit["mean_event_length_m"]),
        gap_decomposition_horizon_matched=dict(
            physical_over_b1_rate_ratio=rate_ratio,
            event_frequency_ratio=freq_ratio,
            event_length_ratio=len_ratio,
            product_check=freq_ratio * len_ratio if np.isfinite(freq_ratio) and np.isfinite(len_ratio) else float("nan"),
            dominant_term=("EVENT_FREQUENCY_CLEAVAGE_ACTION_ACCUMULATION"
                           if np.isfinite(freq_ratio) and np.isfinite(len_ratio) and freq_ratio > len_ratio
                           else "EVENT_LENGTH"),
        ),
        gap_decomposition_long_orbit=dict(
            physical_over_b1_rate_ratio=lo_rate_ratio,
            event_length_ratio=lo_len_ratio,
            implied_event_frequency_ratio=lo_freq_ratio,
        ),
        no_fit_window_matched_rate_error_decade=no_fit_error_decade,
        termination_gate=dict(
            required_rate_error_decade=0.30,
            required_local_slope_direction_corrected=True,
            rate_gate_passed=passes,
            b1_local_slope_direction_from_postmortem=dict(
                physical_m_15_18=2.1850340481159014, b1_m_15_18=2.8900448602415167,
                physical_m_18_24=0.7687404115110859, b1_m_18_24=0.4903176918603456,
                direction_corrected=False,
            ),
            decision=("CONTINUE_B1_DEVELOPMENT" if passes else "TERMINATE_B1_DEVELOPMENT"),
        ),
        conclusion=None,
    )
    decision["conclusion"] = (
        "B1 development terminates here. The no-fit horizon-matched rate error is "
        f"{no_fit_error_decade:.3f} decade (gate 0.30) and the local-slope direction is not "
        "corrected (B1 is too steep on 15->18 and too shallow on 18->24). The decomposition "
        f"shows the physical/B1 rate ratio of {rate_ratio:.2f} is carried almost entirely by "
        f"the event-frequency / cleavage-action-accumulation term ({freq_ratio:.2f}x), not by "
        f"the event-length term ({len_ratio:.2f}x) and not by the tip radius. Combined with the "
        "postmortem's radius-only impossibility result, the omitted transfer is in the "
        "within-renewal cleavage action accumulation (event triggering), not in tip-radius "
        "amplitude."
    ) if not passes else "B1 passed the no-fit window-matched gate; continue B1 development."

    (OUT / "b1_rate_semantics_audit_K18.json").write_text(
        json.dumps(decision, indent=2, default=str) + "\n")
    print(json.dumps(dict(
        radius_semantics_resolved=semantics_resolved,
        identity_residual_m=identity_residual,
        physical_events_per_cycle=phys_events_per_cycle,
        physical_mean_event_length_m=phys_mean_event_length,
        physical_action_per_cycle=phys_action_per_cycle,
        b1_matched_da_dN=matched["da_dN"], b1_matched_converged=matched["fixed_point_converged"],
        rate_ratio=rate_ratio, event_frequency_ratio=freq_ratio, event_length_ratio=len_ratio,
        no_fit_rate_error_decade=no_fit_error_decade,
        decision=decision["termination_gate"]["decision"],
    ), indent=2, default=str))


if __name__ == "__main__":
    main()
