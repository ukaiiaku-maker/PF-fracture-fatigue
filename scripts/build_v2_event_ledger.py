"""Build a portable, tracked event ledger for the crack-rebonding causal
pilot v2, so the verifier no longer depends on the gitignored
runs/.../trajectories.json.

Reads the full trajectories.json (written by
run_v10_2_30_crack_rebonding_causal_pilot_v2.py, which lives under runs/
and is not git-tracked, per the same convention S8/v1 used for physical run
output) and extracts, per trajectory and per event, every field needed to
independently reproduce the mission's hard gates plus the mission's
requested per-event diagnostics: pre-event p_B, pre-event and maximum
phase-resolved K_rebond, cleavage-action-weighted K_rebond, accepted event
length, relevant MPZ-state and RNG/threshold identifiers, and
barrier-floor/cooperative-saturation diagnostics (the last one computed
directly from the frozen config, not from the trajectory data, since it is
a pure function of the config and needs no replay).

Deliberately drops each event's full bulk_action_records list (the
per-bisection-iteration S8C certificates) -- those are reproducibility
detail for the run itself, not gate inputs; the aggregated fields already
extracted per event (all_bulk_action_qualified,
max_phase_resolved_K_rebond_Pa_sqrt_m, action_weighted_K_rebond_Pa_sqrt_m,
cleavage_action) are what the gates and causal comparison actually need,
and dropping the raw list keeps the tracked ledger a reasonable size.

Usage:
    <pinned interpreter> scripts/build_v2_event_ledger.py \\
        --run-root runs/crack_rebonding_causal_pilot_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"

EVENT_FIELDS = [
    "event_index", "blocks_to_fire", "waiting_time_s_this_event", "cumulative_time_s",
    "accepted_length_m", "cumulative_extension_m",
    "pre_event_max_pB", "pre_event_K_rebond_Pa_sqrt_m",
    "max_pB_post_commit", "max_K_rebond_post_commit_Pa_sqrt_m",
    "max_phase_resolved_K_rebond_Pa_sqrt_m", "action_weighted_K_rebond_Pa_sqrt_m",
    "cleavage_action", "hazard_threshold_action", "hazard_event_index", "engine_id",
    "any_bulk_action_used", "all_bulk_action_qualified",
]
MPZ_STATE_FIELDS = [
    "mpz_mobile_count", "mpz_retained_count", "mpz_emitted_total", "mpz_escaped_total",
    "mpz_recovered_total", "r_eff", "sigma_tip", "mpz_total_K_shield_Pa_sqrt_m",
]


def _config_from_frozen(frozen: dict, key: str) -> CrackRebondingControls | None:
    payload = frozen["configs"].get(key)
    if payload is None:
        return None
    kwargs = dict(payload)
    kwargs["model_level"] = RebondModelLevel(kwargs["model_level"])
    kwargs["contact_model"] = ContactModel(kwargs["contact_model"])
    kwargs["feedback_mode"] = FeedbackMode(kwargs["feedback_mode"])
    kwargs["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(kwargs["initial_precrack_wake_mode"])
    return CrackRebondingControls(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    trajectories: dict[str, dict] = json.loads((run_root / "trajectories.json").read_text())
    frozen: dict = json.loads((run_root / "frozen_configuration.json").read_text())
    reference_contact_radius_m = frozen["material"]["reference_contact_radius_m"]

    # Barrier-floor/cooperative-saturation diagnostics: pure function of the
    # frozen config, computed once per {reversible, persistent} preset (the
    # two presets differ only in target reference actions -> different
    # resolved barriers; zero/finite-cohesion twins share identical
    # formation kinetics by construction, so this applies to both).
    rb2_rev_finite = _config_from_frozen(frozen, "RB2_reversible_finite")
    rb2_pers_finite = _config_from_frozen(frozen, "RB2_persistent_finite")
    barrier_diagnostics = {
        "reversible": pilot.barrier_floor_saturation_diagnostics(
            rb2_rev_finite, reference_contact_radius_m=reference_contact_radius_m
        ),
        "persistent": pilot.barrier_floor_saturation_diagnostics(
            rb2_pers_finite, reference_contact_radius_m=reference_contact_radius_m
        ),
    }

    ledger_trajectories: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for name, res in trajectories.items():
        events_out = []
        for e in res["events"]:
            event_row = {k: e.get(k) for k in EVENT_FIELDS}
            mpz_state = e.get("mpz_state") or {}
            for k in MPZ_STATE_FIELDS:
                event_row[k] = mpz_state.get(k)
            events_out.append(event_row)
            csv_rows.append({"trajectory": name, **event_row})

        ledger_trajectories[name] = {
            "R": res["R"],
            "rebonding_model_level": res["rebonding_model_level"],
            "restored_work_of_separation_J_m2": res["restored_work_of_separation_J_m2"],
            "manifest_audit": res["manifest_audit"],
            "seed": res["seed"],
            "n_accepted_events": res["n_accepted_events"],
            "cumulative_extension_m": res["cumulative_extension_m"],
            "cumulative_time_s": res["cumulative_time_s"],
            "censored": res["censored"],
            "censor_reason": res["censor_reason"],
            "uncensored": res["uncensored"],
            "events": events_out,
            "post_first_event_intervals": res["post_first_event_intervals"],
        }

    ledger = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_event_ledger_v1",
        "frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
        "barrier_floor_saturation_diagnostics": barrier_diagnostics,
        "trajectories": ledger_trajectories,
    }

    ledger_path = ARTIFACTS_DIR / "event_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ledger_path}")

    csv_path = ARTIFACTS_DIR / "event_ledger.csv"
    fieldnames = ["trajectory"] + EVENT_FIELDS + MPZ_STATE_FIELDS
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}  ({len(csv_rows)} event rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
