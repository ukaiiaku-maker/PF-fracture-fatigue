"""PX5 (mission section 8, review-scoped): compare each static-shield
control trajectory's developed da/dN against PX4's own zero-cohesion and
dynamic-rebonding trajectories at the SAME protocol/Kmax/seed=1720, and
classify each (protocol, Kmax) point:

- FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT: both static controls
  reproduce >=80% of the dynamic slowdown (S_h_static/S_h_dynamic >= 0.8)
  -- the crack tip's instantaneous/average shielding level alone explains
  the dynamic-rebonding effect; formation/rupture KINETICS add little.
- DYNAMIC_REBONDING_HISTORY_REQUIRED: both static controls reproduce
  <=30% of the dynamic slowdown -- static shielding at any fixed level
  cannot explain the effect; the TIMING/HISTORY of bond formation and
  rupture relative to the loading cycle is doing the work.
- MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY: anything between (or the
  two controls disagree with each other, straddling the two thresholds).

Reuses stable_growth_gate verbatim (same primitive PX4's own analysis
uses) -- computed here directly from developed_event_ledger.json's raw
events for PX4's finite/zero baselines (never refit or reimplemented),
and from the PX5 result.json files for the 20 static-shield trajectories.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import stable_growth_gate  # noqa: E402

REPRODUCTION_FRACTION_DOMINANT = 0.80
REPRODUCTION_FRACTION_HISTORY_REQUIRED = 0.30
KMAX_GRID_Pa_sqrt_m = [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]


def _load_px5_registry() -> list[dict]:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")))
    retry_path = ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv"
    if retry_path.is_file():
        rows += list(csv.DictReader(open(retry_path)))
    return [r for r in rows if r["status"] == "AUTHORIZED_PX5"]


def _load_results_by_key() -> dict[str, dict]:
    by_key = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        job = r.get("job")
        if job is not None:
            by_key[job["canonical_job_key"]] = r
    return by_key


def _developed_da_dN(events: list[dict], frequency_Hz: float) -> dict:
    return stable_growth_gate(events, frequency_Hz=frequency_Hz)


def main() -> None:
    px5_rows = _load_px5_registry()
    if len(px5_rows) != 20:
        raise RuntimeError(f"expected exactly 20 AUTHORIZED_PX5 rows (18 original + 2 retry), found {len(px5_rows)}")

    px4_registry = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")))
    px4_authorized = [r for r in px4_registry if r["status"] == "AUTHORIZED_PX4"]
    px4_key_by = {(r["protocol"], round(float(r["Kmax_Pa_sqrt_m"])), r["cohesion"]): r["canonical_job_key"] for r in px4_authorized}

    results = _load_results_by_key()

    px4_gate_cache: dict[str, dict] = {}
    def _px4_gate(key: str, frequency_Hz: float) -> dict:
        if key not in px4_gate_cache:
            traj = results[key]["trajectory"]
            px4_gate_cache[key] = _developed_da_dN(traj["events"], frequency_Hz)
        return px4_gate_cache[key]

    rows_out = []
    pending = []
    for protocol in ("D2", "D5"):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            zero_key = px4_key_by[(protocol, round(Kmax), "zero")]
            dyn_key = px4_key_by[(protocol, round(Kmax), "finite")]
            gate_zero = _px4_gate(zero_key, 1000.0)
            gate_dyn = _px4_gate(dyn_key, 1000.0)
            da_dN_zero = gate_zero["developed_da_dN_m_per_cycle"]
            da_dN_dyn = gate_dyn["developed_da_dN_m_per_cycle"]
            S_h_dynamic = math.log10(da_dN_dyn / da_dN_zero) if (da_dN_zero and da_dN_dyn and da_dN_zero > 0 and da_dN_dyn > 0) else None

            per_control = {}
            for control_mode in ("ceiling_static", "orbit_matched_static"):
                match = [r for r in px5_rows if r["protocol"] == protocol and round(float(r["Kmax_Pa_sqrt_m"])) == round(Kmax) and r["cohesion"] == control_mode]
                if not match:
                    raise RuntimeError(f"missing PX5 authorized row for {protocol}/{Kmax}/{control_mode}")
                row = match[0]
                key = row["canonical_job_key"]
                if key not in results:
                    pending.append((protocol, Kmax, control_mode))
                    per_control[control_mode] = None
                    continue
                traj = results[key]["trajectory"]
                if traj["censored"] or traj["n_accepted_events"] != 30:
                    pending.append((protocol, Kmax, control_mode))
                    per_control[control_mode] = None
                    continue
                gate_static = _developed_da_dN(traj["events"], 1000.0)
                da_dN_static = gate_static["developed_da_dN_m_per_cycle"]
                S_h_static = (
                    math.log10(da_dN_static / da_dN_zero)
                    if (da_dN_zero and da_dN_static and da_dN_zero > 0 and da_dN_static > 0) else None
                )
                reproduction_fraction = (
                    S_h_static / S_h_dynamic if (S_h_static is not None and S_h_dynamic not in (None, 0.0)) else None
                )
                per_control[control_mode] = {
                    "canonical_job_key": key, "stable_growth": gate_static["stable_growth_provisional"],
                    "developed_da_dN_m_per_cycle": da_dN_static, "S_h_static": S_h_static,
                    "reproduction_fraction_of_dynamic": reproduction_fraction,
                }

            fractions = [
                per_control[c]["reproduction_fraction_of_dynamic"] for c in ("ceiling_static", "orbit_matched_static")
                if per_control[c] is not None and per_control[c]["reproduction_fraction_of_dynamic"] is not None
            ]
            if len(fractions) < 2:
                classification = "PENDING_DATA"
            elif all(f >= REPRODUCTION_FRACTION_DOMINANT for f in fractions):
                classification = "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT"
            elif all(f <= REPRODUCTION_FRACTION_HISTORY_REQUIRED for f in fractions):
                classification = "DYNAMIC_REBONDING_HISTORY_REQUIRED"
            else:
                classification = "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY"

            rows_out.append({
                "protocol": protocol, "Kmax_Pa_sqrt_m": Kmax, "S_h_dynamic": S_h_dynamic,
                "ceiling_static": per_control["ceiling_static"], "orbit_matched_static": per_control["orbit_matched_static"],
                "classification": classification,
            })

    out = {
        "schema": "v10230_part_x_px5_static_shield_analysis_v1",
        "reproduction_fraction_thresholds": {
            "dominant_if_both_controls_ge": REPRODUCTION_FRACTION_DOMINANT,
            "history_required_if_both_controls_le": REPRODUCTION_FRACTION_HISTORY_REQUIRED,
        },
        "pending_conditions": pending,
        "points": rows_out,
    }
    (ARTIFACTS_DIR / "px5_static_shield_analysis.json").write_text(json.dumps(out, indent=2, default=str))

    flat = []
    for row in rows_out:
        flat_row = {"protocol": row["protocol"], "Kmax_Pa_sqrt_m": row["Kmax_Pa_sqrt_m"], "S_h_dynamic": row["S_h_dynamic"]}
        for c in ("ceiling_static", "orbit_matched_static"):
            pc = row[c]
            flat_row[f"{c}_S_h"] = pc["S_h_static"] if pc else None
            flat_row[f"{c}_reproduction_fraction"] = pc["reproduction_fraction_of_dynamic"] if pc else None
        flat_row["classification"] = row["classification"]
        flat.append(flat_row)
    with (ARTIFACTS_DIR / "px5_static_shield_analysis.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)

    print(f"Wrote px5_static_shield_analysis.{{json,csv}}: {len(rows_out)} (protocol, Kmax) points")
    if pending:
        print(f"  PENDING (not yet complete/uncensored): {pending}")
    for row in rows_out:
        print(f"  {row['protocol']} Kmax={row['Kmax_Pa_sqrt_m']/1e6:5.1f}MPa  S_h_dynamic={row['S_h_dynamic']}  "
              f"classification={row['classification']}")


if __name__ == "__main__":
    main()
