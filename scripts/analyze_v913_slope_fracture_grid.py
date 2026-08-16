#!/usr/bin/env python3
"""Qualify the prospective slope-design monotonic fracture grid.

This is a read-only analysis of case JSON and additive event-state archives.  A
row is eligible for fatigue transfer only when all eleven requested
temperatures are complete, every state archive hashes correctly, and its exact
300 K K50 lies within the declared five-percent parent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TEMPERATURES = (300, 700, 800, 900, 950, 1000, 1050, 1100, 1200, 1300, 1400)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def morphology(temperature: np.ndarray, toughness: np.ndarray) -> str:
    historical = temperature != 300.0
    t, k = temperature[historical], toughness[historical]
    peak = int(np.argmax(k))
    prominence = min(k[peak] - k[0], k[peak] - k[-1])
    if 0 < peak < len(k) - 1 and prominence >= 5.0:
        return "PEAK_T"
    if k[-1] - k[0] >= 5.0:
        return "DBTT_LIKE"
    if np.ptp(k) <= 5.0:
        return "WEAK_T"
    if k[-1] - k[0] <= -5.0:
        return "CERAMIC_OR_INVERSE_T"
    return "INTERMEDIATE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k300-relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    rows: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    for source in registry.itertuples(index=False):
        candidate = str(source.prospective_candidate_id)
        cases = []
        failures: list[str] = []
        snapshot_count = 0
        for temperature in TEMPERATURES:
            path = args.case_root / "cases" / f"{candidate}__T{temperature}K.json"
            if not path.exists():
                failures.append(f"missing_T{temperature}")
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                failures.append(f"status_T{temperature}:{payload.get('status')}")
            archive = Path(str(payload.get("event_state_npz", "")))
            if not archive.exists() or sha256(archive) != payload.get("event_state_npz_sha256"):
                failures.append(f"state_hash_T{temperature}")
            else:
                with np.load(archive) as state:
                    event_index = np.asarray(state["event_index"])
                    if len(event_index) != int(payload.get("full_state_snapshot_count", -1)):
                        failures.append(f"state_count_T{temperature}")
                    snapshot_count += len(event_index)
            cases.append(payload)
        cases.sort(key=lambda value: float(value["temperature_K"]))
        temperatures = np.asarray([float(value["temperature_K"]) for value in cases])
        toughness = np.asarray([float(value["K_50um_MPa_sqrt_m"]) for value in cases])
        complete_grid = not failures and tuple(temperatures.astype(int)) == TEMPERATURES
        k300 = float(toughness[np.where(temperatures == 300.0)[0][0]]) if 300.0 in temperatures else np.nan
        parent_k300 = float(source.parent_K300_MPa_sqrt_m)
        relative = abs(k300 - parent_k300) / parent_k300 if math.isfinite(k300) else np.nan
        qualified = complete_grid and relative <= args.k300_relative_tolerance + 1e-12
        historical = temperatures != 300.0
        ht, hk = temperatures[historical], toughness[historical]
        derivative = np.gradient(hk, ht) if len(hk) == 10 else np.full(len(hk), np.nan)
        peak = int(np.argmax(hk)) if len(hk) else 0
        row = {
            "candidate_id": candidate,
            "parent_candidate_id": source.parent_candidate_id,
            "parent_family": source.parent_family,
            "design_axis": source.design_axis,
            "design_sign": source.design_sign,
            "parameter_fingerprint": source.parameter_fingerprint,
            "case_count": len(cases),
            "full_state_snapshot_count": snapshot_count,
            "complete_temperature_grid": complete_grid,
            "state_archives_hash_valid": not any("state_" in item for item in failures),
            "K300_MPa_sqrt_m": k300,
            "parent_K300_MPa_sqrt_m": parent_k300,
            "K300_relative_error": relative,
            "K300_gate_tolerance": args.k300_relative_tolerance,
            "fracture_qualified_for_fatigue": qualified,
            "morphology_class": morphology(temperatures, toughness) if complete_grid else "INCOMPLETE",
            "K700_MPa_sqrt_m": float(hk[0]) if len(hk) else np.nan,
            "K1400_MPa_sqrt_m": float(hk[-1]) if len(hk) else np.nan,
            "K_span_MPa_sqrt_m": float(np.ptp(hk)) if len(hk) else np.nan,
            "DBTT_magnitude_MPa_sqrt_m": float(hk[-1] - hk[0]) if len(hk) else np.nan,
            "peak_temperature_K": float(ht[peak]) if len(ht) and 0 < peak < len(ht)-1 else np.nan,
            "peak_prominence_MPa_sqrt_m": float(min(hk[peak]-hk[0], hk[peak]-hk[-1])) if len(hk) else np.nan,
            "mean_dK_dT_MPa_sqrt_m_per_K": float(np.polyfit(ht, hk, 1)[0]) if len(hk) == 10 else np.nan,
            "max_dK_dT_MPa_sqrt_m_per_K": float(np.max(derivative)) if len(derivative) else np.nan,
            "min_dK_dT_MPa_sqrt_m_per_K": float(np.min(derivative)) if len(derivative) else np.nan,
            "failure_reasons": ";".join(failures),
            "physics_changed": False,
        }
        rows.append(row)
        for temperature, value in zip(temperatures, toughness):
            points.append({
                "candidate_id": candidate,
                "parent_family": source.parent_family,
                "design_axis": source.design_axis,
                "temperature_K": temperature,
                "K50_MPa_sqrt_m": value,
                "K50_over_parent_K300": value / parent_k300,
                "fracture_qualified_for_fatigue": qualified,
            })
    result = pd.DataFrame(rows).sort_values("candidate_id")
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "prospective_slope_fracture_results.csv", index=False)
    pd.DataFrame(points).to_csv(args.out / "prospective_slope_fracture_curve_points.csv", index=False)
    manifest = {
        "schema": "v913_prospective_slope_fracture_qualification_v1",
        "candidate_count": len(result),
        "complete_candidate_count": int(result.complete_temperature_grid.sum()),
        "fatigue_qualified_candidate_count": int(result.fracture_qualified_for_fatigue.sum()),
        "full_state_snapshot_count": int(result.full_state_snapshot_count.sum()),
        "requested_temperatures_K": list(TEMPERATURES),
        "K300_relative_tolerance": args.k300_relative_tolerance,
        "qualification_is_pre_fatigue": True,
        "physics_changed": False,
    }
    (args.out / "prospective_slope_fracture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"V913_SLOPE_FRACTURE_ANALYSIS_COMPLETE candidates={len(result)} qualified={manifest['fatigue_qualified_candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
