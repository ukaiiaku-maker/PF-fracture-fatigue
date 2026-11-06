"""Small reference-energy refinement after the shared thermal search."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import time

from .multi_temperature_bending_fit import (
    CONDITIONS, NAMES_THERMAL, ROOT, _candidate_summary, _jsonable,
    _preference, base_material, one_condition, thermal_material,
)
from .objective import physically_admissible
from .surfaces import surface_diagnostics

PROPOSALS = (
    ("thermal_best", 0., 0.),
    ("opening_floor_plus_0p01eV", .01, 0.),
    ("opening_floor_minus_0p01eV", -.01, 0.),
    ("emission_plus_0p02eV", 0., .02),
    ("emission_minus_0p02eV", 0., -.02),
)


def shifted_material(base, thermal_coordinates, delta_floor_eV, delta_emit_eV):
    material = thermal_material(base, thermal_coordinates)
    material = replace(material,
        cleavage=replace(material.cleavage,
                         Gf_ref_eV=material.cleavage.Gf_ref_eV + delta_floor_eV),
        emission=replace(material.emission,
                         G00_eV=material.emission.G00_eV + delta_emit_eV),
        candidate_id="ast2018_shared_bending_thermal_reference_refinement")
    material.cleavage.validate()
    if not physically_admissible(material):
        raise ValueError("reference-energy shift violates physical bounds")
    for T in (294.15, 373.15, 523.15, 773.15):
        for stress in (0., 1e9, 2e9, 4e9, 6e9, 8e9, 12e9):
            opening = material.cleavage.derivatives(stress, T)
            emission = surface_diagnostics(material.emission, stress, T, 1e11)
            if min(opening["H_eV"], emission["H_eV"]) <= 0:
                raise ValueError("nonpositive activation enthalpy")
            if max(opening["S_over_kB"], emission["S_over_kB"]) > 1e-9:
                raise ValueError("barrier does not rise with temperature")
    return material


_PRIMARY = None
_BASE = None


def _init():
    global _PRIMARY, _BASE
    _PRIMARY, _, _BASE = base_material()


def _case(job):
    x, df, de, curve_id, condition_id = job
    started = time.monotonic()
    try:
        material = shifted_material(_BASE, x, df, de)
        record, _ = one_condition(_PRIMARY, material, curve_id, condition_id)
        return record
    except Exception as exc:
        return {"curve_id": curve_id, "condition_id": condition_id,
                "status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}",
                "supported_points": 0, "total_points": 0,
                "supported_weight": 0., "normalized_objective": None,
                "wall_seconds": time.monotonic() - started}


def refine(output: Path = ROOT):
    output = Path(output)
    original = [json.loads(line) for line in
                (output / "search_evaluations.jsonl").read_text().splitlines()]
    best = min(original, key=_preference)
    x = [best["coordinates"][name] for name in NAMES_THERMAL]
    primary, _, base = base_material()
    for _, df, de in PROPOSALS:
        shifted_material(base, x, df, de)
    config = {"schema": "ast2018_reference_energy_refinement_v1",
              "thermal_source_candidate": best["name"],
              "thermal_coordinates": dict(zip(NAMES_THERMAL, x)),
              "proposals": [{"name": n, "delta_floor_eV": df,
                             "delta_emission_eV": de} for n, df, de in PROPOSALS],
              "case_count_per_proposal": len(CONDITIONS),
              "selection": "numerical errors, support, equal-curve level and shape"}
    path = output / "refinement_configuration.json"
    if path.is_file() and json.loads(path.read_text()) != config:
        raise RuntimeError("refinement configuration changed")
    path.write_text(json.dumps(config, indent=2) + "\n")
    log = output / "refinement_evaluations.jsonl"
    existing = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    done = {row["name"] for row in existing}
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as pool:
        for name, df, de in PROPOSALS:
            if name in done:
                continue
            started = time.monotonic()
            if name == "thermal_best":
                records = best["conditions"]
            else:
                futures = [pool.submit(_case, (x, df, de, curve_id, cid))
                           for curve_id, cid in CONDITIONS]
                records = [future.result() for future in as_completed(futures)]
                records.sort(key=lambda r: [c[0] for c in CONDITIONS].index(r["curve_id"]))
            row = {"name": name, "thermal_coordinates": dict(zip(NAMES_THERMAL, x)),
                   "delta_floor_eV": df, "delta_emission_eV": de,
                   "conditions": records, "summary": _candidate_summary(records),
                   "stage_wall_seconds": time.monotonic() - started}
            with log.open("a") as stream:
                stream.write(json.dumps(row, default=_jsonable, allow_nan=False) + "\n")
                stream.flush()
            existing.append(row); done.add(name)
            winner = min(existing, key=_preference)
            (output / "refinement_summary.json").write_text(json.dumps({
                "best_name": winner["name"], "best_summary": winner["summary"],
                "candidates": [{"name": r["name"], "summary": r["summary"]}
                               for r in existing]}, indent=2) + "\n")
            print(name, row["summary"], round(row["stage_wall_seconds"], 1),
                  "best", winner["name"], flush=True)
    return min(existing, key=_preference)


if __name__ == "__main__":
    refine()
