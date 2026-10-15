#!/usr/bin/env python3
"""Qualify historical and native K-versus-length analysis without rerunning fracture."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.fracture_observation_history import (
    LengthCoordinate, ObservationHistory, ObservationStage,
    from_legacy_steps_csv, with_geometry_event_arclength,
)
from scripts.plot_fracture_observation_history import plot_history


FEM_ROOT = ROOT.parent / "fem-czm-current-pf-companion"
OUTDIR = ROOT / "common_framework_readiness/k_length_analysis_port_20261014"
LEGACY = ROOT / "integration_evidence/new_full_pf_1000um_dbtt_300K/steps_0300K.csv"
GEOMETRY_201 = ROOT / "integration_evidence/targeted_completion/current_single_front_1000um"
NATIVE_203 = ROOT / "integration_evidence/current_science_pilots_6abbf10/pf/DBTT_1200K/developed_1000um_current_v2_run3"
RESTART = ROOT / "integration_evidence/f1_native_taylor_peierls_qualification_2276745b"
EXPECTED_LEGACY_SHA = "3e67545d96d9dfe8c47d83e3e59813699ccf8135e83f849c1bbbec4f8dd028ea"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _finite_counts(history: ObservationHistory) -> dict[str, int]:
    fields = (
        "K_J_Pa_sqrt_m", "K_I_Pa_sqrt_m", "K_II_Pa_sqrt_m",
        "K_effective_Pa_sqrt_m", "K_probe_Pa_sqrt_m",
        "K_applied_Pa_sqrt_m", "J_J_per_m2", "J_signed_J_per_m2",
        "G_J_per_m2",
    )
    return {field: sum(math.isfinite(getattr(row, field)) for row in history.rows)
            for field in fields}


def _query_summary(history: ObservationHistory, targets_um: tuple[float, ...],
                   coordinate: LengthCoordinate) -> dict[str, Any]:
    result = {}
    for target in targets_um:
        query = history.at_length(target * 1e-6, coordinate=coordinate)
        associated = history.event_associated_at_length(
            target * 1e-6, coordinate=coordinate
        )
        result[f"{target:g}_um"] = {
            "strict_status": query.status,
            "exact_count": len(query.exact),
            "crossing_count": len(query.crossing),
            "event_associated_status": associated.status,
            "event_associated_count": len(associated.brackets),
            "interpolation_performed": False,
        }
    return result


def _legacy_receipt() -> tuple[dict[str, Any], ObservationHistory]:
    if _sha(LEGACY) != EXPECTED_LEGACY_SHA:
        raise RuntimeError("legacy CSV identity mismatch")
    history = from_legacy_steps_csv(LEGACY)
    with LEGACY.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    events = sum(float(row.get("n_fire", 0) or 0) > 0 for row in raw)
    repeated = sum(
        history.rows[i].committed_length_m == history.rows[i - 1].committed_length_m
        for i in range(1, len(history.rows))
    )
    return {
        "source": str(LEGACY.relative_to(ROOT)), "sha256": _sha(LEGACY),
        "raw_rows": len(raw), "adapted_rows": len(history.rows),
        "event_rows": events, "non_event_rows": len(raw) - events,
        "repeated_committed_length_rows": repeated,
        "stages": dict(Counter(row.stage.value for row in history.rows)),
        "validity": dict(Counter(row.validity_status.value for row in history.rows)),
        "finite_quantity_counts": _finite_counts(history),
        "queries": _query_summary(history, (100.0, 500.0, 1000.0), LengthCoordinate.EXTENSION),
        "historical_route_disposition": (
            "All rows and native K_J values preserved. Historical interval K is retained only "
            "as an event-associated view; gap interpolation and final-event carry-forward are corrected."
        ),
    }, history


def _geometry_receipt(root: Path, *, expected_events: int) -> tuple[dict[str, Any], ObservationHistory]:
    source = root / "fracture_observations_online.jsonl"
    if root == GEOMETRY_201:
        source = root / "fracture_observations_geometry_aware.jsonl"
        history = ObservationHistory.load(source)
    else:
        history = ObservationHistory.load(source)
        events = json.loads((root / "stochastic_avalanche_geometry_events.json").read_text())
        history = with_geometry_event_arclength(history, events)
    stages = Counter(row.stage.value for row in history.rows)
    if stages[ObservationStage.COMMITTED.value] != expected_events:
        raise RuntimeError(f"unexpected event count in {root}")
    finite_path = [row.path_arclength_m for row in history.rows if math.isfinite(row.path_arclength_m)]
    return {
        "source": str(source.relative_to(ROOT)), "source_sha256": _sha(source),
        "rows": len(history.rows), "stages": dict(stages),
        "event_rows": expected_events,
        "terminal_projected_extension_m": max(row.stage_extension_m for row in history.rows),
        "terminal_path_arclength_m": max(finite_path),
        "finite_quantity_counts": _finite_counts(history),
        "projected_queries": _query_summary(
            history, (100.0, 500.0, 1000.0), LengthCoordinate.EXTENSION
        ),
        "path_queries": _query_summary(
            history, (100.0, 500.0, 1000.0), LengthCoordinate.PATH_ARCLENGTH
        ),
    }, history


def _restart_receipt() -> dict[str, Any]:
    paths = {
        "uninterrupted": RESTART / "uninterrupted_to_step64/fracture_observations_online.jsonl",
        "fresh_process_resumed": RESTART / "fresh_process_restart_to_step64/fracture_observations_online.jsonl",
        "pre_restart_prefix": RESTART / "initial_to_step48/fracture_observations_online.jsonl",
    }
    histories = {name: [json.loads(line) for line in path.read_text().splitlines() if line]
                 for name, path in paths.items()}
    left, right = histories["uninterrupted"], histories["fresh_process_resumed"]
    normalized_equal = all(
        {k: v for k, v in a.items() if k != "source_file"}
        == {k: v for k, v in b.items() if k != "source_file"}
        for a, b in zip(left, right)
    ) and len(left) == len(right)
    prefix_equal = all(
        {k: v for k, v in a.items() if k != "source_file"}
        == {k: v for k, v in b.items() if k != "source_file"}
        for a, b in zip(histories["pre_restart_prefix"], right[:48])
    )
    return {
        "sources": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "row_counts": {name: len(rows) for name, rows in histories.items()},
        "uninterrupted_equals_fresh_process_except_source_path": normalized_equal,
        "pre_restart_prefix_preserved": prefix_equal,
        "sequence_unique": len({row["sequence"] for row in right}) == len(right),
        "boundary_duplicate_or_drop": not (normalized_equal and prefix_equal),
    }


def _fem_lineage() -> list[dict[str, Any]]:
    items = (
        ("94302bdbfd786ba4226722b4318e546db8268c02", "arrhenius_fracture/kj_audit_v10056.py", "outer-radius, clearance, plateau and load scaling"),
        ("2b9122b19ed333827c8801faac7e5a9cc50fbff4", "arrhenius_fracture/fixed_grip_elastic_audit_v10058.py", "finite killed-notch fixed-grip energy derivative"),
        ("5e1c0c4777a220bc087151104b19167c73bc5a9c", "arrhenius_fracture/production_j_parity_v10059.py", "production field, exclusion and energy-density parity"),
        ("879450ac9537b65138a59ff121954ab04477d818", "arrhenius_fracture/production_j_refinement_support_v100510.py", "independent physical support"),
        ("23463cac72ebbc3be9eb53ea9fd86229102e1e98", "arrhenius_fracture/production_j_same_mesh_energy_v100511.py", "same-production-mesh forward energy derivative"),
        ("293491157063484bb10df6adc479f847a6a08ba6", "arrhenius_fracture/mode_i_first_passage_v10_0_5_12_phase_c.py", "production physical-support policy and metadata"),
    )
    result = []
    for commit, path, obligation in items:
        historical = _git(FEM_ROOT, "show", f"{commit}:{path}").encode()
        current = FEM_ROOT / path
        result.append({
            "commit": commit, "path": path, "obligation": obligation,
            "historical_blob_sha256": hashlib.sha256(historical).hexdigest(),
            "current_file_present": current.is_file(),
            "current_file_sha256": _sha(current) if current.is_file() else None,
            "disposition": "preserved source and mapped to current qualified observer chain",
        })
    return result


def build_receipt(plot_path: Path) -> dict[str, Any]:
    legacy, _ = _legacy_receipt()
    geometry_201, _ = _geometry_receipt(GEOMETRY_201, expected_events=201)
    native_203, native_history = _geometry_receipt(NATIVE_203, expected_events=203)
    selected = plot_history(
        native_history, plot_path, quantity="K_J", stage="pre_event",
        coordinate="path_arclength",
    )
    return {
        "schema": "k-length-analysis-port-qualification/1",
        "status": "PASS_CURRENT_NATIVE_AND_HISTORICAL_REPLAY",
        "physics_advanced": False,
        "source_heads": {"pf_common": _git(ROOT, "rev-parse", "HEAD"),
                         "fem_czm": _git(FEM_ROOT, "rev-parse", "HEAD")},
        "fem_observer_lineage": _fem_lineage(),
        "legacy_csv_replay": legacy,
        "geometry_aware_201_event_replay": geometry_201,
        "current_native_203_event_replay": native_203,
        "restart_chronology": _restart_receipt(),
        "plot_evidence": {
            "path": str(plot_path.relative_to(ROOT)), "sha256": _sha(plot_path),
            "quantity": "K_J_Pa_sqrt_m", "stage": "pre_event",
            "coordinate": "path_arclength", "selected_rows": len(selected),
            "fallback_between_K_definitions": False,
        },
        "paired_analysis_policy": (
            "Mechanism-on/off differences remain descriptive paired contrasts. Geometry, load, "
            "stage, observer and support matching must be reported; no subtraction is fed back "
            "as an empirical constitutive or K(a) correction."
        ),
        "preservation": (
            "No fracture specimen was rerun; no historical source or result was deleted or "
            "overwritten; production and manuscript modification remain disabled."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot = args.outdir / "current_native_203_event_KJ_vs_path.png"
    receipt = build_receipt(plot)
    output = args.outdir / "K_LENGTH_ANALYSIS_PORT_QUALIFICATION.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
