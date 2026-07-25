#!/usr/bin/env python3
"""Create initial, intermediate, final, and extension-averaged K(T) plots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

CASE_RE = re.compile(
    r"T(?P<T>\d+(?:\.\d+)?)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$"
)
DEFAULT_OPTION_ORDER = (
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
)
SHORT_LABELS = {
    DEFAULT_OPTION_ORDER[0]: "Peak 0242980",
    DEFAULT_OPTION_ORDER[1]: "DBTT 0202500",
    DEFAULT_OPTION_ORDER[2]: "Weak-T/FCC-like 0257068",
    DEFAULT_OPTION_ORDER[3]: "Ceramic-like 0189364",
}
METRICS = (
    ("K_initial_MPa_sqrt_m", "Initial"),
    ("K_intermediate_MPa_sqrt_m", "Intermediate (500 µm)"),
    ("K_end_MPa_sqrt_m", "End (1000 µm)"),
    ("K_average_MPa_sqrt_m", "Extension-averaged"),
)


def _load_manifest(outroot: Path) -> dict[str, Any]:
    path = outroot / "v10_2_27_campaign_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_steps(case: Path) -> np.ndarray:
    files = sorted(case.glob("steps_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one steps CSV in {case}; found {files}")
    data = np.atleast_1d(
        np.genfromtxt(files[0], delimiter=",", names=True, dtype=float)
    )
    required = {"KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire"}
    missing = required - set(data.dtype.names or ())
    if missing:
        raise RuntimeError(f"{files[0]} missing columns {sorted(missing)}")
    return data


def event_curve(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fired = np.asarray(data["n_fire"], dtype=float) > 0.0
    if not np.any(fired):
        empty = np.array([], dtype=float)
        return empty, empty, empty
    post = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)[fired]
    increment = 1.0e6 * np.asarray(data["da_block_m"], dtype=float)[fired]
    pre = post - increment
    resistance = 1.0e-6 * np.asarray(data["KJ_Pa_sqrtm"], dtype=float)[fired]
    valid = (
        np.isfinite(pre)
        & np.isfinite(post)
        & np.isfinite(resistance)
        & (post >= pre)
    )
    pre = np.maximum(pre[valid], 0.0)
    post = np.maximum(post[valid], pre)
    resistance = resistance[valid]
    order = np.argsort(pre, kind="stable")
    return pre[order], post[order], resistance[order]


def resistance_at_extension(
    pre: np.ndarray,
    post: np.ndarray,
    resistance: np.ndarray,
    target_um: float,
    achieved_um: float,
) -> float:
    if pre.size == 0 or not np.isfinite(achieved_um):
        return float("nan")
    target = float(target_um)
    tol = 1.0e-9 * max(abs(achieved_um), abs(target), 1.0)
    if target < -tol or target > achieved_um + tol:
        return float("nan")
    if target <= pre[0] + tol:
        return float(resistance[0])
    for index in range(pre.size):
        if pre[index] - tol <= target <= post[index] + tol:
            return float(resistance[index])
        if index + 1 < pre.size and post[index] < target < pre[index + 1]:
            return float(
                np.interp(
                    target,
                    [post[index], pre[index + 1]],
                    [resistance[index], resistance[index + 1]],
                )
            )
    if target <= achieved_um + tol:
        return float(resistance[-1])
    return float("nan")


def extension_weighted_average(
    pre: np.ndarray,
    post: np.ndarray,
    resistance: np.ndarray,
    target_um: float,
    achieved_um: float,
) -> float:
    """Return target^-1 integral K(Delta a) d(Delta a).

    Knots are every event start/end plus the requested endpoints.  On each open
    interval the realized curve is either constant or linear, so midpoint
    quadrature is exact and is insensitive to jump values at shared boundaries.
    """
    target = float(target_um)
    if target <= 0.0 or pre.size == 0 or achieved_um < target:
        return float("nan")
    knots = [0.0, target]
    for left, right in zip(pre, post):
        if 0.0 < left < target:
            knots.append(float(left))
        if 0.0 < right < target:
            knots.append(float(right))
    x = np.asarray(sorted(set(knots)), dtype=float)
    area = 0.0
    for left, right in zip(x[:-1], x[1:]):
        midpoint = 0.5 * (left + right)
        value = resistance_at_extension(
            pre, post, resistance, midpoint, achieved_um
        )
        if not math.isfinite(value):
            return float("nan")
        area += value * (right - left)
    return float(area / target)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _ordered_options(records: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    present = {str(row["option_key"]) for row in records}
    requested = [str(value) for value in manifest.get("options", DEFAULT_OPTION_ORDER)]
    ordered = [option for option in requested if option in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True)
    parser.add_argument("--plot-dir")
    parser.add_argument("--target-extension-um", type=float)
    args = parser.parse_args()

    outroot = Path(args.outroot).expanduser().resolve()
    manifest = _load_manifest(outroot)
    target_um = float(
        args.target_extension_um
        if args.target_extension_um is not None
        else manifest["target_crack_extension_um"]
    )
    intermediate_um = 0.5 * target_um
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir
        else outroot / "plots" / "K_vs_temperature"
    )

    records: list[dict[str, Any]] = []
    for option_dir in sorted(path for path in outroot.iterdir() if path.is_dir()):
        option = option_dir.name
        for case in sorted(path for path in option_dir.iterdir() if path.is_dir()):
            match = CASE_RE.match(case.name)
            if not match or not (case / "COMPLETE").is_file():
                continue
            data = _load_steps(case)
            pre, post, resistance = event_curve(data)
            if pre.size == 0:
                continue
            achieved_values = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)
            achieved_values = achieved_values[np.isfinite(achieved_values)]
            achieved_um = (
                float(np.max(achieved_values))
                if achieved_values.size
                else float("nan")
            )
            records.append(
                {
                    "option_key": option,
                    "label": SHORT_LABELS.get(option, option),
                    "temperature_K": float(match.group("T")),
                    "theta_deg": float(match.group("theta")),
                    "seed": int(match.group("seed")),
                    "target_extension_um": target_um,
                    "intermediate_extension_um": intermediate_um,
                    "achieved_extension_um": achieved_um,
                    "K_initial_MPa_sqrt_m": float(resistance[0]),
                    "K_intermediate_MPa_sqrt_m": resistance_at_extension(
                        pre, post, resistance, intermediate_um, achieved_um
                    ),
                    "K_end_MPa_sqrt_m": resistance_at_extension(
                        pre, post, resistance, target_um, achieved_um
                    ),
                    "K_average_MPa_sqrt_m": extension_weighted_average(
                        pre, post, resistance, target_um, achieved_um
                    ),
                    "average_definition": (
                        "integral_0_to_target K(Delta_a) dDelta_a / target; "
                        "piecewise constant across accepted event intervals and "
                        "linear across inter-event gaps"
                    ),
                    "case_root": str(case),
                }
            )

    if not records:
        raise SystemExit(f"no complete event curves found below {outroot}")
    options = _ordered_options(records, manifest)
    rank = {option: index for index, option in enumerate(options)}
    records.sort(
        key=lambda row: (rank.get(row["option_key"], 999), row["temperature_K"])
    )

    expected = len(manifest.get("options", [])) * len(
        manifest.get("temperatures_K", [])
    )
    if expected and len(records) != expected:
        raise RuntimeError(
            f"temperature summary found {len(records)} complete curves; expected {expected}"
        )
    for row in records:
        for key, _ in METRICS:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f"non-finite {key} for {row['option_key']} "
                    f"at {row['temperature_K']} K"
                )

    csv_path = outroot / "v10_2_27_paper_four_class_K_vs_temperature_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    markers = ["o", "s", "^", "D"]
    for option in options:
        subset = [row for row in records if row["option_key"] == option]
        subset.sort(key=lambda row: row["temperature_K"])
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for index, (key, label) in enumerate(METRICS):
            ax.plot(
                [row["temperature_K"] for row in subset],
                [row[key] for row in subset],
                marker=markers[index % len(markers)],
                linewidth=1.4,
                markersize=5.0,
                label=label,
            )
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Fracture resistance, K (MPa√m)")
        ax.set_title(
            f"{SHORT_LABELS.get(option, option)}: resistance metrics versus T"
        )
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True)
        _save(
            fig,
            plot_dir / "by_candidate" / f"{option}_K_metrics_vs_temperature.png",
        )

    for key, label in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for index, option in enumerate(options):
            subset = [row for row in records if row["option_key"] == option]
            subset.sort(key=lambda row: row["temperature_K"])
            ax.plot(
                [row["temperature_K"] for row in subset],
                [row[key] for row in subset],
                marker=markers[index % len(markers)],
                linewidth=1.4,
                markersize=5.0,
                label=SHORT_LABELS.get(option, option),
            )
        ax.set_xlabel("Temperature, T (K)")
        ax.set_ylabel("Fracture resistance, K (MPa√m)")
        ax.set_title(f"{label} resistance versus temperature")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=True)
        _save(fig, plot_dir / "by_metric" / f"{key}_vs_temperature.png")

    audit = {
        "schema": "v10.2.27_four_class_K_vs_temperature_postprocess_v1",
        "target_extension_um": target_um,
        "intermediate_extension_um": intermediate_um,
        "initial_definition": "first accepted cleavage-event resistance",
        "end_definition": "resistance at target projected crack extension",
        "average_definition": records[0]["average_definition"],
        "case_count": len(records),
        "expected_case_count": expected or None,
        "all_expected_cases_processed": (
            len(records) == expected if expected else None
        ),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "plot_directory": str(plot_dir),
    }
    (outroot / "v10_2_27_K_vs_temperature_postprocess_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote K-versus-temperature summaries for {len(records)} cases: {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
