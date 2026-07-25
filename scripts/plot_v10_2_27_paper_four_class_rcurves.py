#!/usr/bin/env python3
"""Event-resolved K-R curves for the v10.2.27 four-class paper campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(
    r"T(?P<T>\d+(?:\.\d+)?)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$"
)
CHECKPOINT_CANDIDATES_UM = (
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
    200.0,
    300.0,
    400.0,
    500.0,
    750.0,
    1000.0,
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


def checkpoints_for_target(target_um: float) -> tuple[float, ...]:
    tol = 1.0e-9 * max(abs(float(target_um)), 1.0)
    return tuple(value for value in CHECKPOINT_CANDIDATES_UM if value <= target_um + tol)


def _load_steps(case: Path) -> tuple[np.ndarray, Path]:
    files = sorted(case.glob("steps_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one steps CSV in {case}; found {files}")
    data = np.atleast_1d(
        np.genfromtxt(files[0], delimiter=",", names=True, dtype=float)
    )
    names = set(data.dtype.names or ())
    required = {"KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire"}
    missing = required - names
    if missing:
        raise RuntimeError(f"{files[0]} missing columns {sorted(missing)}")
    return data, files[0]


def _event_curve(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-event extension, post-event extension, and event resistance."""
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


def _achieved_extension_um(data: np.ndarray) -> float:
    values = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if values.size else float("nan")


def _resistance_at_extension(
    pre: np.ndarray,
    post: np.ndarray,
    resistance: np.ndarray,
    target_um: float,
    achieved_um: float,
) -> float:
    """Return resistance supported by the realized discrete crack path."""
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


def _case_diagnostics(data: np.ndarray) -> dict[str, float]:
    names = set(data.dtype.names or ())

    def finite_stat(name: str, fn, scale: float = 1.0) -> float:
        if name not in names:
            return float("nan")
        values = np.asarray(data[name], dtype=float) / scale
        values = values[np.isfinite(values)]
        return float(fn(values)) if values.size else float("nan")

    return {
        "max_backstress_GPa": finite_stat("sigma_back_Pa", np.max, 1.0e9),
        "min_available_site_fraction": finite_stat(
            "mpz_available_site_fraction", np.min
        ),
        "max_abs_shield_MPa_sqrt_m": finite_stat(
            "mpz_K_shield_Pa_sqrt_m", lambda values: np.max(np.abs(values)), 1.0e6
        ),
    }


def _persistent_diagnostics(case: Path) -> dict[str, float]:
    path = case / "anisotropic_emission_audit_v10174.json"
    if not path.exists():
        return {
            "min_front_width_um": float("nan"),
            "max_tip_radius_um": float("nan"),
            "max_persistent_hazard_per_s": float("nan"),
        }
    payload = json.loads(path.read_text())
    records = payload.get("records", [])

    def finite_stat(key: str, fn, scale: float = 1.0) -> float:
        values = np.asarray(
            [record.get(key, float("nan")) for record in records], dtype=float
        )
        values = values[np.isfinite(values)] / scale
        return float(fn(values)) if values.size else float("nan")

    return {
        "min_front_width_um": finite_stat(
            "persistent_site_front_width_m", np.min, 1.0e-6
        ),
        "max_tip_radius_um": finite_stat(
            "persistent_tip_radius_m", np.max, 1.0e-6
        ),
        "max_persistent_hazard_per_s": finite_stat(
            "persistent_aggregate_emission_hazard_s", np.max
        ),
    }


def _selection_metadata(case: Path, option: str) -> dict[str, str]:
    transfer_path = case / "v10_2_27_paper_four_class_parameter_transfer.json"
    transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else {}
    selection = transfer.get("paper_campaign_selection") or {}

    legacy_path = case / "v10_2_22_parameter_selection.json"
    legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else {}

    candidate = str(
        transfer.get("selected_candidate")
        or selection.get("candidate_id")
        or legacy.get("candidate_id")
        or option
    )
    return {
        "candidate_id": candidate,
        "response_class": str(selection.get("response_class", "")),
        "interpretation": str(selection.get("interpretation", "")),
        "short_label": SHORT_LABELS.get(option, candidate.split("_")[-1]),
    }


def _campaign_manifest(outroot: Path) -> dict[str, Any]:
    path = outroot / "v10_2_27_campaign_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _campaign_target(
    outroot: Path, override: float | None, manifest: dict[str, Any]
) -> float:
    if override is not None:
        return float(override)
    value = manifest.get("target_crack_extension_um")
    return float(value) if value is not None else 1000.0


def _target_tag(target_um: float) -> str:
    rounded = round(float(target_um))
    if np.isclose(target_um, rounded, rtol=0.0, atol=1.0e-9):
        return str(int(rounded))
    return f"{target_um:g}".replace(".", "p")


def _ordered_present_options(
    records: list[dict[str, object]], manifest: dict[str, Any]
) -> list[str]:
    present = {str(record["option_key"]) for record in records}
    requested = [str(value) for value in manifest.get("options", DEFAULT_OPTION_ORDER)]
    ordered = [option for option in requested if option in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _plot_event_intervals(
    ax,
    pre: np.ndarray,
    post: np.ndarray,
    resistance: np.ndarray,
    *,
    marker: str = "o",
    markersize: float = 4.0,
    linewidth: float = 1.5,
    color=None,
    label: str | None = None,
) -> None:
    resolved_color = color if color is not None else ax._get_lines.get_next_color()
    for index, (x0, x1, value) in enumerate(zip(pre, post, resistance)):
        ax.plot(
            [x0, x1],
            [value, value],
            linewidth=linewidth,
            color=resolved_color,
            label=label if index == 0 else None,
        )
    ax.plot(
        pre,
        resistance,
        linestyle="none",
        marker=marker,
        markersize=markersize,
        color=resolved_color,
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--target-extension-um", type=float, default=None)
    args = parser.parse_args()

    outroot = Path(args.outroot).expanduser().resolve()
    manifest = _campaign_manifest(outroot)
    target_extension_um = _campaign_target(outroot, args.target_extension_um, manifest)
    checkpoints = checkpoints_for_target(target_extension_um)
    target_tag = _target_tag(target_extension_um)
    plot_dir = (
        Path(args.plot_dir).expanduser().resolve()
        if args.plot_dir
        else outroot / "plots"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    curves: dict[
        tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    labels: dict[str, str] = {}

    for option_dir in sorted(path for path in outroot.iterdir() if path.is_dir()):
        option = option_dir.name
        for case in sorted(path for path in option_dir.iterdir() if path.is_dir()):
            match = CASE_RE.match(case.name)
            if not match or not (case / "COMPLETE").exists():
                continue
            temperature = float(match.group("T"))
            theta = float(match.group("theta"))
            seed = int(match.group("seed"))
            data, source = _load_steps(case)
            pre, post, resistance = _event_curve(data)
            if pre.size == 0:
                continue

            achieved = _achieved_extension_um(data)
            metadata = _selection_metadata(case, option)
            labels[option] = metadata["short_label"]
            curves[(option, temperature)] = (pre, post, resistance)

            row: dict[str, object] = {
                "option_key": option,
                "candidate_id": metadata["candidate_id"],
                "response_class": metadata["response_class"],
                "interpretation": metadata["interpretation"],
                "temperature_K": temperature,
                "theta_deg": theta,
                "seed": seed,
                "campaign_target_extension_um": target_extension_um,
                "steps_file": str(source),
                "n_events": int(pre.size),
                "final_event_start_extension_um": float(pre[-1]),
                "final_event_extension_um": float(post[-1]),
                "achieved_extension_um": achieved,
                "K_first_MPa_sqrt_m": float(resistance[0]),
            }
            for target in checkpoints:
                value = _resistance_at_extension(
                    pre, post, resistance, target, achieved
                )
                key = f"{int(target)}um"
                row[f"K_{key}_MPa_sqrt_m"] = value
                row[f"deltaK_{key}_from_first_MPa_sqrt_m"] = (
                    value - resistance[0] if np.isfinite(value) else float("nan")
                )
            row.update(_case_diagnostics(data))
            row.update(_persistent_diagnostics(case))
            records.append(row)

            fig, ax = plt.subplots(figsize=(7.0, 5.1))
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker="o",
                markersize=3.5,
                linewidth=1.2,
            )
            ax.set_xlabel("Crack extension, Δa (µm)")
            ax.set_ylabel("Event resistance, K (MPa√m)")
            ax.set_title(
                f"{metadata['short_label']} — {temperature:g} K, θ={theta:g}°, "
                f"seed={seed}"
            )
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
            _save(
                fig,
                plot_dir
                / "individual"
                / option
                / f"K_vs_crack_extension_{temperature:g}K_seed{seed}.png",
            )

    if not records:
        raise SystemExit(f"no complete event curves found below {outroot}")

    options = _ordered_present_options(records, manifest)
    temperatures = sorted({float(record["temperature_K"]) for record in records})

    cmap = plt.get_cmap("turbo")
    for option in options:
        available = [temperature for temperature in temperatures if (option, temperature) in curves]
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        for index, temperature in enumerate(available):
            pre, post, resistance = curves[(option, temperature)]
            color = cmap(index / max(len(available) - 1, 1))
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker="o",
                markersize=2.8,
                linewidth=1.0,
                color=color,
                label=f"{temperature:g} K",
            )
        ax.set_xlabel("Crack extension, Δa (µm)")
        ax.set_ylabel("Event resistance, K (MPa√m)")
        ax.set_title(f"{labels.get(option, option)}: K–Δa by temperature")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(ncol=3, fontsize=7.5, frameon=True)
        _save(
            fig,
            plot_dir / "by_candidate" / f"{option}_K_vs_crack_extension.png",
        )

    markers = ["o", "s", "^", "D"]
    for temperature in temperatures:
        available = [option for option in options if (option, temperature) in curves]
        fig, ax = plt.subplots(figsize=(7.6, 5.5))
        for index, option in enumerate(available):
            pre, post, resistance = curves[(option, temperature)]
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker=markers[index % len(markers)],
                markersize=3.2,
                linewidth=1.1,
                label=labels.get(option, option),
            )
        ax.set_xlabel("Crack extension, Δa (µm)")
        ax.set_ylabel("Event resistance, K (MPa√m)")
        ax.set_title(f"{temperature:g} K: four response classes")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, frameon=True)
        _save(
            fig,
            plot_dir
            / "by_temperature"
            / f"K_vs_crack_extension_{temperature:g}K.png",
        )

    option_rank = {option: index for index, option in enumerate(options)}
    records = sorted(
        records,
        key=lambda record: (
            option_rank.get(str(record["option_key"]), len(option_rank)),
            float(record["temperature_K"]),
        ),
    )
    fieldnames = list(records[0])
    summary_stem = f"v10_2_27_paper_four_class_{target_tag}um_summary"
    summary_csv = outroot / f"{summary_stem}.csv"
    with summary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_json = outroot / f"{summary_stem}.json"
    summary_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    expected_count = len(manifest.get("options", [])) * len(
        manifest.get("temperatures_K", [])
    )
    postprocess_audit = {
        "schema": "v10.2.27_four_class_postprocess_audit_v1",
        "target_extension_um": target_extension_um,
        "checkpoints_um": list(checkpoints),
        "complete_curves_processed": len(records),
        "expected_case_count": expected_count or None,
        "all_expected_curves_processed": (
            len(records) == expected_count if expected_count else None
        ),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "plot_directory": str(plot_dir),
    }
    (outroot / "v10_2_27_postprocess_audit.json").write_text(
        json.dumps(postprocess_audit, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {len(records)} case summaries and R-curve plots to {plot_dir}; "
        f"summary={summary_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
