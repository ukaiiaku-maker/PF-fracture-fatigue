#!/usr/bin/env python3
"""Event-resolved K-R curves for the v10.2.26 two-candidate paper campaign."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(r"T(?P<T>\d+(?:\.\d+)?)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$")
CHECKPOINTS_UM = (10.0, 25.0, 50.0, 75.0, 100.0)


def _load_steps(case: Path) -> tuple[np.ndarray, Path]:
    files = sorted(case.glob("steps_*K.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one steps CSV in {case}; found {files}")
    data = np.atleast_1d(np.genfromtxt(files[0], delimiter=",", names=True, dtype=float))
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
            "mpz_K_shield_Pa_sqrt_m", lambda v: np.max(np.abs(v)), 1.0e6
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
    transfer_path = case / "v10_2_25_v913_paper_campaign_parameter_transfer.json"
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
    response_class = str(selection.get("response_class", ""))
    interpretation = str(selection.get("interpretation", ""))
    if "peak01" in option:
        short_label = "Peak 0242980"
    elif "dbtt01" in option:
        short_label = "DBTT 0202500"
    else:
        short_label = candidate.split("_")[-1]

    return {
        "candidate_id": candidate,
        "response_class": response_class,
        "interpretation": interpretation,
        "short_label": short_label,
    }


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
    args = parser.parse_args()

    outroot = Path(args.outroot).expanduser().resolve()
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

    for option_dir in sorted(p for p in outroot.iterdir() if p.is_dir()):
        option = option_dir.name
        for case in sorted(p for p in option_dir.iterdir() if p.is_dir()):
            match = CASE_RE.match(case.name)
            if not match or not (case / "COMPLETE").exists():
                continue
            temperature = float(match.group("T"))
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
                "seed": int(match.group("seed")),
                "steps_file": str(source),
                "n_events": int(pre.size),
                "final_event_start_extension_um": float(pre[-1]),
                "final_event_extension_um": float(post[-1]),
                "achieved_extension_um": achieved,
                "K_first_MPa_sqrt_m": float(resistance[0]),
            }
            for target in CHECKPOINTS_UM:
                value = _resistance_at_extension(pre, post, resistance, target, achieved)
                row[f"K_{int(target)}um_MPa_sqrt_m"] = value
                row[f"deltaK_{int(target)}um_from_first_MPa_sqrt_m"] = (
                    value - resistance[0] if np.isfinite(value) else float("nan")
                )
            row.update(_case_diagnostics(data))
            row.update(_persistent_diagnostics(case))
            records.append(row)

            fig, ax = plt.subplots(figsize=(6.8, 5.0))
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker="o",
                markersize=4.5,
                linewidth=1.5,
            )
            ax.set_xlabel("Crack extension, Δa (µm)")
            ax.set_ylabel("Event resistance, K (MPa√m)")
            ax.set_title(f"{metadata['short_label']} — {temperature:g} K")
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
            _save(
                fig,
                plot_dir
                / "individual"
                / option
                / f"K_vs_crack_extension_{temperature:07.2f}K.png",
            )

    if not records:
        raise SystemExit(f"no complete event curves found below {outroot}")

    options = sorted({str(record["option_key"]) for record in records})
    temperatures = sorted({float(record["temperature_K"]) for record in records})

    cmap = plt.get_cmap("turbo")
    for option in options:
        available = [T for T in temperatures if (option, T) in curves]
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        for index, temperature in enumerate(available):
            pre, post, resistance = curves[(option, temperature)]
            color = cmap(index / max(len(available) - 1, 1))
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker="o",
                markersize=3.4,
                linewidth=1.2,
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

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for temperature in temperatures:
        available = [option for option in options if (option, temperature) in curves]
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        for index, option in enumerate(available):
            pre, post, resistance = curves[(option, temperature)]
            _plot_event_intervals(
                ax,
                pre,
                post,
                resistance,
                marker=markers[index % len(markers)],
                markersize=4.0,
                linewidth=1.3,
                label=labels.get(option, option),
            )
        ax.set_xlabel("Crack extension, Δa (µm)")
        ax.set_ylabel("Event resistance, K (MPa√m)")
        ax.set_title(f"{temperature:g} K: preferred peak and DBTT")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, frameon=True)
        _save(
            fig,
            plot_dir
            / "by_temperature"
            / f"K_vs_crack_extension_{temperature:07.2f}K.png",
        )

    records = sorted(
        records,
        key=lambda record: (
            str(record["option_key"]),
            float(record["temperature_K"]),
        ),
    )
    fieldnames = list(records[0])
    summary_csv = outroot / "v10_2_26_paper_top2_100um_summary.csv"
    with summary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    (outroot / "v10_2_26_paper_top2_100um_summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(records)} case summaries and R-curve plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
