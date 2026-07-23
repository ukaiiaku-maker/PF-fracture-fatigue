#!/usr/bin/env python3
"""Plot event-based K resistance curves for the v10.2.22 DBTT screen."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(r"T(?P<T>\d+)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$")


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


def _event_curve(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fired = np.asarray(data["n_fire"], dtype=float) > 0.0
    if not np.any(fired):
        return np.array([], dtype=float), np.array([], dtype=float)
    x = 1.0e6 * (
        np.asarray(data["crack_extension_m"], dtype=float)[fired]
        - np.asarray(data["da_block_m"], dtype=float)[fired]
    )
    y = 1.0e-6 * np.asarray(data["KJ_Pa_sqrtm"], dtype=float)[fired]
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.maximum(x[valid], 0.0)
    y = y[valid]
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _interp_left(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if x.size == 0:
        return float("nan")
    if target <= x[0]:
        return float(y[0])
    if target >= x[-1]:
        return float(y[-1])
    return float(np.interp(target, x, y))


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
        "max_shield_MPa_sqrt_m": finite_stat(
            "mpz_K_shield_Pa_sqrt_m", lambda v: np.max(np.abs(v)), 1.0e6
        ),
    }


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
    curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    labels: dict[str, str] = {}

    for option_dir in sorted(p for p in outroot.iterdir() if p.is_dir()):
        option = option_dir.name
        for case in sorted(p for p in option_dir.iterdir() if p.is_dir()):
            match = CASE_RE.match(case.name)
            if not match or not (case / "COMPLETE").exists():
                continue
            temperature = int(match.group("T"))
            data, source = _load_steps(case)
            x, y = _event_curve(data)
            if x.size == 0:
                continue
            selection_file = case / "v10_2_22_parameter_selection.json"
            selection = json.loads(selection_file.read_text()) if selection_file.exists() else {}
            candidate = str(selection.get("candidate_id", option))
            role = str(selection.get("role", candidate))
            labels[option] = f"{candidate.split('_')[-1]}"
            curves[(option, temperature)] = (x, y)
            diag = _case_diagnostics(data)
            records.append(
                {
                    "option_key": option,
                    "candidate_id": candidate,
                    "role": role,
                    "temperature_K": temperature,
                    "seed": int(match.group("seed")),
                    "steps_file": str(source),
                    "n_events": int(x.size),
                    "final_event_extension_um": float(x[-1]),
                    "K_first_MPa_sqrt_m": float(y[0]),
                    "K_10um_MPa_sqrt_m": _interp_left(x, y, 10.0),
                    "K_25um_MPa_sqrt_m": _interp_left(x, y, 25.0),
                    "K_50um_MPa_sqrt_m": _interp_left(x, y, 50.0),
                    **diag,
                }
            )

            fig, ax = plt.subplots(figsize=(6.8, 5.0))
            ax.plot(x, y, marker="o", markersize=4.5, linewidth=1.5)
            ax.set_xlabel("Crack extension, Δa (µm)")
            ax.set_ylabel("Event resistance, K (MPa√m)")
            ax.set_title(f"{candidate} — {temperature} K")
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
            _save(
                fig,
                plot_dir / "individual" / option / f"K_vs_crack_extension_{temperature:04d}K.png",
            )

    if not records:
        raise SystemExit(f"no complete event curves found below {outroot}")

    options = sorted({str(r["option_key"]) for r in records})
    temperatures = sorted({int(r["temperature_K"]) for r in records})

    cmap = plt.get_cmap("turbo")
    for option in options:
        available = [T for T in temperatures if (option, T) in curves]
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        for index, temperature in enumerate(available):
            x, y = curves[(option, temperature)]
            color = cmap(index / max(len(available) - 1, 1))
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.6,
                linewidth=1.25,
                color=color,
                label=f"{temperature} K",
            )
        ax.set_xlabel("Crack extension, Δa (µm)")
        ax.set_ylabel("Event resistance, K (MPa√m)")
        ax.set_title(f"{labels.get(option, option)}: K–Δa by temperature")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(ncol=2, fontsize=8, frameon=True)
        _save(fig, plot_dir / "by_candidate" / f"{option}_K_vs_crack_extension.png")

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for temperature in temperatures:
        available = [option for option in options if (option, temperature) in curves]
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        for index, option in enumerate(available):
            x, y = curves[(option, temperature)]
            ax.plot(
                x,
                y,
                marker=markers[index % len(markers)],
                markersize=3.8,
                linewidth=1.25,
                label=labels.get(option, option),
            )
        ax.set_xlabel("Crack extension, Δa (µm)")
        ax.set_ylabel("Event resistance, K (MPa√m)")
        ax.set_title(f"{temperature} K: candidate K–Δa comparison")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8, frameon=True)
        _save(fig, plot_dir / "by_temperature" / f"K_vs_crack_extension_{temperature:04d}K.png")

    fieldnames = list(records[0])
    summary_csv = outroot / "v10_2_22_dbtt_50um_screen_summary.csv"
    with summary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: (str(r["option_key"]), int(r["temperature_K"]))))

    (outroot / "v10_2_22_dbtt_50um_screen_summary.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(records)} case summaries and R-curve plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
