#!/usr/bin/env python3
"""Plot sharp-fracture and plastic-terminal energy measures versus temperature.

Closed symbols are the positive configurational J at first sharp-tip cleavage
passage. Open symbols are the cumulative bulk-plastic dissipation intensity at
a plastic-flow terminal. The quantities share units but are not joined as one
continuous fracture-toughness curve.
"""
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

SHORT_LABELS = {
    "v913_paper_peak01_0242980_persistent_sites": "Peak",
    "v913_paper_dbtt01_0202500_persistent_sites": "DBTT",
    "v913_paper_weakT01_0129902_persistent_sites": "Weak-T/FCC-like",
    "v913_paper_ceramic01_0077080_persistent_sites": "Ceramic-like",
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _steps(case: Path) -> np.ndarray:
    paths = sorted(case.glob("steps_*K.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one steps CSV in {case}; found {paths}")
    return np.atleast_1d(
        np.genfromtxt(paths[0], delimiter=",", names=True, dtype=float)
    )


def _fracture_measure(case: Path) -> dict[str, float]:
    data = _steps(case)
    names = set(data.dtype.names or ())
    if "n_fire" not in names or "KJ_Pa_sqrtm" not in names:
        raise ValueError(f"steps file lacks fracture-event fields: {case}")
    fired = np.flatnonzero(np.asarray(data["n_fire"], dtype=float) > 0.0)
    if fired.size == 0:
        raise ValueError(f"complete case has no fracture event: {case}")
    index = int(fired[0])
    K = float(np.asarray(data["KJ_Pa_sqrtm"], dtype=float)[index])
    if "J_effective_direct_J_per_m2" in names:
        J = float(
            np.asarray(data["J_effective_direct_J_per_m2"], dtype=float)[index]
        )
    else:
        J = float("nan")
    return {
        "J_fracture_J_per_m2": J,
        "K_fracture_MPa_sqrt_m": K / 1.0e6,
    }


def collect(outroot: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for option_dir in sorted(path for path in outroot.iterdir() if path.is_dir()):
        option = option_dir.name
        for case in sorted(path for path in option_dir.iterdir() if path.is_dir()):
            match = CASE_RE.match(case.name)
            if not match:
                continue
            status_path = case / "stage3_case_status.json"
            if not status_path.is_file():
                continue
            status = _json(status_path)
            record: dict[str, Any] = {
                "option": option,
                "label": SHORT_LABELS.get(option, option),
                "temperature_K": float(match.group("T")),
                "theta_deg": float(match.group("theta")),
                "seed": int(match.group("seed")),
                "case_root": str(case.resolve()),
                "status": status.get("status"),
                "measure_type": None,
                "plot_symbol": None,
                "J_fracture_J_per_m2": float("nan"),
                "K_fracture_MPa_sqrt_m": float("nan"),
                "J_pl_diss_J_per_m2": float("nan"),
                "K_pl_equivalent_MPa_sqrt_m": float("nan"),
                "J_contour_shielding_J_per_m2": float("nan"),
                "J_tip_positive_final_J_per_m2": float("nan"),
                "J_outer_positive_final_J_per_m2": float("nan"),
            }
            if (case / "COMPLETE").is_file() and status.get("complete") is True:
                record.update(_fracture_measure(case))
                record["measure_type"] = "sharp_tip_fracture"
                record["plot_symbol"] = "closed"
            elif (
                (case / "PLASTIC_FLOW").is_file()
                and status.get("status") == "plastic_flow_no_sharp_fracture"
            ):
                audit = _json(case / "plastic_flow_terminal_audit.json")
                record.update({
                    "measure_type": "bulk_plastic_flow_terminal",
                    "plot_symbol": "open",
                    "J_pl_diss_J_per_m2": audit.get(
                        "J_pl_diss_J_per_m2", float("nan")
                    ),
                    "K_pl_equivalent_MPa_sqrt_m": audit.get(
                        "K_pl_equivalent_MPa_sqrt_m", float("nan")
                    ),
                    "J_contour_shielding_J_per_m2": audit.get(
                        "J_contour_shielding_J_per_m2", float("nan")
                    ),
                    "J_tip_positive_final_J_per_m2": audit.get(
                        "J_tip_positive_final_J_per_m2", float("nan")
                    ),
                    "J_outer_positive_final_J_per_m2": audit.get(
                        "J_outer_positive_final_J_per_m2", float("nan")
                    ),
                })
            else:
                continue
            records.append(record)
    return records


def _write_tables(records: list[dict[str, Any]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    json_path = plot_dir / "v10_4_2_temperature_energy_measures.json"
    csv_path = plot_dir / "v10_4_2_temperature_energy_measures.csv"
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    fields = list(records[0])
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_option(records: list[dict[str, Any]], option: str, plot_dir: Path) -> None:
    subset = sorted(
        (record for record in records if record["option"] == option),
        key=lambda record: float(record["temperature_K"]),
    )
    if not subset:
        return

    fracture = [record for record in subset if record["measure_type"] == "sharp_tip_fracture"]
    plastic = [record for record in subset if record["measure_type"] == "bulk_plastic_flow_terminal"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if fracture:
        ax.scatter(
            [record["temperature_K"] for record in fracture],
            [record["J_fracture_J_per_m2"] for record in fracture],
            marker="o",
            label="Sharp-tip fracture J (closed)",
        )
    if plastic:
        ax.scatter(
            [record["temperature_K"] for record in plastic],
            [record["J_pl_diss_J_per_m2"] for record in plastic],
            marker="o",
            facecolors="none",
            label="Bulk plastic dissipation intensity (open)",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Energy intensity (J m$^{-2}$)")
    ax.set_title(f"{SHORT_LABELS.get(option, option)}: fracture and plastic-flow regimes")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    base = plot_dir / f"J_fracture_and_Jpl_vs_temperature_{option}"
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if fracture:
        ax.scatter(
            [record["temperature_K"] for record in fracture],
            [record["K_fracture_MPa_sqrt_m"] for record in fracture],
            marker="o",
            label="Sharp-tip fracture K (closed)",
        )
    if plastic:
        ax.scatter(
            [record["temperature_K"] for record in plastic],
            [record["K_pl_equivalent_MPa_sqrt_m"] for record in plastic],
            marker="o",
            facecolors="none",
            label="Equivalent plastic dissipation intensity (open)",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Equivalent intensity (MPa√m)")
    ax.set_title(f"{SHORT_LABELS.get(option, option)}: K-scale regime comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    base = plot_dir / f"K_fracture_and_Kpl_vs_temperature_{option}"
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if plastic:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        ax.scatter(
            [record["temperature_K"] for record in plastic],
            [record["J_pl_diss_J_per_m2"] for record in plastic],
            marker="o",
            facecolors="none",
            label="J plastic dissipation",
        )
        ax.scatter(
            [record["temperature_K"] for record in plastic],
            [record["J_contour_shielding_J_per_m2"] for record in plastic],
            marker="s",
            facecolors="none",
            label="J contour shielding",
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Diagnostic energy intensity (J m$^{-2}$)")
        ax.set_title(f"{SHORT_LABELS.get(option, option)}: plasticity diagnostics")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        base = plot_dir / f"Jpl_and_contour_shielding_vs_temperature_{option}"
        fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--plot-dir", type=Path, default=None)
    args = parser.parse_args()

    outroot = args.outroot.expanduser().resolve()
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else outroot / "plots" / "fracture_plastic_temperature"
    )
    records = collect(outroot)
    if not records:
        raise SystemExit(f"no terminal fracture or plastic-flow cases found in {outroot}")
    _write_tables(records, plot_dir)
    for option in sorted({record["option"] for record in records}):
        _plot_option(records, option, plot_dir)
    print(json.dumps({
        "record_count": len(records),
        "fracture_count": sum(
            record["measure_type"] == "sharp_tip_fracture" for record in records
        ),
        "plastic_terminal_count": sum(
            record["measure_type"] == "bulk_plastic_flow_terminal"
            for record in records
        ),
        "plot_dir": str(plot_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
