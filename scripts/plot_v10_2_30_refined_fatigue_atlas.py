#!/usr/bin/env python3
"""Plot completed campaign-summary cases without pooling their Paris fits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.summaries) != len(args.labels):
        parser.error("--labels must contain one label per summary")

    rows = []
    for path, label in zip(args.summaries, args.labels, strict=True):
        payload = json.loads(path.read_text())
        for case in payload.get("cases", []):
            row = dict(case)
            row["series"] = label
            row["source_summary"] = str(path.resolve())
            rows.append(row)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (out / "all_refined_fatigue_cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    for label in args.labels:
        group = [row for row in rows if row["series"] == label]
        developed = [
            row for row in group
            if row.get("developed_da_dN_m_per_cycle") not in (None, 0.0)
        ]
        censored = [row for row in group if row not in developed]
        if developed:
            ax.scatter(
                [row["deltaK_MPa_sqrt_m"] for row in developed],
                [row["developed_da_dN_m_per_cycle"] for row in developed],
                s=44,
                label=label,
            )
        if censored:
            floor = min(
                [row["developed_da_dN_m_per_cycle"] for row in rows
                 if row.get("developed_da_dN_m_per_cycle") not in (None, 0.0)],
                default=1.0e-12,
            ) / 3.0
            ax.scatter(
                [row["deltaK_MPa_sqrt_m"] for row in censored
                 if row.get("deltaK_MPa_sqrt_m") is not None],
                [floor for row in censored if row.get("deltaK_MPa_sqrt_m") is not None],
                marker="v", facecolors="none", edgecolors="0.35", s=50,
            )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{\mathrm{m}}$)")
    ax.set_ylabel(r"Developed $da/dN$ (m/cycle)")
    ax.set_title("All measured refined-solver fatigue data")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "all_refined_da_dN_vs_deltaK.png", dpi=220)
    plt.close(fig)

    manifest = {
        "schema": "v10.2.30_refined_fatigue_atlas_plot_v1",
        "case_count": len(rows),
        "series": args.labels,
        "fit_policy": "plot_only_no_cross_resolution_or_cross_seed_pooling",
        "censor_marker": "open downward triangle at a shared visual-only ordinate",
        "csv": "all_refined_fatigue_cases.csv",
        "plot": "all_refined_da_dN_vs_deltaK.png",
    }
    (out / "all_refined_fatigue_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
