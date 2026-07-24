#!/usr/bin/env python3
"""Compare v9.13 upper-shelf 1-D K50 predictions with v10.2.24 2-D results."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_RE = re.compile(r"K50_T(?P<T>[0-9]+)K_MPa_sqrt_m$")


def long_reference(path: Path) -> pd.DataFrame:
    wide = pd.read_csv(path)
    columns = []
    for name in wide.columns:
        match = REFERENCE_RE.match(name)
        if match:
            columns.append((float(match.group("T")), name))
    if not columns:
        raise RuntimeError("1-D reference has no K50_T...K_MPa_sqrt_m columns")
    records = []
    for _, row in wide.iterrows():
        for temperature, name in sorted(columns):
            records.append(
                {
                    "shelf_rank": int(row["shelf_rank"]),
                    "candidate_id": str(row["candidate_id"]),
                    "temperature_K": temperature,
                    "K50_1d_MPa_sqrt_m": float(row[name]),
                    "directional_dbtt_gain_1d_MPa_sqrt_m": float(
                        row["y__directional_dbtt_gain"]
                    ),
                    "upper_shelf_1d_MPa_sqrt_m": float(
                        row["y__high_temperature_plateau"]
                    ),
                    "peak_prominence_1d_MPa_sqrt_m": float(
                        row["y__peak_prominence"]
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-d-summary", type=Path, required=True)
    parser.add_argument("--one-d-reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    two = pd.read_csv(args.two_d_summary)
    one = long_reference(args.one_d_reference)
    required_two = {"candidate_id", "temperature_K", "K_50um_MPa_sqrt_m"}
    if missing := sorted(required_two - set(two.columns)):
        raise RuntimeError(f"2-D summary missing columns: {missing}")

    two = two.rename(columns={"K_50um_MPa_sqrt_m": "K50_2d_MPa_sqrt_m"})
    merged = one.merge(
        two[["option_key", "candidate_id", "temperature_K", "K50_2d_MPa_sqrt_m"]],
        on=["candidate_id", "temperature_K"],
        how="left",
        validate="one_to_one",
    )
    merged["delta_K50_2d_minus_1d_MPa_sqrt_m"] = (
        merged["K50_2d_MPa_sqrt_m"] - merged["K50_1d_MPa_sqrt_m"]
    )
    merged["absolute_error_MPa_sqrt_m"] = np.abs(
        merged["delta_K50_2d_minus_1d_MPa_sqrt_m"]
    )
    merged["relative_error"] = np.divide(
        merged["delta_K50_2d_minus_1d_MPa_sqrt_m"],
        merged["K50_1d_MPa_sqrt_m"],
        out=np.full(len(merged), np.nan),
        where=np.abs(merged["K50_1d_MPa_sqrt_m"]) > 0.0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        args.out_dir / "v10_2_24_upper_shelf_1d_2d_K50_comparison.csv",
        index=False,
    )

    summaries = []
    for candidate, local in merged.groupby("candidate_id", sort=False):
        finite = local.dropna(subset=["K50_1d_MPa_sqrt_m", "K50_2d_MPa_sqrt_m"])
        summaries.append(
            {
                "shelf_rank": int(local["shelf_rank"].iloc[0]),
                "candidate_id": candidate,
                "matched_temperatures": len(finite),
                "MAE_MPa_sqrt_m": finite["absolute_error_MPa_sqrt_m"].mean(),
                "RMSE_MPa_sqrt_m": np.sqrt(
                    np.mean(np.square(finite["delta_K50_2d_minus_1d_MPa_sqrt_m"]))
                ) if len(finite) else np.nan,
                "maximum_absolute_error_MPa_sqrt_m": finite[
                    "absolute_error_MPa_sqrt_m"
                ].max(),
            }
        )

        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        local = local.sort_values("temperature_K")
        ax.plot(
            local["temperature_K"],
            local["K50_1d_MPa_sqrt_m"],
            marker="o",
            label="1-D",
        )
        ax.plot(
            local["temperature_K"],
            local["K50_2d_MPa_sqrt_m"],
            marker="s",
            label="2-D",
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$K_{50\,\mu m}$ (MPa$\sqrt{m}$)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{candidate}_K50_1d_vs_2d.png", dpi=240)
        fig.savefig(args.out_dir / f"{candidate}_K50_1d_vs_2d.pdf")
        plt.close(fig)

    summary = pd.DataFrame(summaries).sort_values("shelf_rank")
    summary.to_csv(
        args.out_dir / "v10_2_24_upper_shelf_1d_2d_candidate_errors.csv",
        index=False,
    )
    print(
        "V10224_UPPER_SHELF_1D_2D_COMPARISON "
        f"candidates={len(summary)} matched_cases={merged['K50_2d_MPa_sqrt_m'].notna().sum()} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
