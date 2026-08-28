#!/usr/bin/env python3
"""Render the complete PF model-native KJ driving-trajectory atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

mpl.rcParams["svg.hashsalt"] = "pf-canonical-full-trajectory-audit-v1"


INPUT_DEFAULT = Path(
    "analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit/"
    "pf_canonical_full_step_trajectories.parquet"
)
OUTPUT_DEFAULT = Path(
    "analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit"
)
RAW_DEFAULT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "canonical_pf_fracture_v2_20260826"
)

TEMPERATURES = [300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
THETAS = [0, 15, 30, 45]
RATES = ["rate0p01x", "rate1x", "rate100x"]
CLASSES = ["Peak", "DBTT", "weak-T", "ceramic-like"]
CLASS_STEM = {
    "Peak": "PEAK",
    "DBTT": "DBTT",
    "weak-T": "WEAKT",
    "ceramic-like": "CERAMICLIKE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def temperature_style() -> tuple[mpl.colors.Colormap, mpl.colors.BoundaryNorm]:
    cmap = mpl.colormaps["viridis"].resampled(len(TEMPERATURES))
    centers = np.asarray(TEMPERATURES, dtype=float)
    bounds = np.r_[
        centers[0] - (centers[1] - centers[0]) / 2,
        (centers[:-1] + centers[1:]) / 2,
        centers[-1] + (centers[-1] - centers[-2]) / 2,
    ]
    return cmap, mpl.colors.BoundaryNorm(bounds, cmap.N)


def color_for(temp: float, cmap: mpl.colors.Colormap) -> tuple[float, ...]:
    return cmap(TEMPERATURES.index(int(round(temp))))


def slim_source(data: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "case_id", "material_class", "temperature_K", "theta_deg", "rate_tag",
        "loading_rate_factor", "seed", "accepted_step_index", "raw_step",
        "physical_time_s", "applied_opening_m", "projected_crack_extension_um",
        "projected_total_crack_length_um", "native_J_J_per_m2",
        "native_KJ_MPa_sqrt_m", "reaction_N", "event_state",
        "crack_event_transaction_index", "physical_avalanche_index",
        "is_initial_onset", "is_reload_separated_reinitiation_onset",
        "is_target_right_censored_endpoint",
    ]
    return data[[column for column in columns if column in data.columns]].copy()


def marker_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D([], [], color="0.2", marker="*", linestyle="None", markersize=9,
               label="Initial onset"),
        Line2D([], [], color="0.2", marker="o", markerfacecolor="none",
               linestyle="None", markersize=5, label="Reload-separated onset"),
        Line2D([], [], color="0.2", marker=">", linestyle="None", markersize=6,
               label="Target-right-censored endpoint"),
    ]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=7)


def plot_panel(ax: plt.Axes, data: pd.DataFrame, cmap: mpl.colors.Colormap) -> None:
    for temp in TEMPERATURES:
        run = data.loc[data.temperature_K.eq(float(temp))]
        if run.empty:
            continue
        if run.case_id.nunique() != 1:
            raise ValueError(f"plot panel has {run.case_id.nunique()} cases at {temp} K")
        run = run.sort_values("accepted_step_index", kind="stable")
        color = color_for(temp, cmap)
        ax.plot(
            run.projected_crack_extension_um,
            run.native_KJ_MPa_sqrt_m,
            color=color,
            linewidth=0.7,
            alpha=0.9,
        )
        initial = run.loc[run.is_initial_onset]
        reload = run.loc[run.is_reload_separated_reinitiation_onset]
        endpoint = run.loc[run.is_target_right_censored_endpoint]
        ax.scatter(initial.projected_crack_extension_um, initial.native_KJ_MPa_sqrt_m,
                   marker="*", s=55, color=[color], edgecolors="0.15", linewidths=0.35,
                   zorder=4)
        ax.scatter(reload.projected_crack_extension_um, reload.native_KJ_MPa_sqrt_m,
                   marker="o", s=22, facecolors="none", edgecolors=[color],
                   linewidths=1.0, zorder=4)
        ax.scatter(endpoint.projected_crack_extension_um, endpoint.native_KJ_MPa_sqrt_m,
                   marker=">", s=28, color=[color], edgecolors="0.15", linewidths=0.3,
                   zorder=4)
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_xlabel(r"Projected crack extension, $\Delta a_{proj}$ [$\mu$m]")
    ax.set_ylabel(r"PF model-native $K_J$ [MPa$\sqrt{\mathrm{m}}$]")


def add_colorbar(fig: plt.Figure, axes: list[plt.Axes], cmap, norm) -> None:
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, ax=axes, ticks=TEMPERATURES, pad=0.02, fraction=0.035)
    cbar.set_label("Temperature [K]")
    cbar.ax.tick_params(labelsize=7)


def save_figure(
    fig: plt.Figure,
    stem: str,
    figure_dir: Path,
    source_dir: Path,
    source: pd.DataFrame,
    records: list[dict[str, object]],
    category: str,
    variant: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{stem}.parquet"
    slim_source(source).to_parquet(
        source_path, index=False, compression="zstd", compression_level=19
    )
    paths: dict[str, str] = {}
    formats = (
        ("pdf", {"metadata": {"Creator": "PF canonical audit", "CreationDate": None,
                               "ModDate": None}}),
        ("svg", {"metadata": {"Creator": "PF canonical audit", "Date": None}}),
        ("png", {"dpi": 600, "metadata": {"Software": "PF canonical audit"}}),
    )
    for suffix, kwargs in formats:
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[suffix] = str(path)
    plt.close(fig)
    records.append(
        {
            "stem": stem,
            "category": category,
            "variant": variant,
            "case_count": int(source.case_id.nunique()),
            "temperature_count": int(source.temperature_K.nunique()),
            "source_data": str(source_path),
            "source_data_sha256": sha256(source_path),
            "outputs": {kind: {"path": path, "sha256": sha256(Path(path))}
                        for kind, path in paths.items()},
        }
    )


def individual(
    data: pd.DataFrame,
    stem_base: str,
    title: str,
    figure_dir: Path,
    source_dir: Path,
    records: list[dict[str, object]],
    category: str,
    available_temperatures: int = 12,
) -> None:
    cmap, norm = temperature_style()
    if data.temperature_K.nunique() != available_temperatures:
        raise ValueError(f"{stem_base}: expected {available_temperatures} temperatures")
    for variant, xlim in (("FULL", (0, 1050)), ("EARLY", (0, 150))):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        plot_panel(ax, data, cmap)
        ax.set_xlim(*xlim)
        ax.set_title(f"{title}\nPF MODEL-NATIVE KJ TRAJECTORY — {variant} RANGE", fontsize=10)
        marker_legend(ax)
        add_colorbar(fig, [ax], cmap, norm)
        stem = f"{stem_base}_{variant}"
        plotted_source = data.loc[data.projected_crack_extension_um.le(xlim[1])]
        save_figure(
            fig, stem, figure_dir, source_dir, plotted_source, records, category, variant
        )


def composite(
    panel_data: list[tuple[str, pd.DataFrame]],
    stem_base: str,
    title: str,
    figure_dir: Path,
    source_dir: Path,
    records: list[dict[str, object]],
    category: str,
) -> None:
    cmap, norm = temperature_style()
    source = pd.concat([frame for _, frame in panel_data], ignore_index=True)
    ymax = float(source.native_KJ_MPa_sqrt_m.max()) * 1.04
    n = len(panel_data)
    for variant, xlim in (("FULL", (0, 1050)), ("EARLY", (0, 150))):
        fig, axes = plt.subplots(
            1, n, figsize=(4.5 * n, 4.6), sharex=True, sharey=True,
            constrained_layout=True,
        )
        axes_list = np.atleast_1d(axes).tolist()
        for ax, (label, frame) in zip(axes_list, panel_data):
            if frame.temperature_K.nunique() != 12:
                raise ValueError(f"{stem_base} panel {label} does not have 12 temperatures")
            plot_panel(ax, frame, cmap)
            ax.set_title(label)
            ax.set_xlim(*xlim)
            ax.set_ylim(0, ymax)
        marker_legend(axes_list[0])
        add_colorbar(fig, axes_list, cmap, norm)
        fig.suptitle(
            f"{title}\nPF MODEL-NATIVE KJ TRAJECTORY — {variant} RANGE", fontsize=11
        )
        stem = f"{stem_base}_{variant}"
        plotted_source = source.loc[source.projected_crack_extension_um.le(xlim[1])]
        save_figure(
            fig, stem, figure_dir, source_dir, plotted_source, records, category, variant
        )


def load_supplemental(raw_root: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"canonical_strain_rate__(?P<class>peak|dbtt|weakt|ceramiclike)__"
        r"T(?P<T>\d{4})K__theta45__rate0p01x__seed(?P<seed>\d+)$"
    )
    rows: list[pd.DataFrame] = []
    labels = {"peak": "Peak", "dbtt": "DBTT", "weakt": "weak-T",
              "ceramiclike": "ceramic-like"}
    for directory in sorted(raw_root.glob("canonical_strain_rate__*__theta45__rate0p01x__seed*")):
        match = pattern.match(directory.name)
        if not match:
            continue
        temperature = int(match.group("T"))
        files = list(directory.glob("steps_*K.csv"))
        if len(files) != 1:
            raise ValueError(f"supplemental steps file is not unique in {directory}")
        raw = pd.read_csv(files[0])
        fired = raw.n_fire.fillna(0).astype(int).to_numpy() > 0
        fire_indices = np.flatnonzero(fired)
        initial = np.zeros(len(raw), dtype=bool)
        reload = np.zeros(len(raw), dtype=bool)
        if fire_indices.size:
            initial[fire_indices[0]] = True
            for index in fire_indices[1:]:
                previous_event = fire_indices[fire_indices < index][-1]
                reload[index] = bool((~fired[previous_event + 1:index]).any())
        endpoint = np.zeros(len(raw), dtype=bool)
        endpoint[-1] = True
        frame = pd.DataFrame(
            {
                "case_id": directory.name,
                "material_class": labels[match.group("class")],
                "temperature_K": float(temperature),
                "theta_deg": 45.0,
                "rate_tag": "rate0p01x",
                "loading_rate_factor": 0.01,
                "seed": int(match.group("seed")),
                "accepted_step_index": np.arange(len(raw)),
                "raw_step": raw.step.astype(int),
                "physical_time_s": np.cumsum(raw.dt_cur_s.astype(float)),
                "applied_opening_m": raw.Uapp_m.astype(float),
                "projected_crack_extension_um": raw.crack_extension_m.astype(float) * 1e6,
                "projected_total_crack_length_um": raw.a_tip_m.astype(float) * 1e6,
                "native_J_J_per_m2": raw.J_effective_direct_J_per_m2.astype(float),
                "native_KJ_MPa_sqrt_m": raw.KJ_Pa_sqrtm.astype(float) / 1e6,
                "reaction_N": raw.Ftop_N.astype(float),
                "event_state": np.where(fired, "EVENT", "ACCEPTED_LOADING_STATE"),
                "crack_event_transaction_index": np.where(
                    fired, np.cumsum(fired) - 1, np.nan
                ),
                "physical_avalanche_index": np.nan,
                "is_initial_onset": initial,
                "is_reload_separated_reinitiation_onset": reload,
                "is_target_right_censored_endpoint": endpoint,
            }
        )
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if result.case_id.nunique() != 42:
        raise ValueError(f"expected 42 supplemental cases, found {result.case_id.nunique()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    args = parser.parse_args()
    data = pd.read_parquet(args.input)
    if data.case_id.nunique() != 288:
        raise ValueError("trajectory atlas input is not the 288-case canonical table")

    figures = args.output / "figures/full_KJ_trajectories"
    source = args.output / "figure_source_data/full_KJ_trajectories"
    records: list[dict[str, object]] = []

    for material in CLASSES:
        class_data = data.loc[data.material_class.eq(material)]
        orientation_panels = []
        for theta in THETAS:
            frame = class_data.loc[
                class_data.theta_deg.eq(float(theta)) & class_data.rate_tag.eq("rate1x")
            ]
            base = f"{CLASS_STEM[material]}_THETA{theta}_RATE1X_ALL_T_KJ_VS_CRACK_LENGTH"
            individual(
                frame, base, f"{material}: theta={theta} deg, rate=1x",
                figures / "orientation", source / "orientation", records,
                "canonical_orientation_individual",
            )
            orientation_panels.append((f"theta={theta} deg", frame))
        composite(
            orientation_panels, f"{CLASS_STEM[material]}_ORIENTATION_COMPOSITE",
            f"{material}: orientation comparison at rate=1x",
            figures / "composites", source / "composites", records,
            "canonical_orientation_composite",
        )

        rate_panels = []
        for rate in RATES:
            frame = class_data.loc[
                class_data.theta_deg.eq(0.0) & class_data.rate_tag.eq(rate)
            ]
            base = f"{CLASS_STEM[material]}_THETA0_{rate.upper()}_ALL_T_KJ_VS_CRACK_LENGTH"
            individual(
                frame, base, f"{material}: theta=0 deg, {rate}",
                figures / "rate", source / "rate", records,
                "canonical_rate_individual",
            )
            rate_panels.append((rate, frame))
        composite(
            rate_panels, f"{CLASS_STEM[material]}_RATE_COMPOSITE",
            f"{material}: rate comparison at theta=0 deg",
            figures / "composites", source / "composites", records,
            "canonical_rate_composite",
        )

    supplemental = load_supplemental(args.raw_root)
    supplemental.to_parquet(
        args.output / "pf_theta45_rate0p01x_supplemental_full_trajectories.parquet",
        index=False, compression="zstd",
    )
    for material in CLASSES:
        frame = supplemental.loc[supplemental.material_class.eq(material)]
        count = 6 if material == "ceramic-like" else 12
        status = "INCOMPLETE — 6 TEMPERATURES" if count == 6 else "12 TEMPERATURES"
        base = f"{CLASS_STEM[material]}_THETA45_RATE0P01X_SUPPLEMENTAL_KJ_VS_CRACK_LENGTH"
        individual(
            frame, base,
            f"{material}: theta=45 deg, rate=0.01x ({status})\n"
            "SUPPLEMENTAL CURRENT-SOURCE DATA — INCOMPLETE RATE–ORIENTATION MATRIX",
            figures / "supplemental", source / "supplemental", records,
            "supplemental_theta45_rate0p01x", available_temperatures=count,
        )

    manifest = {
        "schema": "pf_canonical_full_KJ_atlas_manifest_v1",
        "quantity": "PF_MODEL_NATIVE_KJ_MPa_sqrt_m",
        "quantity_is_applied_K": False,
        "quantity_is_conventional_R_curve": False,
        "chronological_lines_preserve_duplicate_extension": True,
        "fixed_temperature_colormap": "viridis_discrete_12_level_300_to_1300K",
        "canonical_case_count": int(data.case_id.nunique()),
        "supplemental_case_count": int(supplemental.case_id.nunique()),
        "canonical_orientation_individual_plot_count": 16,
        "canonical_rate_individual_plot_count": 12,
        "canonical_orientation_composite_count": 4,
        "canonical_rate_composite_count": 4,
        "supplemental_plot_count": 4,
        "full_and_early_versions": True,
        "figure_records": records,
    }
    path = args.output / "pf_canonical_full_KJ_atlas_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"rendered {len(records)} figures ({len(records) * 3} image files)")


if __name__ == "__main__":
    main()
