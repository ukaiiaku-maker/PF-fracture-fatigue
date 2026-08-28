#!/usr/bin/env python3
"""Create the production-ready PF rate/orientation mechanism figure set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

mpl.rcParams["svg.hashsalt"] = "pf-canonical-mechanism-audit-v1"

from plot_pf_canonical_full_trajectory_atlas import (
    add_colorbar, plot_panel, temperature_style,
)


OUTPUT_DEFAULT = Path("analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit")
DEEP_T = [900, 1000, 1050, 1100, 1150, 1200]
RATE_COLORS = {"rate0p01x": "#0072B2", "rate1x": "#009E73", "rate100x": "#D55E00"}
RATE_LABELS = {"rate0p01x": "0.01x", "rate1x": "1x", "rate100x": "100x"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(fig, stem: str, source: pd.DataFrame, output: Path, records: list[dict]):
    figdir = output / "figures/mechanisms"
    sourcedir = output / "figure_source_data/mechanisms"
    figdir.mkdir(parents=True, exist_ok=True)
    sourcedir.mkdir(parents=True, exist_ok=True)
    source_path = sourcedir / f"{stem}.parquet"
    source.to_parquet(
        source_path, index=False, compression="zstd", compression_level=19
    )
    files = {}
    formats = (
        ("pdf", {"metadata": {"Creator": "PF canonical audit", "CreationDate": None,
                               "ModDate": None}}),
        ("svg", {"metadata": {"Creator": "PF canonical audit", "Date": None}}),
        ("png", {"dpi": 600, "metadata": {"Software": "PF canonical audit"}}),
    )
    for suffix, options in formats:
        path = figdir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        files[suffix] = {"path": str(path), "sha256": sha256(path)}
    plt.close(fig)
    records.append({
        "stem": stem, "source_data": str(source_path),
        "source_data_sha256": sha256(source_path), "outputs": files,
    })


def rate_trajectory(data, early: bool):
    subset = data.loc[
        data.material_class.eq("Peak") & data.theta_deg.eq(0.0)
        & data.is_rate_matrix_case
    ]
    cmap, norm = temperature_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, rate in zip(axes, RATE_COLORS):
        frame = subset.loc[subset.rate_tag.eq(rate)]
        plot_panel(ax, frame, cmap)
        ax.set_title(RATE_LABELS[rate])
        ax.set_xlim(0, 150 if early else 1050)
    marker_handles = [
        Line2D([], [], color="0.2", marker="*", linestyle="None", label="Initial onset"),
        Line2D([], [], color="0.2", marker="o", markerfacecolor="none", linestyle="None",
               label="Reload-separated onset"),
        Line2D([], [], color="0.2", marker=">", linestyle="None",
               label="Target-right-censored endpoint"),
    ]
    axes[0].legend(handles=marker_handles, frameon=False, fontsize=7)
    add_colorbar(fig, list(axes), cmap, norm)
    fig.suptitle("Peak theta=0 deg: PF model-native KJ driving trajectories")
    if early:
        subset = subset.loc[subset.projected_crack_extension_um.le(150)].copy()
    return fig, subset


def preinit_state(pre: pd.DataFrame):
    source = pre.loc[pre.temperature_K.isin(DEEP_T)].copy()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    specifications = [
        ("observer_persistent_tip_radius_m", 1e6, "Tip radius [um]"),
        ("observer_developed_state_mobile_count", 1.0, "Mobile population"),
        ("observer_developed_state_retained_count", 1.0, "Retained population"),
        ("observer_persistent_sigma_back_Pa", 1e-9, "Backstress [GPa]"),
        ("observer_active_K_shield_signed_Pa_sqrt_m", 1e-6,
         "Signed shielding [MPa sqrt(m)]"),
        ("observer_persistent_site_multiplicity_per_system", 1.0,
         "Multiplicity per system"),
    ]
    cmap, _ = temperature_style()
    for ax, (field, scale, ylabel) in zip(axes, specifications):
        for temperature in DEEP_T:
            for rate, color in RATE_COLORS.items():
                run = source.loc[
                    source.temperature_K.eq(float(temperature)) & source.rate_tag.eq(rate)
                ].sort_values("accepted_step_index")
                ax.plot(run.applied_opening_m * 1e6, run[field].astype(float) * scale,
                        color=cmap([300,600,800,900,950,1000,1050,1100,1150,1200,1250,1300].index(temperature)),
                        linestyle={"rate0p01x":"-", "rate1x":"--", "rate100x":":"}[rate],
                        linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Applied opening [um]")
        ax.grid(True, color="0.9")
    axes[0].legend(handles=[Line2D([], [], color="0.2", linestyle=style, label=RATE_LABELS[rate])
                            for rate, style in zip(RATE_COLORS, ["-", "--", ":"])],
                   frameon=False, fontsize=8)
    fig.suptitle("Peak theta=0 deg pre-initiation state versus opening (900–1200 K)")
    return fig, source


def competition(pre: pd.DataFrame):
    source = pre.loc[pre.temperature_K.isin(DEEP_T)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    cmap, _ = temperature_style()
    fields = [
        ("observer_lambda_c_s-1", "Cleavage rate [s^-1]", True),
        ("observer_tip_source_emission_rate_s", "Aggregate emission rate [s^-1]", True),
        ("observer_hazard_action_current", "Cleavage cumulative action", False),
    ]
    for ax, (field, ylabel, log) in zip(axes, fields):
        for temperature in DEEP_T:
            for rate in RATE_COLORS:
                run = source.loc[
                    source.temperature_K.eq(float(temperature)) & source.rate_tag.eq(rate)
                ].sort_values("accepted_step_index")
                values = run[field].astype(float)
                if log:
                    values = np.maximum(values, 1e-300)
                ax.plot(run.applied_opening_m * 1e6, values,
                        color=cmap([300,600,800,900,950,1000,1050,1100,1150,1200,1250,1300].index(temperature)),
                        linestyle={"rate0p01x":"-", "rate1x":"--", "rate100x":":"}[rate],
                        linewidth=0.8)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Applied opening [um]")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9")
    axes[0].legend(handles=[Line2D([], [], color="0.2", linestyle=style, label=RATE_LABELS[rate])
                            for rate, style in zip(RATE_COLORS, ["-", "--", ":"])],
                   frameon=False, fontsize=8)
    fig.suptitle("Peak theta=0 deg cleavage/emission competition (900–1200 K)")
    return fig, source


def rate_decomposition(table: pd.DataFrame):
    source = table.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for rate in RATE_COLORS:
        group = source.loc[source.rate_tag.eq(rate)].sort_values("temperature_K")
        axes[0].plot(group.temperature_K, group.onset_KJ_MPa_sqrt_m, marker="o",
                     color=RATE_COLORS[rate], label=RATE_LABELS[rate])
    slow = source.loc[source.rate_tag.eq("rate0p01x")].sort_values("temperature_K")
    axes[1].plot(slow.temperature_K, slow.required_opening_contribution_MPa_sqrt_m,
                 marker="o", color="#0072B2", label="Required opening/local state")
    axes[1].plot(slow.temperature_K,
                 slow.structural_KJ_per_opening_contribution_MPa_sqrt_m,
                 marker="s", color="#D55E00", label="Structural KJ/opening")
    axes[0].set_ylabel("Initial PF model-native KJ [MPa sqrt(m)]")
    axes[1].set_ylabel("Slow minus 1x contribution [MPa sqrt(m)]")
    for ax in axes:
        ax.set_xlabel("Temperature [K]"); ax.grid(True, color="0.9"); ax.legend(frameon=False)
    fig.suptitle("Peak theta=0 deg rate-onset decomposition")
    return fig, source


def orientation_initial_reinit(initial, reinit, material: str, temperatures: list[int]):
    source_i = initial.loc[
        initial.material_class.eq(material) & initial.temperature_K.isin(temperatures)
    ].copy()
    source_r = reinit.loc[
        reinit.material_class.eq(material) & reinit.temperature_K.isin(temperatures)
    ].copy()
    source_i["record_kind"] = "INITIAL_ONSET"
    source_r["record_kind"] = "REINITIATION_ONSET"
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    cmap, norm = temperature_style()
    temp_list = [300,600,800,900,950,1000,1050,1100,1150,1200,1250,1300]
    for temperature in temperatures:
        color = cmap(temp_list.index(temperature))
        group = source_i.loc[source_i.temperature_K.eq(float(temperature))].sort_values("theta_deg")
        ax.plot(group.theta_deg, group.K0_theta_MPa_sqrt_m, color=color, marker="*",
                linewidth=0.9, label=f"{temperature} K")
        rr = source_r.loc[source_r.temperature_K.eq(float(temperature))]
        ax.scatter(rr.theta_deg, rr.reinitiation_KJ_MPa_sqrt_m, facecolors="none",
                   edgecolors=[color], marker="o", s=35)
    ax.set_xticks([0,15,30,45]); ax.set_xlabel("Crystal orientation theta [deg]")
    ax.set_ylabel("PF model-native onset KJ [MPa sqrt(m)]")
    ax.grid(True, color="0.9"); ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title(f"{material}: initial stars and reload-separated onset circles")
    return fig, pd.concat([source_i, source_r], ignore_index=True, sort=False)


def structural_transfer(swap: pd.DataFrame):
    source = swap.loc[swap.tip_radius_um.eq(4.0)].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for theta, group in source.groupby("theta_deg"):
        ax.plot(group.extension_um, group.KJ_native_over_U * 1e-6,
                label=f"theta={theta:g} deg", linewidth=1.1)
    ax.set_xlabel("Projected crack extension [um]")
    ax.set_ylabel("Frozen structural KJ/opening [MPa sqrt(m) / um]")
    ax.grid(True, color="0.9"); ax.legend(frameon=False)
    ax.set_title("Zero-history sharp-wake structural transfer coefficient")
    return fig, source


def local_tensor(swap: pd.DataFrame):
    source = swap.loc[swap.tip_radius_um.eq(4.0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for theta, group in source.groupby("theta_deg"):
        axes[0].plot(group.extension_um, group.tau_signed_system_0_Pa / 1e9,
                     label=f"theta={theta:g} deg, system 0")
        axes[0].plot(group.extension_um, group.tau_signed_system_1_Pa / 1e9,
                     linestyle="--", label=f"theta={theta:g} deg, system 1")
        selected = np.where(
            np.abs(group.tau_signed_system_0_Pa) >= np.abs(group.tau_signed_system_1_Pa), 0, 1
        )
        axes[1].plot(group.extension_um, selected, label=f"theta={theta:g} deg")
    axes[0].set_ylabel("Resolved signed shear [GPa]")
    axes[1].set_ylabel("Maximum-|shear| channel index")
    for ax in axes:
        ax.set_xlabel("Projected crack extension [um]"); ax.grid(True, color="0.9")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Deterministic local tensor/source-channel orientation diagnostic")
    return fig, source


def reinit_decomposition(reinit: pd.DataFrame):
    source = reinit.copy()
    summary = source.groupby(["material_class", "theta_deg"], as_index=False).agg(
        required_opening=("required_opening_local_state_contribution_MPa_sqrt_m", "mean"),
        structural_wake=("structural_wake_transfer_contribution_MPa_sqrt_m", "mean"),
        delta_K=("delta_K_reinit_MPa_sqrt_m", "mean"), count=("case_id", "size")
    )
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = [f"{m}\n{t:g} deg" for m, t in zip(summary.material_class, summary.theta_deg)]
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary.required_opening, width=0.36, color="#0072B2",
           label="Required opening/local state")
    ax.bar(x + 0.18, summary.structural_wake, width=0.36, color="#D55E00",
           label="Structural wake transfer")
    ax.axhline(0, color="0.2", linewidth=0.7)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Conditional mean exact contribution [MPa sqrt(m)]")
    ax.legend(frameon=False); ax.grid(True, axis="y", color="0.9")
    ax.set_title("Reload-separated Delta K decomposition (conditional on finite reinitiation)")
    return fig, source


def mechanism_summary(initial: pd.DataFrame, stats: pd.DataFrame):
    onset = initial.groupby(["material_class", "theta_deg"], as_index=False).agg(
        mean_initial_KJ=("K0_theta_MPa_sqrt_m", "mean"),
        mean_opening_term=("opening_local_threshold_contribution_MPa_sqrt_m", "mean"),
        mean_structural_term=("structural_KJ_per_opening_contribution_MPa_sqrt_m", "mean"),
    ).merge(stats, on=["material_class", "theta_deg"])
    classes = ["Peak", "DBTT", "weak-T", "ceramic-like"]
    thetas = [0,15,30,45]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    fields = [
        ("mean_initial_KJ", "Mean initial KJ"),
        ("finite_reinitiation_fraction", "Finite reinitiation fraction"),
        ("conditional_mean_delta_K_reinit_MPa_sqrt_m", "Conditional mean Delta K reinit"),
    ]
    for ax, (field, title) in zip(axes, fields):
        matrix = onset.pivot(index="material_class", columns="theta_deg", values=field).reindex(
            index=classes, columns=thetas
        )
        im = ax.imshow(matrix, aspect="auto", cmap="cividis")
        for i in range(len(classes)):
            for j in range(len(thetas)):
                value = matrix.iloc[i, j]
                ax.text(j, i, "—" if pd.isna(value) else f"{value:.2f}", ha="center",
                        va="center", color="white" if not pd.isna(value) and value > np.nanmedian(matrix.values) else "black",
                        fontsize=8)
        ax.set_xticks(range(4), thetas); ax.set_yticks(range(4), classes)
        ax.set_xlabel("theta [deg]"); ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Canonical PF rate/orientation mechanism summary")
    return fig, onset


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args(); output = args.output
    data = pd.read_parquet(output / "pf_canonical_full_step_trajectories.parquet")
    pre = pd.read_parquet(output / "pf_peak_theta0_rate_state_history.parquet")
    rate = pd.read_csv(output / "pf_peak_theta0_rate_onset_decomposition.csv")
    initial = pd.read_csv(output / "pf_orientation_initial_onset_decomposition.csv")
    reinit = pd.read_csv(output / "pf_orientation_reinitiation_decomposition.csv")
    stats = pd.read_csv(output / "pf_orientation_conditional_reinitiation_statistics.csv")
    swap = pd.read_csv(output / "pf_orientation_frozen_swap_matrix.csv")
    records = []

    fig, src = rate_trajectory(data, False); save(fig, "PEAK_THETA0_RATE_FULL_KJ_TRAJECTORIES", src, output, records)
    fig, src = rate_trajectory(data, True); save(fig, "PEAK_THETA0_RATE_EARLY_GROWTH", src, output, records)
    fig, src = preinit_state(pre); save(fig, "PEAK_THETA0_PREINITIATION_STATE_VS_OPENING", src, output, records)
    fig, src = competition(pre); save(fig, "PEAK_THETA0_CLEAVAGE_EMISSION_COMPETITION", src, output, records)
    fig, src = rate_decomposition(rate); save(fig, "PEAK_THETA0_RATE_ONSET_DECOMPOSITION", src, output, records)
    fig, src = orientation_initial_reinit(initial, reinit, "Peak", [900,1050,1150]); save(fig, "PEAK_ORIENTATION_INITIAL_AND_REINITIATION", src, output, records)
    fig, src = orientation_initial_reinit(initial, reinit, "DBTT", [950,1000,1050,1100,1150,1200]); save(fig, "DBTT_ORIENTATION_INITIAL_AND_REINITIATION", src, output, records)
    fig, src = structural_transfer(swap); save(fig, "ORIENTATION_STRUCTURAL_TRANSFER_COEFFICIENT", src, output, records)
    fig, src = local_tensor(swap); save(fig, "ORIENTATION_LOCAL_TENSOR_AND_SOURCE_SELECTION", src, output, records)
    fig, src = reinit_decomposition(reinit); save(fig, "ORIENTATION_DELTAK_REINIT_DECOMPOSITION", src, output, records)
    fig, src = mechanism_summary(initial, stats); save(fig, "CANONICAL_RATE_ORIENTATION_MECHANISM_SUMMARY", src, output, records)

    manifest = {
        "schema": "pf_canonical_mechanism_figure_manifest_v1",
        "figure_count": len(records), "all_pdf_svg_png_600dpi": True,
        "every_figure_has_source_data": True, "records": records,
    }
    (output / "pf_canonical_mechanism_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"rendered {len(records)} mechanism figures")


if __name__ == "__main__":
    main()
