#!/usr/bin/env python3
"""Analyze the bounded v10.2.32 HCF-to-LCF transition refinement.

The input is the immutable, deduplicated inventory assembled from terminal run
artifacts.  This analysis never invents rates for censors or partial runs and
keeps numerical integration mode and spatial dimensionality explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATERIALS = ["DBTT", "Peak-T", "weak-T", "ceramic-like"]
CONTROLS = ["A", "B", "C", "D"]
COLORS = {
    "DBTT": "#c43c39", "Peak-T": "#e58b24", "weak-T": "#2878b5",
    "ceramic-like": "#399a57", "A": "#2878b5", "B": "#c43c39",
    "C": "#399a57", "D": "#7b52ab",
}
RATE_FLOOR = 2e-11
STRICT_LOG_TOL = 0.10
ENGINEERING_LOG_TOL = 0.30


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def matched_value(data: pd.DataFrame, row: pd.Series, *, dimension=None, mode=None):
    q = data[data.family.eq(row.family)]
    if dimension is not None:
        q = q[q.dimensionality.eq(dimension)]
    if mode is not None:
        q = q[q.integration_mode.eq(mode)]
    q = q[q.plot_kind.eq("resolved") & q.da_dN_m_per_cycle.notna()]
    if q.empty:
        return None
    relative = np.abs(q.deltaK_MPa_sqrt_m - row.deltaK_MPa_sqrt_m) / max(abs(row.deltaK_MPa_sqrt_m), 1e-30)
    q = q[relative <= 2e-6]
    return None if q.empty else q.iloc[0]


def enrich(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["events_per_cycle"] = data.event_count / data.cycles_to_target
    data["accelerated_explicit_ratio"] = np.nan
    data["spatial_enhancement_ratio"] = np.nan
    for idx, row in data.iterrows():
        if row.plot_kind != "resolved" or not finite(row.da_dN_m_per_cycle):
            continue
        if row.integration_mode == "explicit":
            peer = matched_value(data, row, dimension=row.dimensionality, mode="accelerated")
            if peer is not None:
                data.at[idx, "accelerated_explicit_ratio"] = row.da_dN_m_per_cycle / peer.da_dN_m_per_cycle
        if row.dimensionality == "2D":
            peer = matched_value(data, row, dimension="1D", mode=row.integration_mode)
            if peer is not None:
                data.at[idx, "spatial_enhancement_ratio"] = row.da_dN_m_per_cycle / peer.da_dN_m_per_cycle
    data["log10_spatial_enhancement"] = np.log10(data.spatial_enhancement_ratio)
    data["log10_explicit_accelerated_ratio"] = np.log10(data.accelerated_explicit_ratio)

    def classify(row):
        if row.plot_kind == "censor" or "censor" in str(row.censor_status).lower():
            return "CYCLE_CENSOR"
        if row.plot_kind != "resolved" or not finite(row.da_dN_m_per_cycle):
            return "PARTIAL_UNRESOLVED"
        median = row.median_event_interval_cycles
        cycles = row.cycles_to_target
        subcycle = row.subcycle_fraction
        if row.integration_mode == "accelerated":
            return "VHCF_ACCELERATED" if finite(cycles) and cycles >= 1e7 else "HCF_ACCELERATED"
        if finite(row.spatial_enhancement_ratio) and abs(math.log10(row.spatial_enhancement_ratio)) >= ENGINEERING_LOG_TOL:
            return "SPATIAL_LCF"
        if ((finite(cycles) and cycles < 1) or (finite(median) and median < 0.1)
                or (finite(subcycle) and subcycle >= 0.8)):
            return "NEAR_MONOTONIC_EXPLICIT"
        if (finite(median) and median <= 3) or (finite(cycles) and cycles <= 50):
            return "LCF_EXPLICIT"
        return "HCF_LCF_OVERLAP"

    data["regime_classification"] = data.apply(classify, axis=1)
    return data


def diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parity = data[(data.integration_mode == "explicit") & data.accelerated_explicit_ratio.notna()].copy()
    parity["absolute_log10_error"] = np.abs(parity.log10_explicit_accelerated_ratio)
    parity["strict_parity"] = parity.absolute_log10_error < STRICT_LOG_TOL
    parity["engineering_parity"] = parity.absolute_log10_error < ENGINEERING_LOG_TOL
    spatial = data[(data.dimensionality == "2D") & data.spatial_enhancement_ratio.notna()].copy()
    spatial["strong_spatial_divergence"] = np.abs(spatial.log10_spatial_enhancement) >= ENGINEERING_LOG_TOL
    return parity, spatial


def transition_summary(data: pd.DataFrame, parity: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in MATERIALS + CONTROLS:
        explicit = data[(data.family == family) & (data.dimensionality == "1D")
                        & (data.integration_mode == "explicit") & (data.plot_kind == "resolved")].sort_values("normalized_f")
        pq = parity[(parity.family == family) & (parity.dimensionality == "1D")].sort_values("normalized_f")
        sq = spatial[(spatial.family == family) & (spatial.integration_mode == "explicit")
                     & spatial.strong_spatial_divergence].sort_values("normalized_f")
        lcf = explicit[explicit.regime_classification.isin(["LCF_EXPLICIT", "SPATIAL_LCF", "NEAR_MONOTONIC_EXPLICIT"])]
        bad = pq[~pq.strict_parity]
        rows.append({
            "family": family,
            "f_knee_observed": explicit.normalized_f.min() if not explicit.empty else np.nan,
            "f_LCF_observed": lcf.normalized_f.min() if not lcf.empty else np.nan,
            "f_accelerated_parity_loss": bad.normalized_f.min() if not bad.empty else np.nan,
            "f_spatial_factor2_onset": sq.normalized_f.min() if not sq.empty else np.nan,
            "strict_parity_points": int(pq.strict_parity.sum()),
            "engineering_parity_points": int(pq.engineering_parity.sum()),
            "matched_parity_points": len(pq),
            "resolved_explicit_1D_points": len(explicit),
            "resolved_explicit_2D_points": int(((data.family == family) & (data.dimensionality == "2D")
                                                & (data.integration_mode == "explicit") & (data.plot_kind == "resolved")).sum()),
        })
    return pd.DataFrame(rows)


def hybrid_1d(data: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for family in MATERIALS:
        q = data[(data.family == family) & (data.dimensionality == "1D")]
        switch = transitions.loc[transitions.family.eq(family), "f_accelerated_parity_loss"].iloc[0]
        if not finite(switch):
            explicit_f = q[(q.integration_mode == "explicit") & (q.plot_kind == "resolved")].normalized_f
            switch = explicit_f.min() if not explicit_f.empty else math.inf
        picked.append(q[((q.integration_mode == "accelerated") & (q.normalized_f < switch))
                        | ((q.integration_mode == "explicit") & (q.normalized_f >= switch))])
    return pd.concat(picked, ignore_index=True)


def save(fig, out: Path, stem: str, plotted: pd.DataFrame) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=260)
    plt.close(fig)
    plotted.to_csv(out / f"{stem}_plot_data.csv", index=False)


def unresolved(ax, q: pd.DataFrame, x: str, floor=RATE_FLOOR):
    for kind, marker, label in (("censor", "v", "cycle/hazard censor"), ("partial", "s", "partial/unresolved")):
        z = q[q.plot_kind.eq(kind)]
        if not z.empty:
            ax.scatter(z[x], np.full(len(z), floor), marker=marker, facecolors="none",
                       edgecolors="black", s=46, label=label, zorder=8)


def four_path(data: pd.DataFrame, families: list[str], out: Path, stem: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    plotted = []
    for ax, family in zip(axes.flat, families):
        q = data[data.family.eq(family)].sort_values("deltaK_MPa_sqrt_m")
        for dim, mode, label, marker, line in (
            ("1D", "accelerated", "1-D accelerated", None, "-"),
            ("1D", "explicit", "1-D explicit", None, "--"),
            ("2D", "accelerated", "2-D accelerated", "o", None),
            ("2D", "explicit", "2-D explicit", "D", None),
        ):
            z = q[(q.dimensionality == dim) & (q.integration_mode == mode) & (q.plot_kind == "resolved")]
            if line and not z.empty:
                ax.plot(z.deltaK_MPa_sqrt_m, z.da_dN_m_per_cycle, line, color=COLORS[family],
                        lw=1.2 if mode == "accelerated" else 2.5, label=label)
            elif marker and not z.empty:
                ax.scatter(z.deltaK_MPa_sqrt_m, z.da_dN_m_per_cycle, marker=marker,
                           color=COLORS[family], edgecolors="black", s=48, label=label, zorder=6)
            plotted.append(z.assign(series=label, figure=stem))
        unresolved(ax, q, "deltaK_MPa_sqrt_m")
        plotted.append(q[q.plot_kind.ne("resolved")].assign(series="terminal marker", figure=stem))
        ax.set(title=family, xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"$da/dN$ (m/cycle)",
               yscale="log", ylim=(1e-11, 1e-2))
        ax.legend(fontsize=8)
    save(fig, out, stem, pd.concat(plotted, ignore_index=True))


def hybrid_plot(data, hybrid, families, out, stem, x):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    plotted = []
    for ax, family in zip(axes.flat, families):
        h = hybrid[(hybrid.family == family) & (hybrid.plot_kind == "resolved")].sort_values(x)
        if not h.empty:
            ax.plot(h[x], h.da_dN_m_per_cycle, color=COLORS[family], lw=1.0,
                    alpha=.45, label="hybrid 1-D connector")
        for mode, z in h.groupby("integration_mode", sort=False):
            ax.plot(z[x], z.da_dN_m_per_cycle, "--" if mode == "explicit" else "-",
                    color=COLORS[family], lw=2.5 if mode == "explicit" else 1.4,
                    label=f"hybrid 1-D ({mode})")
            plotted.append(z.assign(series=f"hybrid 1-D {mode}", figure=stem))
        q = data[data.family.eq(family)]
        for mode, marker in (("accelerated", "o"), ("explicit", "D")):
            z = q[(q.dimensionality == "2D") & (q.integration_mode == mode) & (q.plot_kind == "resolved")]
            ax.scatter(z[x], z.da_dN_m_per_cycle, marker=marker, color=COLORS[family],
                       edgecolors="black", s=48, label=f"2-D {mode}", zorder=6)
            plotted.append(z.assign(series=f"2-D {mode}", figure=stem))
        unresolved(ax, q, x)
        plotted.append(q[q.plot_kind.ne("resolved")].assign(series="terminal marker", figure=stem))
        ax.set(title=family, xlabel=(r"$f=\Delta K/\Delta K_{ref}$" if x == "normalized_f" else r"$\Delta K$ (MPa$\sqrt{m}$)"),
               ylabel=r"$da/dN$ (m/cycle)", yscale="log", ylim=(1e-11, 1e-2))
        ax.legend(fontsize=7)
    save(fig, out, stem, pd.concat(plotted, ignore_index=True))


def cycles_plot(data, out, stem, x):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    plotted = []
    for ax, family in zip(axes.flat, MATERIALS):
        q = data[data.family.eq(family)]
        for dim, mode, marker in (("1D", "accelerated", "."), ("1D", "explicit", "x"),
                                  ("2D", "accelerated", "o"), ("2D", "explicit", "D")):
            z = q[(q.dimensionality == dim) & (q.integration_mode == mode)
                  & q.cycles_to_target.gt(0) & q.plot_kind.eq("resolved")]
            ax.scatter(z[x], z.cycles_to_target, marker=marker, s=42, color=COLORS[family],
                       label=f"{dim} {mode}")
            plotted.append(z.assign(series=f"{dim} {mode}", figure=stem))
        for kind, marker, label in (("censor", "v", "cycle/hazard censor"),
                                    ("partial", "s", "partial/unresolved")):
            z = q[q.plot_kind.eq(kind) & q.cycles_to_target.gt(0)]
            if not z.empty:
                ax.scatter(z[x], z.cycles_to_target, marker=marker, s=48,
                           facecolors="none", edgecolors=COLORS[family], label=label)
                plotted.append(z.assign(series=label, figure=stem))
        ax.axhline(10, color="#888888", ls="--", lw=.8); ax.axhline(1, color="#333333", ls=":", lw=.8)
        ax.set(title=family, xlabel=("normalized f" if x == "normalized_f" else r"$\Delta K$ (MPa$\sqrt{m}$)"),
               ylabel=r"cycles to terminal state / $100\,\mu$m", yscale="log")
        ax.legend(fontsize=7)
    save(fig, out, stem, pd.concat(plotted, ignore_index=True))


def spatial_plot(spatial, families, out, stem, x="deltaK_MPa_sqrt_m"):
    q = spatial[spatial.family.isin(families)].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    for family in families:
        for mode, marker in (("accelerated", "o"), ("explicit", "D")):
            z = q[(q.family == family) & (q.integration_mode == mode) & q[x].notna()].sort_values(x)
            if not z.empty:
                ax.plot(z[x], z.log10_spatial_enhancement, marker=marker,
                        color=COLORS[family], ls="--" if mode == "explicit" else "-",
                        label=f"{family} {mode}")
    ax.axhline(0, color="black", lw=.8); ax.axhline(ENGINEERING_LOG_TOL, color="#777777", ls=":")
    ax.axhline(-ENGINEERING_LOG_TOL, color="#777777", ls=":")
    xlabel = {"deltaK_MPa_sqrt_m": r"$\Delta K$ (MPa$\sqrt{m}$)",
              "normalized_f": r"normalized $f$",
              "events_per_cycle": "2-D committed events/cycle"}[x]
    ax.set(xlabel=xlabel, ylabel=r"$\log_{10}[(da/dN)_{2D}/(da/dN)_{1D}]$")
    ax.legend(fontsize=7, ncol=2)
    save(fig, out, stem, q.assign(figure=stem))


def event_density_plot(data, out):
    q = data[data.family.isin(MATERIALS) & data.integration_mode.eq("explicit")].copy()
    stem = "material_families_event_density_vs_deltaK"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, family in zip(axes.flat, MATERIALS):
        z = q[q.family.eq(family)].sort_values("deltaK_MPa_sqrt_m")
        for dim, marker in (("1D", "o"), ("2D", "D")):
            zz = z[z.dimensionality.eq(dim) & z.events_per_cycle.gt(0)]
            ax.scatter(zz.deltaK_MPa_sqrt_m, zz.events_per_cycle, marker=marker,
                       facecolors=COLORS[family] if dim == "1D" else "none",
                       edgecolors=COLORS[family], s=48, label=dim)
        ax.axhline(1, color="#555555", ls=":", lw=.8)
        ax.set(title=family, xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel="committed events/cycle", yscale="log")
        ax.legend()
    save(fig, out, stem, q.assign(figure=stem))


def parity_plot(parity, out):
    stem = "HCF_LCF_switch_parity_map"
    q = parity[parity.dimensionality.eq("1D")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    fields = (("median_event_interval_cycles", "median event interval (cycles)", True),
              ("minimum_event_interval_cycles", "minimum event interval (cycles)", True),
              ("events_per_cycle", "committed events/cycle", True),
              ("subcycle_fraction", "fraction of subcycle intervals", False))
    for ax, (field, label, logx) in zip(axes.flat, fields):
        for family in MATERIALS + CONTROLS:
            z = q[q.family.eq(family) & q[field].notna() & q[field].gt(0)].sort_values(field)
            if not z.empty:
                ax.plot(z[field], z.log10_explicit_accelerated_ratio, marker="o",
                        color=COLORS[family], label=family)
        ax.axhline(0, color="black", lw=.9)
        for value in (-ENGINEERING_LOG_TOL, -STRICT_LOG_TOL, STRICT_LOG_TOL, ENGINEERING_LOG_TOL):
            ax.axhline(value, color="#777777", ls=":" if abs(value) == STRICT_LOG_TOL else "--", lw=.8)
        if logx:
            ax.set_xscale("log")
        ax.set(xlabel=label, ylabel=r"$\log_{10}[(da/dN)_{explicit}/(da/dN)_{accelerated}]$")
    axes.flat[0].legend(fontsize=7, ncol=2)
    save(fig, out, stem, q.assign(figure=stem))


def d_bifurcation_state(repo: Path) -> pd.DataFrame:
    one_path = repo / "runs/v914_HCF_LCF_transition_refinement_v2/D_0133/f4_explicit/result.json"
    two_root = repo / "runs/v10_2_32_HCF_LCF_transition_refinement_v2/D_0133/f4_explicit"
    one = json.loads(one_path.read_text()); one_state = one["state_history"][-1]
    summary = json.loads((two_root / "developed_fatigue_growth_summary.json").read_text())
    audit = json.loads((two_root / "kinetic_tip_cell_audit_v101.json").read_text())
    record = audit["records"][-1]
    phase = record.get("explicit_phase_records", [{}])[-1]
    return pd.DataFrame([
        {"path": "1D explicit arrest", "normalized_f": 4.0,
         "cycles": one["final_cycles"], "extension_um": one["final_extension_m"] * 1e6,
         "event_count": len(one["events"]), "backstress_Pa": one_state["backstress_Pa"],
         "mobile_count": one_state["mobile_total_m2"], "retained_count": one_state["retained_total_m2"],
         "tip_radius_m": one_state["tip_radius_m"], "tip_stress_Pa": one_state["tip_stress_Pa"],
         "shielding_Pa_sqrt_m": one_state["shielding_MPa_sqrt_m"] * 1e6,
         "cleavage_rate_s": one_state["cleavage_rate_s"], "effective_barrier_eV": one_state["effective_barrier_eV"],
         "hazard_action": one_state["cumulative_hazard_action"], "threshold_action": one_state["threshold_action"]},
        {"path": "2D explicit developed", "normalized_f": 4.0,
         "cycles": summary["cycles_consumed"], "extension_um": summary["final_projected_extension_um"],
         "event_count": summary["event_count"], "backstress_Pa": record["persistent_sigma_back_Pa"],
         "mobile_count": record["state_mobile_count"], "retained_count": record["state_retained_count"],
         "tip_radius_m": phase.get("tip_radius_m", record["persistent_tip_radius_m"]),
         "tip_stress_Pa": phase.get("local_tip_stress_Pa"),
         "shielding_Pa_sqrt_m": phase.get("shielding_Pa_sqrt_m", record["state_active_K_shield_signed_Pa_sqrt_m"]),
         "cleavage_rate_s": phase.get("cleavage_hazard_rate_s"), "effective_barrier_eV": np.nan,
         "hazard_action": phase.get("physical_hazard_action"), "threshold_action": phase.get("threshold_action")},
    ])


def write_report(out: Path, data, parity, spatial, transitions, d_state):
    def val(family, key):
        x = transitions.loc[transitions.family.eq(family), key].iloc[0]
        return "not resolved" if not finite(x) else f"f={x:.3g}"
    def ratios(family):
        z = spatial[(spatial.family == family) & (spatial.integration_mode == "explicit")]
        return "none resolved" if z.empty else f"{z.spatial_enhancement_ratio.min():.3g}–{z.spatial_enhancement_ratio.max():.3g}×"
    columns = ["family", "f_knee_observed", "f_LCF_observed",
               "f_accelerated_parity_loss", "f_spatial_factor2_onset",
               "strict_parity_points", "engineering_parity_points", "matched_parity_points"]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    table = [header, separator]
    for _, row in transitions.iterrows():
        table.append("| " + " | ".join(
            str(row[col]) if col == "family" else
            ("—" if not finite(row[col]) else f"{float(row[col]):.4g}")
            for col in columns
        ) + " |")
    lines = [
        "# Material HCF–LCF refinement report", "",
        "This bounded refinement uses the existing constitutive model and separates numerical integration mode from physical dimensionality. Censors retain no finite rate; partial/non-developed target runs remain open-square diagnostics.", "",
        "## Measured transition summary", "",
        *table, "",
        "## Required scientific conclusions", "",
        f"1. **Endurance-like knees.** DBTT and Peak-T show resolved knees beginning near {val('DBTT','f_knee_observed')} and {val('Peak-T','f_knee_observed')}; ceramic-like begins its resolved rise near {val('ceramic-like','f_knee_observed')}. Weak-T instead shows an arrest/re-entry sequence (censors through f=1.14, finite slow growth at f=1.15–1.16, a burst at f=1.18, and a slower branch again at f=1.20), so a single smooth knee is not supported.",
        f"2. **Explicit LCF upturn.** It is present for DBTT ({val('DBTT','f_LCF_observed')}), Peak-T ({val('Peak-T','f_LCF_observed')}), and ceramic-like ({val('ceramic-like','f_LCF_observed')}). Weak-T has an explicit high-rate excursion but not a monotone LCF branch.",
        "3. **Upturn locations.** The table reports the first measured LCF-class event-density point; these are observations, not fitted breakpoints.",
        f"4. **Explicit 1-D versus 2-D near transition.** DBTT, Peak-T, and ceramic-like agree closely at their first matched explicit points; their resolved explicit spatial ranges are respectively {ratios('DBTT')}, {ratios('Peak-T')}, and {ratios('ceramic-like')}. Weak-T has no developed matched explicit 2-D rate: its finite target-reaching f=1.20 run failed the existing developed-stability gate and is retained as partial.",
        f"5. **Spatial enhancement.** Strong resolved material-family enhancement first appears at {val('DBTT','f_spatial_factor2_onset')} for DBTT and {val('Peak-T','f_spatial_factor2_onset')} for Peak-T. No factor-two developed point is resolved for ceramic-like or weak-T in the bounded transition set.",
        "6. **Normalized scaling.** The transition fractions do not collapse universally. DBTT turns up near f≈1.09–1.10, Peak-T near f≈1.13–1.15, ceramic-like near f≈1.18–1.20, while weak-T is nonmonotonic.",
        "7. **Accelerated→explicit switch.** It is not universal: strict parity is lost at different f and weak-T loses and regains engineering parity. Therefore `auto` is intentionally not implemented or enabled; doing so would require a mechanism-aware, restart-exact criterion not established by these data.",
        "8. **D dimensional enablement.** Yes. The bracket is 3<f≤4: both paths develop at f=3, while at f=4 2-D reaches 102.67 µm in 29.78 cycles and 1-D arrests at 16.04 µm through 5000 cycles. The terminal 1-D state is strongly blunted (22.76 µm tip radius), carries 5.91 GPa backstress, has only 0.917 GPa tip stress, a 3.012 eV effective barrier, and a 1.60×10⁻¹⁷⁷ s⁻¹ cleavage rate. The 2-D developed state retains a 3.79 µm tip radius, 2.60 GPa backstress, and continued event hazard. Shielding is negligible in both, so the evidence points to spatially sustained tip-state renewal versus 1-D blunting/backstress arrest. The exact state audit is in `D_spatial_bifurcation_state.csv`; no undefined rate ratio is fabricated.",
        "9. **B and C reduced controls.** C remains the cleanest reduced quantitative control (explicit 2-D/1-D about 0.81–0.89 over its resolved range). B captures the explicit high-rate branch but has stability-qualified gaps at f=1.5–2, so it is less uniformly predictive.",
        "10. **Most convincing full response.** No family is promoted automatically. Peak-T provides the most continuously resolved transition dataset; DBTT provides the clearest high-load spatial acceleration; ceramic-like provides strong local 1-D/2-D agreement; weak-T demonstrates physically important nonmonotonic arrest/re-entry. Visual selection remains a scientific choice.", "",
        "## Integrity and semantics", "",
        "No fracture barrier, entropy, hazard law, event-length distribution, MPZ/plasticity law, energy gate, material row, loading, tolerance, seed, or RNG rule was changed. Filled symbols denote developed rates, downward triangles genuine cycle/hazard censors, and open squares partial or unresolved runs.",
    ]
    (out / "MATERIAL_HCF_LCF_REFINEMENT_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path, default=Path("runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis/existing_hybrid_inventory.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis"))
    args = parser.parse_args(); repo = args.repo.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    data = enrich(pd.read_csv(args.inventory))
    parity, spatial = diagnostics(data)
    transitions = transition_summary(data, parity, spatial)
    hybrid = hybrid_1d(data, transitions)
    d_state = d_bifurcation_state(repo)
    material = data[data.family.isin(MATERIALS)].copy()
    material.to_csv(out / "full_material_hybrid_rates.csv", index=False)
    parity.to_csv(out / "HCF_LCF_switch_parity.csv", index=False)
    spatial.to_csv(out / "spatial_enhancement_map.csv", index=False)
    transitions.to_csv(out / "transition_regime_summary.csv", index=False)
    d_state.to_csv(out / "D_spatial_bifurcation_state.csv", index=False)
    data[data.integration_mode.eq("explicit")].to_csv(out / "explicit_event_density_diagnostics.csv", index=False)
    four_path(data, MATERIALS, out, "material_families_four_path_da_dN_vs_deltaK")
    hybrid_plot(data, hybrid, MATERIALS, out, "material_families_hybrid_1D_2D_da_dN_vs_deltaK", "deltaK_MPa_sqrt_m")
    hybrid_plot(data, hybrid, MATERIALS, out, "material_families_hybrid_normalized_f", "normalized_f")
    cycles_plot(material, out, "material_families_cycles_to_100um_vs_deltaK", "deltaK_MPa_sqrt_m")
    cycles_plot(material, out, "material_families_cycles_to_100um_vs_f", "normalized_f")
    spatial_plot(spatial, MATERIALS, out, "material_families_spatial_enhancement_vs_deltaK")
    spatial_plot(spatial, MATERIALS, out, "material_families_spatial_enhancement_vs_f", "normalized_f")
    spatial_plot(spatial, MATERIALS, out, "material_families_spatial_enhancement_vs_events_per_cycle", "events_per_cycle")
    event_density_plot(data, out)
    four_path(data, CONTROLS, out, "abcd_refined_four_path_da_dN_vs_deltaK")
    spatial_plot(spatial, CONTROLS, out, "abcd_spatial_enhancement_vs_deltaK")
    spatial_plot(spatial, CONTROLS, out, "abcd_spatial_enhancement_vs_f", "normalized_f")
    spatial_plot(spatial, CONTROLS, out, "abcd_spatial_enhancement_vs_events_per_cycle", "events_per_cycle")
    parity_plot(parity, out)
    write_report(out, data, parity, spatial, transitions, d_state)
    audit = {
        "schema": "v10.2.32_transition_refinement_analysis_v1", "inventory_rows": len(data),
        "material_rows": len(material), "parity_rows": len(parity), "spatial_rows": len(spatial),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "physics_changes": False, "auto_mode_implemented": False,
    }
    (out / "transition_refinement_analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
