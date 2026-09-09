"""Generate the 5 focused da/dN-vs-K figure groups from
all_included_curve_points.csv / global_slope_summary.csv /
local_secant_slopes.csv. No physical simulation; pure plotting.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_focused import R_MARKER, R_LABEL, OPTION_COLOR, PROTOCOL_COLOR, BASELINE_STYLE, savefig_all

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "SOURCE_DATA"
HASHES = {}


def loglog(ax, x_is_narrow_range=True, x_fixed_ticks=None):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="all", numticks=20))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="all", numticks=20))
    if x_is_narrow_range:
        # Every Kmax/DeltaK axis in this package spans less than ~1 decade; the default
        # LogLocator produces overlapping scientific-notation ticks there, so plain numbers
        # are used on the x-axis instead. The da/dN y-axis (multiple decades) keeps log ticks.
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if x_fixed_ticks:
            ax.xaxis.set_major_locator(mticker.FixedLocator(x_fixed_ticks))


def load_points():
    with (SRC / "all_included_curve_points.csv").open() as fh:
        return list(csv.DictReader(fh))


def load_globals():
    with (SRC / "global_slope_summary.csv").open() as fh:
        return list(csv.DictReader(fh))


def load_locals():
    with (SRC / "local_secant_slopes.csv").open() as fh:
        return list(csv.DictReader(fh))


def _series(rows, key_fn, xk="Kmax_MPa_sqrt_m"):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for k, grp in groups.items():
        grp = sorted(grp, key=lambda r: float(r[xk]))
        x = [float(r[xk]) for r in grp]
        y = [float(r["da_dN_m_per_cycle"]) for r in grp]
        out[k] = (x, y)
    return out


# ===========================================================================
# Figure 1: A_NATIVE R-dependence
# ===========================================================================
def figure1(points):
    out_dir = PKG / "01_R_DEPENDENCE"
    a_native = [r for r in points if r["internal_parameterization_id"] == "A_NATIVE"]
    by_R_kmax = _series(a_native, lambda r: float(r["R"]))
    by_R_dk = _series(a_native, lambda r: float(r["R"]), xk="Kmax_MPa_sqrt_m")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
    for R in [-0.95, 0.1, 0.5]:
        x, y = by_R_kmax[R]
        axA.plot(x, y, marker=R_MARKER[R], color="black", markerfacecolor=["none", "0.5", "black"][
            [-0.95, 0.1, 0.5].index(R)], linestyle="-", label=R_LABEL[R])
    loglog(axA, x_fixed_ticks=[12,15,18,24])
    axA.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)")
    axA.set_ylabel(r"Developed $da/dN$ (m/cycle)")
    axA.set_title("A. da/dN vs. $K_{max}$")
    axA.legend()

    rows_by_R = defaultdict(list)
    for r in a_native:
        rows_by_R[float(r["R"])].append(r)
    for R in [-0.95, 0.1, 0.5]:
        grp = sorted(rows_by_R[R], key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        dk = [float(r["DeltaK_applied_MPa_sqrt_m"]) for r in grp]
        y = [float(r["da_dN_m_per_cycle"]) for r in grp]
        idx = [-0.95, 0.1, 0.5].index(R)
        axB.plot(dk, y, marker=R_MARKER[R], color="black",
                 markerfacecolor=["none", "0.5", "black"][idx], linestyle="-", label=R_LABEL[R])
    loglog(axB, x_fixed_ticks=[6, 10, 20, 30, 47])
    axB.set_xlabel(r"Applied $\Delta K=(1-R)K_{max}$ (MPa$\sqrt{m}$)")
    axB.set_ylabel(r"Developed $da/dN$ (m/cycle)")
    axB.set_title("B. da/dN vs. applied full-range $\\Delta K$")
    axB.legend()
    fig.suptitle("Figure 1. A_NATIVE developed da/dN(K) vs. load ratio R "
                 "(single seed 1720; markers=calculated points, lines=visual connector)", fontsize=13)
    fig.tight_layout()
    HASHES["Figure1_A_NATIVE_R_dependence"] = savefig_all(fig, "Figure1_A_NATIVE_R_dependence",
                                                           out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig)
    print("Figure 1 written. R=-0.5 comparison: UNAVAILABLE -- see plot_selection_manifest.csv "
          "(Part X zero-cohesion baseline at R=-0.5 could not be proven to share the same "
          "physical-fingerprint/provenance scheme as the A_NATIVE R-ratio study; not combined).")


# ===========================================================================
# Figure 2: A_NATIVE / PT03 / PT08
# ===========================================================================
def figure2(points):
    out_dir = PKG / "02_TRANSPORT_PARAMETERIZATIONS"
    options = ["A_NATIVE", "PT03", "PT08"]
    label_to_pid = {"A_NATIVE": "A_NATIVE",
                    "PT03": "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5",
                    "PT08": "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    curve_data = {}
    for ax, R in zip(axes, [-0.95, 0.1, 0.5]):
        for opt in options:
            pid = label_to_pid[opt]
            rows = sorted([r for r in points if r["internal_parameterization_id"] == pid
                          and float(r["R"]) == R], key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
            x = [float(r["Kmax_MPa_sqrt_m"]) for r in rows]
            y = [float(r["da_dN_m_per_cycle"]) for r in rows]
            curve_data[(opt, R)] = (x, y)
            ax.plot(x, y, marker="o", color=OPTION_COLOR[opt], label=opt)
        loglog(ax, x_fixed_ticks=[12,15,18,24])
        ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)")
        ax.set_title(f"R={R:g}")
    axes[0].set_ylabel(r"Developed $da/dN$ (m/cycle)")
    axes[0].legend()
    fig.suptitle("Figure 2. A_NATIVE / PT03 / PT08 transport-mechanism substitutions "
                 "(NOT distinct calibrated materials -- Peierls/Taylor sensitivity test)", fontsize=13)
    fig.tight_layout()
    HASHES["Figure2_A_NATIVE_PT03_PT08"] = savefig_all(fig, "Figure2_A_NATIVE_PT03_PT08",
                                                        out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig)

    # Compact ratio figure
    fig2, (axr1, axr2) = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    for R in [-0.95, 0.1, 0.5]:
        xA, yA = curve_data[("A_NATIVE", R)]
        for ax, opt in [(axr1, "PT03"), (axr2, "PT08")]:
            x, y = curve_data[(opt, R)]
            ratio = [yy / ya for yy, ya in zip(y, yA)]
            ax.plot(x, ratio, marker="o", label=R_LABEL[R])
    for ax, opt in [(axr1, "PT03"), (axr2, "PT08")]:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mticker.FixedLocator([12, 15, 18, 24]))
        ax.axhline(1.0, color="0.6", linewidth=1.0, linestyle=":")
        ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)")
        ax.set_title(f"(da/dN)$_{{{opt}}}$ / (da/dN)$_{{A\\_NATIVE}}$")
    axr1.set_ylabel("Rate ratio")
    axr1.legend(fontsize=11)
    fig2.suptitle("Figure 2 (ratio companion). PT/A_NATIVE developed-rate ratio -- "
                 "differences are a few percent at most", fontsize=12)
    fig2.tight_layout()
    HASHES["Figure2_ratio_companion"] = savefig_all(fig2, "Figure2_ratio_companion",
                                                     out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig2)
    print("Figure 2 written (+ ratio companion).")


# ===========================================================================
# Figure 3: crack-rebonding parameterizations
# ===========================================================================
def figure3(points):
    out_dir = PKG / "03_REBONDING_PARAMETERIZATIONS"

    def curve(pid, cat, seed="1720"):
        rows = sorted([r for r in points if r["internal_parameterization_id"].startswith(pid + "/")
                      and r["parameterization_category"] == cat and r["seed"] == seed],
                      key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        x = [float(r["Kmax_MPa_sqrt_m"]) for r in rows]
        y = [float(r["da_dN_m_per_cycle"]) for r in rows]
        return x, y

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)

    # Panel A: D1 (R=-0.95) + D2 (R=-0.5) + matched zero-cohesion baselines
    ax = axes[0]
    for pid, label, color in [("D1", "D1 COMPETING_REVERSIBLE R=-0.95", PROTOCOL_COLOR["D1"]),
                               ("D2", "D2 COMPETING_REVERSIBLE R=-0.50", PROTOCOL_COLOR["D2"])]:
        x, y = curve(pid, "rebonding dynamic")
        ax.plot(x, y, marker="o", color=color, label=label)
        xb, yb = curve(pid, "zero-cohesion control")
        ax.plot(xb, yb, label=f"{pid} zero-cohesion baseline" if pid == "D1" else None, **BASELINE_STYLE)
    loglog(ax, x_fixed_ticks=[12,15,18,21,24.3]); ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)"); ax.set_title("A. R dependence, 1000 Hz")
    ax.legend(fontsize=10)

    # Panel B: baseline + D2 (clean reversible) + D5 (passivation-limited), R=-0.5, 1000Hz
    ax = axes[1]
    xb, yb = curve("D2", "zero-cohesion control")
    ax.plot(xb, yb, label="Zero-cohesion baseline", **BASELINE_STYLE)
    for pid, label in [("D2", "D2 clean reversible"), ("D5", "D5 passivation-limited")]:
        x, y = curve(pid, "rebonding dynamic")
        ax.plot(x, y, marker="o", color=PROTOCOL_COLOR[pid], label=label)
    loglog(ax, x_fixed_ticks=[12,15,18,21,24.3]); ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)"); ax.set_title("B. Passivation, R=-0.50, 1000 Hz")
    ax.legend(fontsize=10)

    # Panel C: baseline + D3 (reversible) + D6 (persistent), R=-0.5, 316.227766 Hz
    ax = axes[2]
    xb, yb = curve("D3", "zero-cohesion control")
    ax.plot(xb, yb, label="Zero-cohesion baseline", **BASELINE_STYLE)
    for pid, label in [("D3", "D3 reversible"), ("D6_conditional_persistent", "D6 persistent")]:
        x, y = curve(pid, "rebonding dynamic")
        ax.plot(x, y, marker="o", color=PROTOCOL_COLOR[pid.split("_")[0] if pid != "D6_conditional_persistent" else "D6_conditional_persistent"], label=label)
    loglog(ax, x_fixed_ticks=[12,15,18,21,24.3]); ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)"); ax.set_title("C. Bond survival, R=-0.50, 316.23 Hz")
    ax.legend(fontsize=10)
    axes[0].set_ylabel(r"Developed $da/dN$ (m/cycle)")

    fig.suptitle("Figure 3. Crack-rebonding parameterizations (PX4 developed points, signed-K contact "
                 "surrogate -- opposing crack-face contact is not resolved)", fontsize=12)
    fig.tight_layout()
    HASHES["Figure3_rebonding_parameterizations"] = savefig_all(
        fig, "Figure3_rebonding_parameterizations", out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig)

    # S_h(K) companion
    import math
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    panel_defs = [
        (axes2[0], [("D1", PROTOCOL_COLOR["D1"], "D1 R=-0.95"), ("D2", PROTOCOL_COLOR["D2"], "D2 R=-0.50")], "A. R dependence"),
        (axes2[1], [("D2", PROTOCOL_COLOR["D2"], "D2 clean"), ("D5", PROTOCOL_COLOR["D5"], "D5 passivation")], "B. Passivation"),
        (axes2[2], [("D3", PROTOCOL_COLOR["D3"], "D3 reversible"),
                    ("D6_conditional_persistent", PROTOCOL_COLOR["D6_conditional_persistent"], "D6 persistent")], "C. Bond survival"),
    ]
    for ax, series, title in panel_defs:
        for pid, color, label in series:
            xf, yf = curve(pid, "rebonding dynamic")
            xb, yb = curve(pid, "zero-cohesion control")
            Sh = [math.log10(f / b) for f, b in zip(yf, yb)]
            ax.plot(xf, Sh, marker="o", color=color, label=label)
        ax.axhline(0.0, color="0.6", linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_locator(mticker.FixedLocator([12, 15, 18, 21, 24.3]))
        ax.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)")
        ax.set_title(title)
        ax.legend(fontsize=10)
    axes2[0].set_ylabel(r"$S_h = \log_{10}[(da/dN)_{finite}/(da/dN)_{zero}]$")
    fig2.suptitle("Figure 3 (companion). Shielding metric $S_h(K)$ -- signed-K contact surrogate, "
                 "opposing crack-face contact not resolved", fontsize=12)
    fig2.tight_layout()
    HASHES["Figure3_Sh_companion"] = savefig_all(fig2, "Figure3_Sh_companion",
                                                  out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig2)
    print("Figure 3 written (+ S_h companion). Invalidated dwell-duration-weighting trajectories excluded.")


# ===========================================================================
# Figure 4: local + global slopes
# ===========================================================================
def figure4(points, globals_, locals_):
    out_dir = PKG / "04_LOCAL_SLOPES"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
    a_native_pids = {-0.95: "A_NATIVE", 0.1: "A_NATIVE", 0.5: "A_NATIVE"}
    for R in [-0.95, 0.1, 0.5]:
        rows = [r for r in locals_ if r["internal_parameterization_id"] == "A_NATIVE"]
        # local secants are per-(pid,cat,R,seed,freq) group but internal_parameterization_id alone
        # is shared across R for A_NATIVE -- filter by re-deriving from global_slope_summary key match
    # Simpler: recompute directly from points here for plotting clarity (uses same formula as compute_slopes.py)
    import math
    def secants_for(pid, R=None, cat="baseline"):
        rows = [r for r in points if r["internal_parameterization_id"] == pid
                and (R is None or float(r["R"]) == R)]
        rows = sorted(rows, key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        K = [float(r["Kmax_MPa_sqrt_m"]) for r in rows]
        G = [float(r["da_dN_m_per_cycle"]) for r in rows]
        out = []
        for i in range(len(K) - 1):
            Kmid = math.sqrt(K[i] * K[i + 1])
            m = (math.log10(G[i + 1]) - math.log10(G[i])) / (math.log10(K[i + 1]) - math.log10(K[i]))
            out.append((Kmid, m))
        return out

    for R in [-0.95, 0.1, 0.5]:
        sec = secants_for("A_NATIVE", R)
        if sec:
            xk, ym = zip(*sec)
            axA.plot(xk, ym, marker=R_MARKER[R], color="black",
                     markerfacecolor=["none", "0.5", "black"][[-0.95, 0.1, 0.5].index(R)],
                     label=R_LABEL[R])
    axA.set_xscale("log")
    axA.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axA.xaxis.set_minor_formatter(mticker.NullFormatter())
    axA.xaxis.set_major_locator(mticker.FixedLocator([13, 14, 16, 18, 21]))
    axA.set_xlabel(r"$K_{mid}=\sqrt{K_iK_{i+1}}$ (MPa$\sqrt{m}$)")
    axA.set_ylabel("Local secant slope $m_{local}$")
    axA.set_title("A. A_NATIVE local slope vs. R")
    axA.legend()

    def pid_for(label):
        rows = [r for r in points if r["visible_curve_label"] == label]
        return rows[0]["internal_parameterization_id"] if rows else None

    rebond_series = [
        ("D1", "rebonding dynamic", "D1 (R=-0.95)", PROTOCOL_COLOR["D1"]),
        ("D2", "rebonding dynamic", "D2 (R=-0.50)", PROTOCOL_COLOR["D2"]),
        ("D3", "rebonding dynamic", "D3 (316 Hz)", PROTOCOL_COLOR["D3"]),
        ("D2", "zero-cohesion control", "zero-cohesion baseline", "0.5"),
    ]
    for pid_base, cat, label, color in rebond_series:
        rows = [r for r in points if r["internal_parameterization_id"].startswith(pid_base + "/")
                and r["parameterization_category"] == cat and r["seed"] == "1720"]
        rows = sorted(rows, key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        K = [float(r["Kmax_MPa_sqrt_m"]) for r in rows]
        G = [float(r["da_dN_m_per_cycle"]) for r in rows]
        sec = []
        for i in range(len(K) - 1):
            Kmid = math.sqrt(K[i] * K[i + 1])
            m = (math.log10(G[i + 1]) - math.log10(G[i])) / (math.log10(K[i + 1]) - math.log10(K[i]))
            sec.append((Kmid, m))
        if sec:
            xk, ym = zip(*sec)
            ls = "--" if "baseline" in label else "-"
            marker = "o" if "baseline" not in label else "s"
            axB.plot(xk, ym, marker=marker, linestyle=ls, color=color, label=label)
    axB.set_xscale("log")
    axB.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axB.xaxis.set_minor_formatter(mticker.NullFormatter())
    axB.xaxis.set_major_locator(mticker.FixedLocator([13, 14, 16, 18, 21]))
    axB.set_xlabel(r"$K_{mid}$ (MPa$\sqrt{m}$)")
    axB.set_title("B. Rebonding-family local slope")
    axB.legend(fontsize=10)
    fig.suptitle("Figure 4. Local (adjacent-secant) slopes -- primary local-slope measure", fontsize=13)
    fig.tight_layout()
    HASHES["Figure4_local_slopes"] = savefig_all(fig, "Figure4_local_slopes",
                                                  out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig)

    # Global slope summary bar chart
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    globals_sorted = sorted(globals_, key=lambda r: float(r["global_slope_m"]))
    labels = [g["visible_curve_label"] for g in globals_sorted]
    vals = [float(g["global_slope_m"]) for g in globals_sorted]
    colors = ["tab:red" if g["classification"] == "CURVED_NON_PARIS_RESPONSE" else "tab:blue"
              for g in globals_sorted]
    y_pos = range(len(labels))
    ax2.barh(y_pos, vals, color=colors)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Global log-log fitted slope $m_{global}$")
    ax2.set_title("Global slope summary (red = CURVED_NON_PARIS_RESPONSE; blue = PARIS_ADEQUATE)",
                  fontsize=11)
    fig2.tight_layout()
    HASHES["Figure4_global_slope_summary"] = savefig_all(fig2, "Figure4_global_slope_summary",
                                                          out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig2)
    print("Figure 4 written (local slopes + global summary bar chart).")


# ===========================================================================
# Figure 5: material-class context
# ===========================================================================
def figure5(points):
    out_dir = PKG / "05_MATERIAL_CLASS_CONTEXT"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    label_to_pid = {"A_NATIVE": "A_NATIVE",
                    "PT03": "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5",
                    "PT08": "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"}
    for opt in ["A_NATIVE", "PT03", "PT08"]:
        pid = label_to_pid[opt]
        rows = sorted([r for r in points if r["internal_parameterization_id"] == pid
                      and float(r["R"]) == 0.1], key=lambda r: float(r["Kmax_MPa_sqrt_m"]))
        x = [float(r["Kmax_MPa_sqrt_m"]) for r in rows]
        y = [float(r["da_dN_m_per_cycle"]) for r in rows]
        axA.plot(x, y, marker="o", color=OPTION_COLOR[opt], label=f"{opt} (R=0.1)")
    loglog(axA, x_fixed_ticks=[12,15,18,24])
    for m, x0 in [(2, 12), (4, 12), (8, 12)]:
        xg = [12, 24]
        y0 = 3e-8
        yg = [y0, y0 * (24 / 12) ** m]
        axA.plot(xg, yg, color="0.75", linewidth=0.8, linestyle=":")
        axA.text(24, yg[-1], f"m={m}", fontsize=8, color="0.5", va="center")
    axA.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$)")
    axA.set_ylabel(r"Developed $da/dN$ (m/cycle)")
    axA.set_title("A. Moderate-slope responses\n(curved, not a fixed Paris exponent)", fontsize=12)
    axA.legend(fontsize=9)
    axA.text(0.98, 0.02, "physical-slope-transfer target-M≈4 row: UNAVAILABLE\n"
             "(no committed/portable per-point table located)",
             transform=axA.transAxes, fontsize=7, color="0.4", va="bottom", ha="right")

    # Panel B: fracture-selected activation-cliff controls, reference lines only (no per-point data available)
    controls = [("Peak", 100.5, "tab:red"), ("DBTT", 133.5, "tab:purple"), ("ceramic-like", 71.7, "tab:green")]
    x_ref = [15, 18]
    for name, m, color in controls:
        y0 = 1.0
        y_ref = [y0, y0 * (x_ref[1] / x_ref[0]) ** m]
        axB.plot(x_ref, y_ref, color=color, linewidth=2.2, label=f"{name}: m≈{m:g} (FRACTURE-SELECTED\nACTIVATION-CLIFF CONTROL)")
    loglog(axB, x_fixed_ticks=[15, 16, 17, 18])
    axB.set_xlabel(r"$K_{max}$ (MPa$\sqrt{m}$, arbitrary offset)")
    axB.set_title("B. Steep fracture-selected controls\n(slope guides -- arbitrary vertical offset)")
    axB.legend(fontsize=8)
    axB.text(0.02, 0.02, "weak-T: no qualified stable global fit -- no line drawn", transform=axB.transAxes,
             fontsize=7, color="0.4", va="bottom")

    fig.suptitle("Figure 5. Material-class context. These are heuristic comparison bands / project-\n"
                 "specific control values, not fitted targets and not universal material-class constants. "
                 "Material class cannot be inferred from Paris slope alone.", fontsize=10.5)
    fig.tight_layout()
    HASHES["Figure5_material_class_context"] = savefig_all(fig, "Figure5_material_class_context",
                                                            out_dir / "PDF", out_dir / "SVG", out_dir / "PNG_600DPI")
    plt.close(fig)
    print("Figure 5 written.")


def main():
    points = load_points()
    globals_ = load_globals()
    locals_ = load_locals()
    figure1(points)
    figure2(points)
    figure3(points)
    figure4(points, globals_, locals_)
    figure5(points)

    import json
    (PKG / "SOURCE_DATA" / "figure_hashes.json").write_text(json.dumps(HASHES, indent=2, default=str))
    print(f"\nAll figures written. {len(HASHES)} figure files hashed.")


if __name__ == "__main__":
    main()
