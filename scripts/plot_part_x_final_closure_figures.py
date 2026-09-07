"""Final closure E: generate the figures required by the Part X figure
manifest that are not already satisfied by an existing qualified figure.
Reads only tracked artifacts/crack_rebonding_part_x_v1/* (and, for figure
01, recomputes the ANALYTICAL periodic-orbit phase trajectory from the
tracked, frozen COMPETING_REVERSIBLE config -- a deterministic zero-
fitting analytical evaluation, never a new physical simulation).

Every figure's title/legend explicitly labels ANALYTICAL PREDICTION vs
PHYSICAL SIMULATION vs POST-PROCESSING per the provenance-labeling
requirement.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

PROTOCOL_LABEL = {
    "D1": "D1 (R=-0.95)", "D2": "D2 (COMPETING_REVERSIBLE)", "D3": "D3 (f=316Hz)",
    "D5": "D5 (PASSIVATION_LIMITED)", "D6_conditional_persistent": "D6 (persistent)",
}
PROTOCOL_COLOR = {
    "D1": "tab:blue", "D2": "tab:orange", "D3": "tab:green", "D5": "tab:red", "D6_conditional_persistent": "tab:purple",
}


def _rate_rows() -> list[dict]:
    return list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))


# --- Figure 01: P/C/B phase-resolved analytical periodic-orbit trajectory ---
def fig01_pcb_trajectory() -> str:
    from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
    from arrhenius_fracture.crack_rebonding_part_x_kinetic_regime_v10230 import analytical_periodic_orbit
    from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa
    from part_x_run_one_job import _base_rebonding_cfg, _resolve_screen_rebonding_cfg  # noqa: E402

    registry = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]
    engine, _ = build_a_native_engine()
    Eprime_Pa = reduced_modulus_Pa(engine.G, engine.nu)
    r_eff_m = float(engine.r_eff())
    finite_job = {"cohesion": "finite", "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0}
    cfg = _resolve_screen_rebonding_cfg(finite_job, registry["COMPETING_REVERSIBLE"], Eprime_Pa)

    orbit = analytical_periodic_orbit(
        cfg=cfg, T_K=300.0, Kmax_Pa_sqrt_m=18.0e6, R=-0.5, frequency_Hz=1000.0,
        minimum_load_hold_s=0.0, r_contact_m=r_eff_m, n_phase=64,
    )
    traj = orbit["trajectory"]
    phase = np.linspace(0.0, 1.0, traj.shape[0])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(phase, traj[:, 0], label="p_P (passivated)", color="tab:green")
    ax.plot(phase, traj[:, 1], label="p_C (clean)", color="tab:blue")
    ax.plot(phase, traj[:, 2], label="p_B (bonded)", color="tab:red")
    ax.set_xlabel("cycle phase (fraction of one period)")
    ax.set_ylabel("occupancy probability")
    ax.set_title("ANALYTICAL PREDICTION: P/C/B periodic-orbit phase trajectory\n"
                  "COMPETING_REVERSIBLE, Kmax=18 MPa$\\sqrt{m}$, R=-0.5, f=1000Hz (zero fitting)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig01_pcb_phase_trajectory.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 02: transition actions and realized transition fluxes ---
def fig02_transition_actions_fluxes() -> str:
    atlas = list(csv.DictReader(open(ARTIFACTS_DIR / "analytical_phase_atlas.csv")))
    sel = {r["row_name"]: r for r in atlas
           if float(r["R"]) == -0.5 and float(r["frequency_Hz"]) == 1000.0
           and float(r["minimum_load_hold_s"]) == 0.0 and float(r["chemistry_factor"]) == 1.0
           and float(r["Kmax_Pa_sqrt_m"]) == 18.0e6 and r["axis"] == "kinetic_chemistry_and_conditions"}
    rows = ["SAT_EXISTING", "COMPETING_REVERSIBLE", "COMPETING_PERSISTENT", "PASSIVATION_LIMITED"]
    channels = ["A_CB", "A_BC", "A_PC", "A_CP"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    x = np.arange(len(rows))
    width = 0.2
    for i, ch in enumerate(channels):
        vals = [float(sel[r][ch]) for r in rows if r in sel]
        axes[0].bar(x[: len(vals)] + i * width, vals, width, label=ch)
    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels(rows, rotation=20, fontsize=8)
    axes[0].set_ylabel("one-cycle transition action")
    axes[0].set_title("Transition actions (analytical)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    channels_F = ["F_CB", "F_BC", "F_PC", "F_CP"]
    for i, ch in enumerate(channels_F):
        vals = [float(sel[r][ch]) for r in rows if r in sel]
        axes[1].bar(x[: len(vals)] + i * width, vals, width, label=ch)
    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels(rows, rotation=20, fontsize=8)
    axes[1].set_ylabel("realized transition flux (1/s)")
    axes[1].set_title("Realized transition fluxes (analytical)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.suptitle("ANALYTICAL PREDICTION: transition actions/fluxes at Kmax=18 MPa$\\sqrt{m}$, R=-0.5, f=1000Hz")
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig02_transition_actions_fluxes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figures 03/04/05: screen R / S_h-vs-R / frequency response ---
def fig03_04_05_screen_R_and_frequency() -> list[str]:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px3_screen_pair_analysis.csv")))
    R_panel = sorted((float(r["R"]), r) for r in rows if r["protocol"] == "7.1_R_panel")
    freq_panel = sorted((float(r["frequency_Hz"]), r) for r in rows if r["protocol"] == "7.2_frequency_panel")

    names = []
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    Rs = [x[0] for x in R_panel]
    contact_finite = [float(x[1]["mean_negative_contact_fraction_finite"]) for x in R_panel]
    contact_zero = [float(x[1]["mean_negative_contact_fraction_zero"]) for x in R_panel]
    ax.plot(Rs, contact_finite, marker="o", label="finite cohesion")
    ax.plot(Rs, contact_zero, marker="s", linestyle="--", label="zero cohesion")
    ax.set_xlabel("R"); ax.set_ylabel("mean negative (compressive) contact fraction")
    ax.set_title("PHYSICAL SIMULATION: PX3 screen contact-exposure fraction vs R\n(Kmax=18 MPa$\\sqrt{m}$, f=1000Hz reference)", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    p = ARTIFACTS_DIR / "fig03_screen_contact_exposure_vs_R.png"
    fig.savefig(p, dpi=180); plt.close(fig); names.append(p.name)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(Rs, [float(x[1]["S_h_all"]) for x in R_panel], marker="o", color="tab:purple")
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("R"); ax.set_ylabel("S_h (screen, all events)")
    ax.set_title("PHYSICAL SIMULATION (POST-PROCESSED): PX3 screen S_h vs R\n(Kmax=18 MPa$\\sqrt{m}$, f=1000Hz reference)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = ARTIFACTS_DIR / "fig04_screen_S_h_vs_R.png"
    fig.savefig(p, dpi=180); plt.close(fig); names.append(p.name)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    freqs = [x[0] for x in freq_panel]
    ax.plot(freqs, [float(x[1]["S_h_all"]) for x in freq_panel], marker="o", color="tab:brown")
    ax.set_xscale("log")
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("S_h (screen, all events)")
    ax.set_title("PHYSICAL SIMULATION (POST-PROCESSED): PX3 screen frequency response\n(Kmax=18 MPa$\\sqrt{m}$, R=-0.5 reference)", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = ARTIFACTS_DIR / "fig05_screen_frequency_response.png"
    fig.savefig(p, dpi=180); plt.close(fig); names.append(p.name)
    return names


# --- Figure 06: dwell response, invalidated vs corrected ---
def fig06_dwell_invalidated_vs_corrected() -> str:
    dwell_audit = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())
    holds = [0.0005, 0.002]
    buggy = [dwell_audit["original_buggy_screen_S_h"]["hold=0.0005_censored_partial_unequal_window"],
             dwell_audit["original_buggy_screen_S_h"]["hold=0.002_uncensored"]]
    corrected = [dwell_audit["corrected_audit_S_h_vs_matched_zero_cohesion"]["dynamic_finite_hold0.0005"],
                 dwell_audit["corrected_audit_S_h_vs_matched_zero_cohesion"]["dynamic_finite_hold0.002"]]
    hold0_S_h = -0.043507401620722155  # PX3 screen hold=0 reference (7.3_dwell_panel, hold=0.0 row)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot([0.0] + holds, [hold0_S_h] + buggy, marker="x", linestyle="--", color="tab:red",
            label="INVALIDATED (duration-weighting bug)", markersize=10)
    ax.plot([0.0] + holds, [hold0_S_h] + corrected, marker="o", color="tab:blue",
            label="corrected dwell-causality audit")
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("minimum_load_hold_s"); ax.set_ylabel("S_h vs zero cohesion")
    ax.set_title("PHYSICAL SIMULATION: dwell response -- invalidated vs corrected\n"
                 f"classification: {dwell_audit['classification']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig06_dwell_invalidated_vs_corrected.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 07: passivation/chemistry response ---
def fig07_passivation_chemistry() -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px3_screen_pair_analysis.csv")))
    panel = sorted((float(r["chemistry_factor"]), r) for r in rows if r["protocol"] == "7.5_passivation_chemistry")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot([x[0] for x in panel], [float(x[1]["S_h_all"]) for x in panel], marker="o", color="tab:olive")
    ax.set_xlabel("chemistry_factor"); ax.set_ylabel("S_h (screen, all events)")
    ax.set_title("PHYSICAL SIMULATION (POST-PROCESSED): PX3 screen passivation/chemistry response\n(PASSIVATION_LIMITED row, Kmax=18 MPa$\\sqrt{m}$ reference)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig07_screen_passivation_chemistry_response.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 08: cohesive-strength response ---
def fig08_cohesive_strength() -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px3_screen_pair_analysis.csv")))
    panel = sorted((float(r["K_rebond_max_target_Pa_sqrt_m"]), r) for r in rows if r["protocol"] == "7.6_cohesive_strength")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot([x[0] / 1.0e6 for x in panel], [float(x[1]["S_h_all"]) for x in panel], marker="o", color="tab:cyan")
    ax.set_xlabel("K_rebond_max target (MPa$\\sqrt{m}$)"); ax.set_ylabel("S_h (screen, all events)")
    ax.set_title("PHYSICAL SIMULATION (POST-PROCESSED): PX3 screen cohesive-strength response\n(Kmax=18 MPa$\\sqrt{m}$, R=-0.5, f=1000Hz reference)", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig08_screen_cohesive_strength_response.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 09: absolute developed da/dN vs Kmax ---
def fig09_absolute_da_dN_vs_Kmax() -> str:
    rows = _rate_rows()
    from collections import defaultdict
    by_protocol = defaultdict(list)
    for r in rows:
        if r["seed"] != "1720":
            continue
        by_protocol[r["protocol"]].append((
            float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["finite_developed_da_dN_m_per_cycle"]),
            float(r["zero_developed_da_dN_m_per_cycle"]),
        ))
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for protocol, pts in by_protocol.items():
        pts.sort()
        x = [p[0] for p in pts]
        ax.plot(x, [p[1] for p in pts], marker="o", color=PROTOCOL_COLOR[protocol], label=f"{PROTOCOL_LABEL[protocol]} finite")
        ax.plot(x, [p[2] for p in pts], marker="s", linestyle="--", color=PROTOCOL_COLOR[protocol], alpha=0.5,
                label=f"{PROTOCOL_LABEL[protocol]} zero" if protocol == "D2" else None)
    ax.set_yscale("log")
    ax.set_xlabel("Kmax (MPa$\\sqrt{m}$)"); ax.set_ylabel("developed da/dN (m/cycle)")
    ax.set_title("PHYSICAL SIMULATION: absolute developed da/dN vs Kmax, seed=1720")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig09_absolute_developed_da_dN_vs_Kmax.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 11: local slopes and curvature ---
def fig11_local_slope_curvature() -> str:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_local_slope_curvature_table.csv")))
    from collections import defaultdict
    by_protocol = defaultdict(list)
    for r in rows:
        mid_K = 0.5 * (float(r["Kmax_lo_Pa_sqrt_m"]) + float(r["Kmax_hi_Pa_sqrt_m"])) / 1.0e6
        by_protocol[r["protocol"]].append((mid_K, float(r["local_secant_slope_m"])))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for protocol, pts in by_protocol.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=PROTOCOL_COLOR[protocol], label=PROTOCOL_LABEL[protocol])
    ax.set_xlabel("Kmax interval midpoint (MPa$\\sqrt{m}$)"); ax.set_ylabel("local secant slope (d S_h / d log10 Kmax)")
    ax.set_title("POST-PROCESSING: local slope of S_h_developed vs log10(Kmax), seed=1720")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig11_local_slope_curvature.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 12: reduced-frequency developed curves (D2 vs D3) ---
def fig12_reduced_frequency_developed() -> str:
    rows = _rate_rows()
    from collections import defaultdict
    by_protocol = defaultdict(list)
    for r in rows:
        if r["seed"] != "1720" or r["protocol"] not in ("D2", "D3"):
            continue
        by_protocol[r["protocol"]].append((float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["S_h_developed"])))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for protocol, pts in by_protocol.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=PROTOCOL_COLOR[protocol],
                label=f"{PROTOCOL_LABEL[protocol]} (f={'1000Hz' if protocol=='D2' else '316.228Hz'})")
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("Kmax (MPa$\\sqrt{m}$)"); ax.set_ylabel("S_h developed")
    ax.set_title("PHYSICAL SIMULATION: reduced-frequency developed curves (D2 vs D3)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig12_reduced_frequency_developed_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 13: clean reversible vs passivation-limited curves (D2 vs D5) ---
def fig13_reversible_vs_passivation() -> str:
    rows = _rate_rows()
    from collections import defaultdict
    by_protocol = defaultdict(list)
    for r in rows:
        if r["seed"] != "1720" or r["protocol"] not in ("D2", "D5"):
            continue
        by_protocol[r["protocol"]].append((float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["S_h_developed"])))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for protocol, pts in by_protocol.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=PROTOCOL_COLOR[protocol], label=PROTOCOL_LABEL[protocol])
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("Kmax (MPa$\\sqrt{m}$)"); ax.set_ylabel("S_h developed")
    ax.set_title("PHYSICAL SIMULATION: clean reversible (D2) vs passivation-limited (D5)\n"
                 "developed curves (nearly indistinguishable -- max|delta S_h|=0.0001)", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig13_reversible_vs_passivation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 14: reversible vs persistent curves (D3 vs D6) ---
def fig14_reversible_vs_persistent() -> str:
    rows = _rate_rows()
    from collections import defaultdict
    by_protocol = defaultdict(list)
    for r in rows:
        if r["seed"] != "1720" or r["protocol"] not in ("D3", "D6_conditional_persistent"):
            continue
        by_protocol[r["protocol"]].append((float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["S_h_developed"])))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for protocol, pts in by_protocol.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=PROTOCOL_COLOR[protocol], label=PROTOCOL_LABEL[protocol])
    ax.axhline(0.0, color="grey", linestyle=":")
    ax.set_xlabel("Kmax (MPa$\\sqrt{m}$)"); ax.set_ylabel("S_h developed")
    ax.set_title("PHYSICAL SIMULATION: reversible (D3) vs persistent (D6) developed curves\n"
                 "6.18x sep. at Kmax=12 (NOT an order of magnitude), decaying to 1.02x by 24.3", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig14_reversible_vs_persistent_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 17: available mechanism diagnostics vs Kmax (analytical) ---
def fig17_mechanism_diagnostics_vs_Kmax() -> str:
    atlas = list(csv.DictReader(open(ARTIFACTS_DIR / "analytical_phase_atlas.csv")))
    rows = ["SAT_EXISTING", "COMPETING_REVERSIBLE", "COMPETING_PERSISTENT", "PASSIVATION_LIMITED"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5))
    for row_name in rows:
        pts = sorted(
            (float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, float(r["mean_p_B"]), float(r["contact_time_s"]))
            for r in atlas if r["row_name"] == row_name and float(r["R"]) == -0.5 and float(r["frequency_Hz"]) == 1000.0
            and float(r["minimum_load_hold_s"]) == 0.0 and float(r["chemistry_factor"]) == 1.0
            and r["axis"] == "kinetic_chemistry_and_conditions"
        )
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=row_name)
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", label=row_name)
    axes[0].set_xlabel("Kmax (MPa$\\sqrt{m}$)"); axes[0].set_ylabel("mean_p_B (analytical periodic orbit)")
    axes[1].set_xlabel("Kmax (MPa$\\sqrt{m}$)"); axes[1].set_ylabel("contact_time_s (analytical periodic orbit)")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("ANALYTICAL PREDICTION: available mechanism diagnostics vs Kmax (PX2 atlas grid: 15/18/21 MPa$\\sqrt{m}$)")
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig17_mechanism_diagnostics_vs_Kmax.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


# --- Figure 18: censor/attempt/invalidation/admissibility map ---
def fig18_admissibility_map() -> str:
    d = json.loads((ARTIFACTS_DIR / "part_x_admissibility_map.json").read_text())
    counts = d["disposition_counts"]
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = {
        "ADMITTED": "tab:green", "NOT_LAUNCHED_BLOCKED": "tab:gray", "NOT_LAUNCHED_ALIAS": "tab:blue",
        "INTERRUPTED_NOT_SCIENCE": "tab:orange", "INVALIDATED_DWELL_DURATION_WEIGHTING_BUG": "tab:red",
        "SUPERSEDED_WALL_BUDGET_TOO_SMALL": "tab:purple",
    }
    bar_colors = [colors.get(l, "black") for l in labels]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    bars = ax.bar(labels, values, color=bar_colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, str(v), ha="center", fontsize=9)
    ax.set_ylabel("number of canonical_job_keys")
    ax.set_title(f"POST-PROCESSING: Part X censor/attempt/invalidation/admissibility map\n"
                 f"({d['n_total_rows']} total unique canonical_job_keys across PX3/PX4/PX5)")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = ARTIFACTS_DIR / "fig18_admissibility_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def main() -> None:
    generated = [
        fig01_pcb_trajectory(), fig02_transition_actions_fluxes(),
        *fig03_04_05_screen_R_and_frequency(),
        fig06_dwell_invalidated_vs_corrected(), fig07_passivation_chemistry(), fig08_cohesive_strength(),
        fig09_absolute_da_dN_vs_Kmax(), fig11_local_slope_curvature(), fig12_reduced_frequency_developed(),
        fig13_reversible_vs_passivation(), fig14_reversible_vs_persistent(),
        fig17_mechanism_diagnostics_vs_Kmax(), fig18_admissibility_map(),
    ]
    for name in generated:
        print(f"wrote {ARTIFACTS_DIR / name}")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    main()
