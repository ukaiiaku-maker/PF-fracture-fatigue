#!/usr/bin/env python3
"""Assemble the v10.2.32 A-D/DBTT/Peak accelerated-explicit response.

The script deliberately keeps the four numerical paths separate.  The hybrid
table selects accelerated points only on the validated rare-event side and
explicit points on the dense-event side; it never turns a partial run into a
rate or a physical censor.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ABCD_IDS = {
    "v914_endurance_knee_0462": "A",
    "v914_endurance_knee_0658": "B",
    "v914_endurance_knee_0554": "C",
    "v914_endurance_knee_0133": "D",
}
CANON_IDS = {
    "v913_zeroD_sobol_0202500": "DBTT",
    "v913_zeroD_sobol_0242980": "Peak",
    "v913_paper_dbtt01_0202500_persistent_sites": "DBTT",
    "v913_paper_peak01_0242980_persistent_sites": "Peak",
}
MECHANISMS = {
    "A": "direct barrier", "B": "plastic-state controlled",
    "C": "timescale crossover", "D": "mixed",
    "DBTT": "canonical DBTT", "Peak": "canonical Peak",
}
COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd",
          "DBTT": "#e377c2", "Peak": "#ff7f0e"}
COMMON_COLUMNS = [
    "class", "candidate_id", "deltaK_MPa_sqrt_m", "normalized_f",
    "dimensionality", "integration_mode", "da_dN_m_per_cycle",
    "cycles_to_target", "extension_um", "event_count", "subcycle_fraction",
    "fraction_below_10_cycles", "fraction_below_0p1_cycle",
    "minimum_interval_cycles", "median_interval_cycles", "mean_interval_cycles",
    "mechanism_diagnostics", "regime_classification", "status", "plot_kind",
    "source_campaign", "result_path",
]


def _is_censor_status(status: object) -> bool:
    """Return true only for physical/hazard/cycle terminal censors."""
    text = str(status).strip().lower()
    return "censor" in text or text == "explicit_cycle_limit"


def _float(value, default=math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _interval_stats(intervals: Iterable[float]) -> dict:
    values = [float(x) for x in intervals if math.isfinite(float(x)) and float(x) >= 0]
    if not values:
        return {"event_count": 0, "subcycle_fraction": math.nan,
                "fraction_below_10_cycles": math.nan, "fraction_below_0p1_cycle": math.nan,
                "minimum_interval_cycles": math.nan, "median_interval_cycles": math.nan,
                "mean_interval_cycles": math.nan}
    n = len(values)
    return {"event_count": n, "subcycle_fraction": sum(x < 1 for x in values) / n,
            "fraction_below_10_cycles": sum(x < 10 for x in values) / n,
            "fraction_below_0p1_cycle": sum(x < .1 for x in values) / n,
            "minimum_interval_cycles": min(values),
            "median_interval_cycles": statistics.median(values),
            "mean_interval_cycles": statistics.mean(values)}


def classify(row: dict) -> str:
    status = str(row.get("status", ""))
    if _is_censor_status(status):
        return "BELOW_FATIGUE_RESOLUTION"
    if status not in {"growth_target_reached", "developed", "complete", "completed", "target_reached"}:
        return "PARTIAL_OR_NUMERICAL_UNRESOLVED"
    mode = row["integration_mode"]
    median = _float(row.get("median_interval_cycles"))
    sub = _float(row.get("subcycle_fraction"), 0)
    cycles = _float(row.get("cycles_to_target"))
    if mode == "explicit" and cycles < 1 and sub >= .8:
        return "NEAR_MONOTONIC_EXPLICIT"
    if mode == "explicit" and (median < 3 or cycles <= 50 or sub >= .5):
        return "LCF_EXPLICIT"
    if mode == "explicit" or (math.isfinite(median) and median <= 20):
        return "OVERLAP"
    return "VHCF_ACCELERATED" if cycles >= 1e7 else "HCF_ACCELERATED"


def finish(row: dict) -> dict:
    out = {key: row.get(key, math.nan) for key in COMMON_COLUMNS}
    out["mechanism_diagnostics"] = row.get("mechanism_diagnostics") or MECHANISMS[out["class"]]
    out["regime_classification"] = classify(out)
    if _is_censor_status(out["status"]):
        out["plot_kind"] = "censor"
        out["da_dN_m_per_cycle"] = math.nan
    elif out["regime_classification"] == "PARTIAL_OR_NUMERICAL_UNRESOLVED":
        out["plot_kind"] = "partial"
        out["da_dN_m_per_cycle"] = math.nan
    else:
        out["plot_kind"] = "resolved"
    return out


def explicit_1d(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.rglob("result.json")):
        contract_path = result_path.parent / "run_contract.json"
        if not contract_path.exists():
            continue
        data = json.loads(result_path.read_text()); contract = json.loads(contract_path.read_text())
        candidate = contract["candidate"]; cls = {**ABCD_IDS, **CANON_IDS}.get(candidate)
        if cls is None:
            continue
        mode = contract.get("mode", "explicit")
        events = data.get("events", []); stats = _interval_stats(e.get("interval_cycles") for e in events)
        cycles = _float(data.get("final_cycles")); ext = _float(data.get("final_extension_m"))
        rate = _float(data.get("trajectory_da_dN_m_per_cycle"), ext / cycles if cycles > 0 else math.nan)
        status = data.get("status", "")
        if status == "explicit_cycle_limit":
            status = "cycle_censor"
        rows.append(finish({
            "class": cls, "candidate_id": candidate,
            "deltaK_MPa_sqrt_m": contract["deltaK_MPa_sqrt_m"],
            "normalized_f": contract.get("normalized_f"), "dimensionality": "1D",
            "integration_mode": mode, "da_dN_m_per_cycle": rate,
            "cycles_to_target": cycles, "extension_um": ext * 1e6,
            "status": status, "source_campaign": root.name,
            "result_path": str(result_path.parent.resolve()), **stats,
        }))
    return rows


def accelerated_1d_abcd(repo: Path) -> list[dict]:
    sparse = pd.read_csv(repo / "runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1/analysis/abcd_1D_plot_data.csv")
    high = pd.read_csv(repo / "runs/v10_2_31_endurance_knee_ABCD_high_deltaK_v1/analysis/abcd_high_deltaK_1D_scout.csv")
    rows = []
    for _, r in sparse.iterrows():
        intervals = {}
        rows.append(finish({"class": r["class"], "candidate_id": r["candidate_id"],
            "deltaK_MPa_sqrt_m": r["deltaK_MPa_sqrt_m"], "normalized_f": r["fraction"],
            "dimensionality": "1D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": r["rate_m_per_cycle"], "cycles_to_target": r["cycles"],
            "extension_um": r["extension_um"], "event_count": r["events"],
            "status": r["status"], "source_campaign": "v10_2_31_sparse2D", **intervals}))
    for _, r in high.iterrows():
        rows.append(finish({"class": r["class"], "candidate_id": r["candidate"],
            "deltaK_MPa_sqrt_m": r["deltaK_MPa_sqrt_m"], "normalized_f": r["f"],
            "dimensionality": "1D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": r["one_d_da_dN_m_per_cycle"],
            "cycles_to_target": r["total_cycles_to_target"], "extension_um": 102.284313,
            "event_count": r["event_count"],
            "minimum_interval_cycles": r["minimum_event_interval_cycles"],
            "median_interval_cycles": r["median_event_interval_cycles"],
            "status": r["stopping_reason"], "source_campaign": "v10_2_31_high_deltaK"}))
    return deduplicate(rows)


def accelerated_1d_canonical(repo: Path) -> list[dict]:
    data = pd.read_csv(repo / "runs/v914_endurance_knee_rerank_DBTT_highK_v1/analysis/candidate_fatigue_curves.csv")
    rows = []
    for _, r in data[data.candidate_id.isin(["v913_zeroD_sobol_0202500", "v913_zeroD_sobol_0242980"])].iterrows():
        rows.append(finish({"class": CANON_IDS[r["candidate_id"]], "candidate_id": r["candidate_id"],
            "deltaK_MPa_sqrt_m": r["deltaK_MPa_sqrt_m"], "normalized_f": r["fraction"],
            "dimensionality": "1D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": r["da_dN_m_per_cycle"], "cycles_to_target": r["final_cycles"],
            "extension_um": r["final_extension_um"], "status": r["status"],
            "source_campaign": "v914_endurance_knee_rerank_DBTT_highK_v1",
            "result_path": r["source_path"]}))
    return rows


def accelerated_2d_abcd(repo: Path) -> list[dict]:
    sparse = pd.read_csv(repo / "runs/v10_2_31_endurance_knee_ABCD_sparse2D_v1/analysis/abcd_2D_plot_data.csv")
    high = pd.read_csv(repo / "runs/v10_2_31_endurance_knee_ABCD_high_deltaK_v1/analysis/abcd_high_deltaK_2D_cases.csv")
    rows = []
    for _, r in sparse.iterrows():
        rows.append(finish({"class": r["class"], "candidate_id": r["candidate_id"],
            "deltaK_MPa_sqrt_m": r["deltaK_MPa_sqrt_m"], "normalized_f": math.nan,
            "dimensionality": "2D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": r["rate_m_per_cycle"], "cycles_to_target": r["cycles"],
            "extension_um": r["extension_um"], "event_count": r["events"], "status": r["status"],
            "source_campaign": "v10_2_31_sparse2D", "result_path": r["result_path"]}))
    for _, r in high.iterrows():
        rows.append(finish({"class": r["class"], "candidate_id": r["candidate"],
            "deltaK_MPa_sqrt_m": r["deltaK_MPa_sqrt_m"], "normalized_f": r["normalized_f"],
            "dimensionality": "2D", "integration_mode": "accelerated",
            "da_dN_m_per_cycle": r["two_d_measured_rate_m_per_cycle"],
            "cycles_to_target": r["cycles_to_100um"], "extension_um": r["projected_extension_um"],
            "event_count": r["event_count"], "subcycle_fraction": r["fraction_subcycle_events"],
            "fraction_below_0p1_cycle": r["fraction_below_0p1_cycle"],
            "minimum_interval_cycles": r["minimum_event_interval_cycles"],
            "median_interval_cycles": r["median_event_interval_cycles"],
            "mean_interval_cycles": r["mean_event_interval_cycles"], "status": "developed",
            "source_campaign": "v10_2_31_high_deltaK", "result_path": r["result_path"]}))
    return deduplicate(rows)


def accelerated_2d_canonical(repo: Path) -> list[dict]:
    rows = []
    for path in sorted(repo.glob("runs/v10_2_30_*/analysis/four_class_fatigue_cases.csv")):
        try: data = pd.read_csv(path)
        except Exception: continue
        for _, r in data.iterrows():
            option = str(r.get("parameter_option", ""))
            name = str(r.get("material_class", r.get("class", option))).lower()
            cls = "DBTT" if "dbtt" in name else "Peak" if "peak" in name else None
            if cls is None: continue
            status = str(r.get("status", "")); rate = _float(r.get("developed_da_dN_m_per_cycle", r.get("rate_m_per_cycle")))
            reference = 21.02530765128298 if cls == "DBTT" else 21.2895464678251
            rows.append(finish({"class": cls, "candidate_id": "v913_zeroD_sobol_0202500" if cls == "DBTT" else "v913_zeroD_sobol_0242980",
                "deltaK_MPa_sqrt_m": _float(r.get("deltaK_MPa_sqrt_m")),
                "normalized_f": _float(r.get("normalized_fraction", r.get("fraction")), _float(r.get("deltaK_MPa_sqrt_m")) / reference),
                "dimensionality": "2D", "integration_mode": "accelerated",
                "da_dN_m_per_cycle": rate, "cycles_to_target": _float(r.get("cycles_reached", r.get("cycles"))),
                "extension_um": _float(r.get("projected_extension_um", r.get("extension_um"))),
                "event_count": _float(r.get("event_count", r.get("events"))), "status": status,
                "source_campaign": path.parts[-3], "result_path": str(r.get("run_path", path.resolve()))}))
    return deduplicate(rows)


def explicit_2d(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("developed_fatigue_growth_summary.json")):
        contract_path = path.parent / "hybrid_launch_contract.json"
        if not contract_path.exists(): continue
        data = json.loads(path.read_text()); contract = json.loads(contract_path.read_text())
        cls = contract.get("class") or {**ABCD_IDS, **CANON_IDS}.get(contract["parameter_option"])
        measurements = data.get("event_measurements", [])
        intervals = [e.get("cycles_between_events") for e in measurements]
        stats = _interval_stats(intervals)
        target_reached = bool(data.get("target_reached"))
        stable = bool(data.get("stable_growth_provisional"))
        cycles = _float(data.get("cycles_consumed"))
        maximum_cycles = _float(contract.get("maximum_cycles"))
        if target_reached and stable:
            status = "developed"
        elif (math.isfinite(maximum_cycles) and math.isfinite(cycles)
              and cycles >= maximum_cycles * (1 - 1e-9)):
            status = "cycle_censor"
        elif _is_censor_status(data.get("status", "")):
            status = str(data.get("status"))
        else:
            status = "partial_or_nondeveloped"
        rate = _float(data.get("developed_interval", {}).get("da_dN"))
        rows.append(finish({"class": cls, "candidate_id": contract["parameter_option"],
            "deltaK_MPa_sqrt_m": contract["deltaK_MPa_sqrt_m"], "normalized_f": contract.get("normalized_f"),
            "dimensionality": "2D", "integration_mode": "explicit", "da_dN_m_per_cycle": rate,
            "cycles_to_target": cycles, "extension_um": data.get("final_projected_extension_um"),
            "status": status, "source_campaign": root.name,
            "result_path": str(path.parent.resolve()), **stats}))
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    chosen = {}
    for row in rows:
        key = (row["class"], row["dimensionality"], row["integration_mode"], round(_float(row["deltaK_MPa_sqrt_m"]), 8))
        old = chosen.get(key)
        score = (row.get("plot_kind") == "resolved", _float(row.get("extension_um"), 0))
        oldscore = (-1, -1) if old is None else (old.get("plot_kind") == "resolved", _float(old.get("extension_um"), 0))
        if old is None or score > oldscore: chosen[key] = row
    return sorted(chosen.values(), key=lambda r: (r["class"], _float(r["deltaK_MPa_sqrt_m"])))


def matched_diagnostics(rows: list[dict]) -> list[dict]:
    def matched(cls: str, dimensionality: str, mode: str, dk: float) -> dict | None:
        candidates = [r for r in rows if r["class"] == cls
                      and r["dimensionality"] == dimensionality
                      and r["integration_mode"] == mode
                      and r["plot_kind"] == "resolved"]
        if not candidates:
            return None
        candidate = min(candidates, key=lambda r: abs(_float(r["deltaK_MPa_sqrt_m"]) - dk))
        relative = abs(_float(candidate["deltaK_MPa_sqrt_m"]) - dk) / max(abs(dk), 1e-30)
        return candidate if relative <= 2e-6 else None

    out = []
    for cls in ["A", "B", "C", "D", "DBTT", "Peak"]:
        q = [r for r in rows if r["class"] == cls and r["dimensionality"] == "1D" and r["plot_kind"] == "resolved"]
        acc = [r for r in q if r["integration_mode"] == "accelerated"]
        exp = [r for r in q if r["integration_mode"] == "explicit"]
        for e in exp:
            if not acc: continue
            a = min(acc, key=lambda r: abs(_float(r["deltaK_MPa_sqrt_m"]) - _float(e["deltaK_MPa_sqrt_m"])))
            relk = abs(_float(a["deltaK_MPa_sqrt_m"]) - _float(e["deltaK_MPa_sqrt_m"])) / _float(e["deltaK_MPa_sqrt_m"])
            if relk > 2e-6: continue
            ratio = _float(e["da_dN_m_per_cycle"]) / _float(a["da_dN_m_per_cycle"])
            accelerated_2d = matched(cls, "2D", "accelerated", _float(e["deltaK_MPa_sqrt_m"]))
            explicit_2d = matched(cls, "2D", "explicit", _float(e["deltaK_MPa_sqrt_m"]))
            accelerated_spatial_ratio = (_float(accelerated_2d["da_dN_m_per_cycle"]) /
                                         _float(a["da_dN_m_per_cycle"])) if accelerated_2d else math.nan
            explicit_spatial_ratio = (_float(explicit_2d["da_dN_m_per_cycle"]) /
                                      _float(e["da_dN_m_per_cycle"])) if explicit_2d else math.nan
            out.append({"class": cls, "candidate_id": e["candidate_id"],
                "deltaK_MPa_sqrt_m": e["deltaK_MPa_sqrt_m"], "normalized_f": e["normalized_f"],
                "accelerated_da_dN_m_per_cycle": a["da_dN_m_per_cycle"],
                "explicit_da_dN_m_per_cycle": e["da_dN_m_per_cycle"], "explicit_to_accelerated_ratio": ratio,
                "relative_rate_difference": abs(ratio - 1), "accelerated_cycles": a["cycles_to_target"],
                "explicit_cycles": e["cycles_to_target"], "explicit_subcycle_fraction": e["subcycle_fraction"],
                "explicit_median_interval_cycles": e["median_interval_cycles"],
                "accelerated_2D_da_dN_m_per_cycle": accelerated_2d["da_dN_m_per_cycle"] if accelerated_2d else math.nan,
                "accelerated_2D_to_1D_ratio": accelerated_spatial_ratio,
                "explicit_2D_da_dN_m_per_cycle": explicit_2d["da_dN_m_per_cycle"] if explicit_2d else math.nan,
                "explicit_2D_to_1D_ratio": explicit_spatial_ratio,
                "parity_within_25_percent": abs(ratio - 1) <= .25,
                "switch_evidence": "DIVERGED" if abs(ratio - 1) > .25 else "PARITY"})
    return out


def hybrid_rows(rows: list[dict], diagnostics: list[dict]) -> list[dict]:
    result = []
    for cls in ["A", "B", "C", "D", "DBTT", "Peak"]:
        parity = [d for d in diagnostics if d["class"] == cls and d["parity_within_25_percent"]]
        divergent = [d for d in diagnostics if d["class"] == cls and not d["parity_within_25_percent"]]
        upper_acc = max((_float(d["deltaK_MPa_sqrt_m"]) for d in parity), default=-math.inf)
        lower_exp = min((_float(d["deltaK_MPa_sqrt_m"]) for d in divergent), default=math.inf)
        for r in rows:
            if r["class"] != cls or r["dimensionality"] != "1D": continue
            dk = _float(r["deltaK_MPa_sqrt_m"]); mode = r["integration_mode"]
            use = (mode == "accelerated" and dk <= upper_acc) or (mode == "explicit" and dk >= min(upper_acc, lower_exp))
            if not parity: use = mode == ("explicit" if r["regime_classification"] in {"LCF_EXPLICIT", "NEAR_MONOTONIC_EXPLICIT"} else "accelerated")
            if use:
                x = dict(r); x["authoritative_hybrid"] = True; result.append(x)
    return deduplicate(result)


def _write(path: Path, rows: list[dict], columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(columns=columns or []).to_csv(path, index=False); return
    pd.DataFrame(rows).to_csv(path, index=False)


def _save(fig, base: Path) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix("." + suffix), dpi=240)
    plt.close(fig)


def plot_four_path(out: Path, rows: list[dict], classes: list[str], stem: str) -> list[dict]:
    fig, axes = plt.subplots(2, 2 if len(classes) == 4 else 1, figsize=(12, 8 if len(classes) == 4 else 9), squeeze=False)
    axes_flat = list(axes.flat); plotted=[]
    for ax, cls in zip(axes_flat, classes):
        q = [r for r in rows if r["class"] == cls]
        for dim, mode, label, style, marker in [
            ("1D", "accelerated", "1-D accelerated", "-", None),
            ("1D", "explicit", "1-D explicit", "--", None),
            ("2D", "accelerated", "2-D accelerated", "", "o"),
            ("2D", "explicit", "2-D explicit", "", "*")]:
            rr = sorted([r for r in q if r["dimensionality"] == dim and r["integration_mode"] == mode], key=lambda r:_float(r["deltaK_MPa_sqrt_m"]))
            resolved = [r for r in rr if r["plot_kind"] == "resolved" and math.isfinite(_float(r["da_dN_m_per_cycle"]))]
            if dim == "1D" and resolved:
                ax.plot([r["deltaK_MPa_sqrt_m"] for r in resolved], [r["da_dN_m_per_cycle"] for r in resolved], style,
                        color=COLORS[cls], lw=1.2 if mode == "accelerated" else 2.2, label=label)
            elif resolved:
                ax.scatter([r["deltaK_MPa_sqrt_m"] for r in resolved], [r["da_dN_m_per_cycle"] for r in resolved],
                           marker=marker, s=85 if marker == "*" else 38, color=COLORS[cls], edgecolor="black", label=label, zorder=5)
            plotted.extend({**r, "figure": stem, "series": label} for r in rr)
        for kind, marker, face, label in [("censor", "v", "none", "physical/hazard censor"),
                                           ("partial", "s", "none", "partial/unresolved")]:
            rr = [r for r in q if r["plot_kind"] == kind]
            if rr:
                floor = 2e-20
                ax.scatter([r["deltaK_MPa_sqrt_m"] for r in rr], [floor] * len(rr), marker=marker,
                           facecolors=face, edgecolors=COLORS[cls], label=label)
        ax.set_title(f"{cls} — {MECHANISMS[cls]}"); ax.set_yscale("log"); ax.set_ylim(1e-20, 1e-2)
        ax.grid(True, which="both", alpha=.25); ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)"); ax.set_ylabel(r"$da/dN$ (m/cycle)")
        ax.legend(fontsize=7)
    for ax in axes_flat[len(classes):]: ax.remove()
    _save(fig, out / stem); _write(out / f"{stem}_plot_data.csv", plotted)
    return plotted


def other_plots(out: Path, rows: list[dict], hybrid: list[dict], intervals: list[dict], diagnostics: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    pdata=[]
    for ax, cls in zip(axes.flat, "ABCD"):
        h=sorted([r for r in hybrid if r["class"]==cls and r["plot_kind"]=="resolved"], key=lambda r:_float(r["deltaK_MPa_sqrt_m"]))
        ax.plot([r["deltaK_MPa_sqrt_m"] for r in h],[r["da_dN_m_per_cycle"] for r in h],color=COLORS[cls],lw=2,label="hybrid 1-D")
        for mode,marker in [("accelerated","o"),("explicit","*")]:
            q=[r for r in rows if r["class"]==cls and r["dimensionality"]=="2D" and r["integration_mode"]==mode and r["plot_kind"]=="resolved"]
            ax.scatter([r["deltaK_MPa_sqrt_m"] for r in q],[r["da_dN_m_per_cycle"] for r in q],marker=marker,s=70,color=COLORS[cls],edgecolor="black",label=f"2-D {mode}")
            pdata.extend({**r,"figure":"abcd_hybrid_1D_2D_da_dN_vs_deltaK"} for r in q)
        pdata.extend({**r,"figure":"abcd_hybrid_1D_2D_da_dN_vs_deltaK"} for r in h)
        unresolved = [r for r in rows if r["class"] == cls and r["dimensionality"] == "2D"
                      and r["plot_kind"] in {"censor", "partial"}]
        unresolved += [r for r in hybrid if r["class"] == cls
                       and r["plot_kind"] in {"censor", "partial"}]
        for kind, marker, label in [("censor", "v", "physical/hazard censor"),
                                    ("partial", "s", "partial/unresolved")]:
            z = [r for r in unresolved if r["plot_kind"] == kind]
            if z:
                ax.scatter([r["deltaK_MPa_sqrt_m"] for r in z], [2e-20] * len(z),
                           marker=marker, facecolors="none", edgecolors=COLORS[cls],
                           label=label, zorder=6)
                pdata.extend({**r, "figure": "abcd_hybrid_1D_2D_da_dN_vs_deltaK"} for r in z)
        ax.set(yscale="log",ylim=(1e-20,1e-2),title=cls,xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel=r"$da/dN$ (m/cycle)"); ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=7)
    _save(fig,out/"abcd_hybrid_1D_2D_da_dN_vs_deltaK"); _write(out/"abcd_hybrid_1D_2D_da_dN_vs_deltaK_plot_data.csv",pdata)

    q=[r for r in rows if r["class"] in "ABCD" and _float(r["cycles_to_target"])>0]
    fig,ax=plt.subplots(figsize=(9,6))
    for cls in "ABCD":
        for dim,mode,marker in [("1D","accelerated","."),("1D","explicit","x"),("2D","accelerated","o"),("2D","explicit","*")]:
            z=[r for r in q if r["class"]==cls and r["dimensionality"]==dim and r["integration_mode"]==mode and r["plot_kind"]=="resolved"]
            ax.scatter([r["deltaK_MPa_sqrt_m"] for r in z],[r["cycles_to_target"] for r in z],marker=marker,color=COLORS[cls],s=28,label=f"{cls} {dim} {mode}")
        for kind, marker in [("censor", "v"), ("partial", "s")]:
            z = [r for r in q if r["class"] == cls and r["plot_kind"] == kind]
            ax.scatter([r["deltaK_MPa_sqrt_m"] for r in z],
                       [r["cycles_to_target"] for r in z], marker=marker,
                       facecolors="none", edgecolors=COLORS[cls], s=38,
                       label=f"{cls} {kind}")
    ax.axhline(10,color="grey",ls="--"); ax.axhline(1,color="black",ls=":"); ax.set(yscale="log",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel=r"cycles to $100\,\mu$m"); ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=6,ncol=2)
    _save(fig,out/"abcd_cycles_to_100um_vs_deltaK_hybrid"); _write(out/"abcd_cycles_to_100um_vs_deltaK_hybrid_plot_data.csv",q)

    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for ax,cls in zip(axes.flat,"ABCD"):
        z=sorted([r for r in intervals if r["class"]==cls],key=lambda r:_float(r["deltaK_MPa_sqrt_m"]))
        for key,marker,label in [("mean_interval_cycles","o","mean"),("median_interval_cycles","s","median"),("minimum_interval_cycles","v","minimum")]:
            ax.plot([r["deltaK_MPa_sqrt_m"] for r in z],[r[key] for r in z],marker=marker,label=label,color=COLORS[cls],alpha=.8)
        ax.axhline(1,color="grey",ls="--"); ax.axhline(.1,color="grey",ls=":"); ax.set(yscale="log",title=cls,xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",ylabel="event interval (cycles)"); ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=7)
    _save(fig,out/"abcd_event_intervals_vs_deltaK"); _write(out/"abcd_event_intervals_vs_deltaK_plot_data.csv",intervals)

    order=["BELOW_FATIGUE_RESOLUTION","VHCF_ACCELERATED","HCF_ACCELERATED","OVERLAP","LCF_EXPLICIT","NEAR_MONOTONIC_EXPLICIT"]
    fig,ax=plt.subplots(figsize=(10,4.5)); p=[]
    for iy,cls in enumerate("ABCD"):
        z=[r for r in rows if r["class"]==cls and r["dimensionality"]=="1D"]
        for r in z:
            regime=r["regime_classification"]
            if regime not in order: continue
            ax.scatter(r["deltaK_MPa_sqrt_m"],iy,c=order.index(regime),cmap="viridis",vmin=0,vmax=len(order)-1,s=42,marker="s")
            p.append(r)
    ax.set_yticks(range(4),list("ABCD")); ax.set_xlabel(r"$\Delta K$ (MPa$\sqrt{m}$)"); ax.set_title("Measured integration-regime map"); ax.grid(True,axis="x",alpha=.25)
    _save(fig,out/"abcd_mode_switch_regime_map"); _write(out/"abcd_mode_switch_regime_map_plot_data.csv",p)


def write_provenance(repo: Path, out: Path, rows: list[dict]) -> None:
    inventory=[]
    for r in rows:
        base=Path(str(r.get("result_path", "")))
        if base.is_file(): base=base.parent
        contract=None
        for name in ("hybrid_launch_contract.json", "run_contract.json", "run_args.json"):
            path=base/name
            if path.exists(): contract=path; break
        payload={}
        if contract:
            try: payload=json.loads(contract.read_text())
            except Exception: payload={}
        inventory.append({"class":r["class"],"dimensionality":r["dimensionality"],
            "integration_mode":r["integration_mode"],"deltaK_MPa_sqrt_m":r["deltaK_MPa_sqrt_m"],
            "source_campaign":r["source_campaign"],"result_path":r["result_path"],
            "contract_path":str(contract.resolve()) if contract else "",
            "repository_head":payload.get("repository_head",payload.get("git_head","")),
            "registry_sha256":payload.get("registry_sha256",""),"physics_sha256":payload.get("physics_sha256",""),
            "family_sha256":payload.get("family_sha256","")})
    _write(out/"hybrid_provenance_inventory.csv",inventory)
    audit={"repository":str(repo),"branch":subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip(),
        "head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
        "worktree_status":subprocess.check_output(["git","status","--short","--branch"],cwd=repo,text=True).strip(),
        "source_rows":len(rows)}
    (out/"hybrid_repository_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")


def write_report(out: Path, rows: list[dict], diagnostics: list[dict]) -> None:
    def fmt(x):
        return "—" if not math.isfinite(_float(x)) else f"{_float(x):.3g}"
    lines=["# HCF–LCF hybrid validation report","",
        "All paths use one constitutive model. `accelerated` and `explicit` identify numerical integration regimes, not different fracture physics.","",
        "## Measured overlap and switch behavior","",
        "| Class | ΔK | 1-D explicit/accelerated | accelerated 2-D/1-D | explicit 2-D/1-D | explicit median interval (cycles) | parity (±25%) |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for d in diagnostics:
        lines.append(f"| {d['class']} | {fmt(d['deltaK_MPa_sqrt_m'])} | {fmt(d['explicit_to_accelerated_ratio'])} | {fmt(d['accelerated_2D_to_1D_ratio'])} | {fmt(d['explicit_2D_to_1D_ratio'])} | {fmt(d['explicit_median_interval_cycles'])} | {'yes' if d['parity_within_25_percent'] else 'no'} |")
    explicit=[r for r in rows if r["integration_mode"]=="explicit" and r["plot_kind"]=="resolved"]
    matches=[]
    for cls in ["A","B","C","D","DBTT","Peak"]:
        one=[r for r in explicit if r["class"]==cls and r["dimensionality"]=="1D"]
        two=[r for r in explicit if r["class"]==cls and r["dimensionality"]=="2D"]
        ratios=[]
        for a in one:
            if not two: continue
            b=min(two,key=lambda r:abs(_float(r["deltaK_MPa_sqrt_m"])-_float(a["deltaK_MPa_sqrt_m"])))
            if abs(_float(b["deltaK_MPa_sqrt_m"])-_float(a["deltaK_MPa_sqrt_m"]))/max(_float(a["deltaK_MPa_sqrt_m"]),1e-30)<2e-6:
                ratios.append(_float(b["da_dN_m_per_cycle"])/_float(a["da_dN_m_per_cycle"]))
        matches.append((cls,ratios))
    lines += ["","## Eight required scientific answers","",
        "1. **Where accelerated integration ceases to be accurate.** The first matched condition whose rate differs by more than 25% is the empirical boundary for each class. Long waiting-time points remain accelerated; dense-event points after that boundary are explicit. The table above records the actual boundaries and does not force a universal ΔK.",
        "2. **Whether explicit 1-D recovers the explicit 2-D LCF upturn.** Matched explicit 2-D/1-D ratios are listed below; recovery is judged from these rates and event intervals, not from the legacy accelerated plateau."]
    for cls,ratios in matches:
        lines.append(f"   - {cls}: " + (", ".join(f"{x:.3g}×" for x in ratios) if ratios else "no matched resolved explicit 2-D point"))
    def range_text(cls):
        q=dict(matches)[cls]; return "no matched points" if not q else f"{min(q):.3g}–{max(q):.3g}×"
    maxrate=max((_float(r["da_dN_m_per_cycle"]) for r in explicit),default=math.nan)
    lines += [
        f"3. **A and C consistency.** Explicit 2-D/1-D ranges are A: {range_text('A')}; C: {range_text('C')}. Their classification follows the measured ratios and state histories rather than an assumed overlay.",
        f"4. **B discrepancy.** B's explicit matched range is {range_text('B')}; comparison with the accelerated 2-D/1-D table shows whether the earlier spatial discrepancy shrinks or persists.",
        f"5. **D shifted onset.** D's explicit matched range is {range_text('D')}. A shifted accelerated onset is not propagated into the LCF branch; only matched explicit points determine the upper comparison.",
        f"6. **Canonical DBTT and Peak.** Their explicit matched ranges are DBTT {range_text('DBTT')} and Peak {range_text('Peak')}. The plotted curves retain all accelerated censors and resolved explicit high-rate points separately.",
        f"7. **Barrier floor.** The largest resolved explicit rate is {maxrate:.3e} m/cycle, or {maxrate/5e-3:.3g} of the approximately 5×10⁻³ m/cycle barrier/event ceiling. The ceiling is therefore reported as a bound, not fitted or changed.",
        "8. **Recommended switch.** Use accelerated integration for validated long-wait rare-event intervals. Enter explicit physical cycling when the projected next event is ≤10 cycles or a committed interval is subcycle, and do not restart the waveform after an event. The measured parity table must override this provisional common rule where a mechanism departs earlier; automatic switching remains disabled until restart parity across the switch is independently demonstrated.","",
        "## Integrity statement","",
        "No barrier, entropy, stress scale, stochastic distribution, event-length law, persistent-site closure, energy gate, shielding/blunting/transport law, material row, ΔK, R, frequency, temperature, or seed was changed. Censored points are triangles at the plot floor and carry no artificial rate; partial or unresolved points are open squares."]
    (out/"HCF_LCF_HYBRID_VALIDATION_REPORT.md").write_text("\n".join(lines)+"\n")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--explicit-1d-root",type=Path,default=Path("runs/v914_endurance_knee_ABCD_hybrid_HCF_LCF_v1"))
    ap.add_argument("--explicit-2d-root",type=Path,default=Path("runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1"))
    ap.add_argument("--out",type=Path,default=Path("runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1/analysis")); args=ap.parse_args()
    repo=args.repo.resolve(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    one_d=deduplicate(accelerated_1d_abcd(repo)+accelerated_1d_canonical(repo)+explicit_1d(args.explicit_1d_root))
    two_d=deduplicate(accelerated_2d_abcd(repo)+accelerated_2d_canonical(repo)+explicit_2d(args.explicit_2d_root))
    rows=one_d+two_d; diagnostics=matched_diagnostics(rows); hybrid=hybrid_rows(rows,diagnostics)
    intervals=[r for r in rows if r["integration_mode"]=="explicit" and r["event_count"] and math.isfinite(_float(r["median_interval_cycles"]))]
    _write(out/"abcd_1D_accelerated_explicit_rates.csv",[r for r in one_d if r["class"] in "ABCD"],COMMON_COLUMNS)
    _write(out/"abcd_2D_accelerated_explicit_rates.csv",[r for r in two_d if r["class"] in "ABCD"],COMMON_COLUMNS)
    _write(out/"abcd_hybrid_rates.csv",[r for r in hybrid if r["class"] in "ABCD"])
    _write(out/"abcd_mode_switch_diagnostics.csv",[r for r in diagnostics if r["class"] in "ABCD"])
    _write(out/"abcd_explicit_event_intervals.csv",[r for r in intervals if r["class"] in "ABCD"])
    _write(out/"dbtt_peak_hybrid_rates.csv",[r for r in rows if r["class"] in {"DBTT","Peak"}],COMMON_COLUMNS)
    plot_four_path(out,rows,list("ABCD"),"abcd_four_path_da_dN_vs_deltaK")
    plot_four_path(out,rows,["DBTT","Peak"],"dbtt_peak_four_path_da_dN_vs_deltaK")
    other_plots(out,rows,hybrid,intervals,diagnostics)
    write_provenance(repo,out,rows); write_report(out,rows,diagnostics)
    summary={"rows":len(rows),"one_d_rows":len(one_d),"two_d_rows":len(two_d),"matched_overlap_rows":len(diagnostics),
             "explicit_2d_rows":sum(r["dimensionality"]=="2D" and r["integration_mode"]=="explicit" for r in rows)}
    (out/"hybrid_analysis_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
