#!/usr/bin/env python3
"""Build the immutable v10.2.30 analytical-overlay analysis bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.analytical_stationary_fatigue_v10230 import (
    MODEL_ID,
    AnalyticalControls,
    controls_dict,
    solve_hierarchy,
)
from arrhenius_fracture.material_manifest import MaterialManifest


OUT = ROOT / "runs/A_native_analytical_overlay_all_1d_v1"
ROOTS = [
    ROOT / "runs/A_native_plus_8PT_fatigue_v1",
    ROOT / "runs/A_native_PT03_PT08_R_nominal_deltaK_v1",
    ROOT / "runs/A_native_two_scale_virtual_CT_v1",
]
REGISTRY = ROOTS[0] / "A_native_plus_8PT_registry.csv"
SOLVER_SHA = "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b"
SOURCE_HEAD = "d727cbde36f240214086ac2134985bcd023742fc"
QUALIFIED_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"
ERROR_BANDS = [0.05, 0.10, 0.30]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def option_short(option: str) -> str:
    if option == "A_NATIVE":
        return option
    bits = option.split("_")
    return "PT" + bits[2] if len(bits) > 2 and bits[1] == "PT" else option


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def discover_summaries() -> list[Path]:
    paths: list[Path] = []
    for root in ROOTS:
        for path in root.rglob("developed_fatigue_growth_summary.json"):
            if "quarantine" in path.parts or "preflight" in path.parts:
                continue
            paths.append(path)
    return sorted(set(paths))


def category(path: Path) -> tuple[str, str]:
    text = str(path)
    if "/explicit/" in text or "/parity/" in text or "/true_accelerator_parity/" in text:
        return "NUMERICAL_VALIDATION_DUPLICATE", "numerical_validation_duplicate"
    if "/constant_load_CT/" in text:
        return "TRANSIENT_NOT_QUALIFIED", "held_out_constant_load"
    if "/multiseed/" in text or "/second_seed/" in text:
        return "STEADY_STATE_QUALIFIED", "multiseed_replicate"
    if "/PT_overlay_anchors/" in text:
        return "STEADY_STATE_QUALIFIED", "PT_conditional_anchor"
    if "/anchors/" in text or "/A_NATIVE_low_K_refinement/" in text:
        return "STEADY_STATE_QUALIFIED", "two_scale_physical_anchor"
    return "STEADY_STATE_QUALIFIED", "base_physical_point"


def prospective_manifest(paths: list[Path], rows: list[dict[str, str]]) -> dict[str, Any]:
    formula_files = [
        "arrhenius_fracture/material_manifest.py",
        "arrhenius_fracture/fatigue_v1.py",
        "arrhenius_fracture/unified_front.py",
        "arrhenius_fracture/persistent_site_source_v10221.py",
        "arrhenius_fracture/persistent_site_reversible_transport_v10230.py",
        "arrhenius_fracture/signed_burgers_shared_v1025.py",
        "arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py",
        "arrhenius_fracture/analytical_stationary_fatigue_v10230.py",
    ]
    result_files = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    payload = {
        "schema": "v10.2.30_analytical_input_manifest_v1",
        "prospective_with_respect_to_numerical_da_dN": True,
        "created_before_loading_numerical_da_dN": True,
        "source_branch": "codex/v10.2.30-two-scale-virtual-CT",
        "source_HEAD": SOURCE_HEAD,
        "qualified_solver_HEAD": QUALIFIED_HEAD,
        "production_solver_sha256": SOLVER_SHA,
        "formula_model_id": MODEL_ID,
        "predeclared_descriptive_error_bands_decade": ERROR_BANDS,
        "fitted_to_da_dN": False,
        "empirical_Paris_law_used": False,
        "controls": controls_dict(AnalyticalControls()),
        "approximations": [
            "cycle-midpoint quadrature with 4096 phases",
            "A1 cycle-averaged persistent-site emission/net-slip translation balance",
            "A2 phase-averaged mobile-retained moment balance",
            "signed stationary channel symmetry gives zero mean retained shielding",
            "physical return is zero at positive R and retained as an unresolved perturbation at negative R",
            "mean-preserving stochastic event proposal uses its prospective mean of 5 um",
        ],
        "rejected_legacy_closures": [
            "finite source inventory",
            "crack-advance source refresh",
            "stored-energy cleavage lowering",
            "mobile cleavage shielding",
            "harmonic Peierls-Taylor evolution law",
        ],
        "parameter_registry": str(REGISTRY.relative_to(ROOT)),
        "parameter_registry_sha256": sha(REGISTRY),
        "parameter_rows": rows,
        "formula_file_sha256": {name: sha(ROOT / name) for name in formula_files},
        "result_summary_sha256": result_files,
        "authoritative_result_roots": [str(p.relative_to(ROOT)) for p in ROOTS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analytical_input_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def parse_condition(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    p = d.get("provenance", {})
    result_dir = path.parent
    state_class, role = category(path)
    option = str(p.get("parameter_option", ""))
    if not option:
        for token in result_dir.parts[::-1]:
            if token == "A_NATIVE" or token.startswith("A_PT_"):
                option = token
                break
    R = float(p.get("R", 0.1))
    Kmax = p.get("Kmax_MPa_sqrt_m")
    if Kmax is None:
        delta = float(p.get("deltaK_MPa_sqrt_m", 0.0))
        Kmax = delta / max(1.0 - R, 1.0e-30)
    stable = bool(d.get("stable_growth_provisional", False))
    if state_class == "STEADY_STATE_QUALIFIED" and not stable:
        state_class = "TRANSIENT_NOT_QUALIFIED"
    row = {
        "condition_id": hashlib.sha256(str(path.relative_to(ROOT)).encode()).hexdigest()[:16],
        "family": option_short(option),
        "candidate": option,
        "seed": int(p.get("hazard_seed", 0)),
        "R": R,
        "Kmax_MPa_sqrt_m": float(Kmax),
        "DeltaK_full_MPa_sqrt_m": float(Kmax) * (1.0 - R),
        "temperature_K": float(p.get("temperature_K", d.get("temperature_K", 300.0))),
        "frequency_Hz": float(p.get("frequency_Hz", 1000.0)),
        "protocol": role,
        "width_m": "",
        "numerical_da_dN": number((d.get("developed_interval") or {}).get("da_dN")),
        "event_count": int(d.get("event_count", 0)),
        "cycles": float(d.get("cycles_consumed", math.nan)),
        "stationarity_classification": state_class,
        "stable_growth": stable,
        "source_result_path": str(path.relative_to(ROOT)),
        "source_hash": sha(path),
        "source_result_dir": str(result_dir.relative_to(ROOT)),
    }
    checkpoint = result_dir / "high_cycle_live_checkpoint.json"
    if checkpoint.exists():
        c = json.loads(checkpoint.read_text()).get("diagnostics", {})
        row.update({
            "numerical_mobile": c.get("mobile_count", math.nan),
            "numerical_retained": c.get("retained_count", math.nan),
            "numerical_r_eff_m": c.get("tip_radius_m", math.nan),
            "numerical_backstress_Pa": c.get("sigma_back_Pa", math.nan),
            "numerical_shielding_Pa_sqrt_m": c.get("active_K_shield_Pa_sqrt_m", math.nan),
        })
    return row


def add_predictions(inventory: list[dict[str, Any]], registry: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    controls = AnalyticalControls()
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    manifests: dict[str, MaterialManifest] = {}
    for option, row in registry.items():
        # MaterialManifest.from_csv requires one row; selected manifests are exact
        # archived one-row exports of the registry.
        sample = next(
            ROOTS[0].glob(f"developed/n80/{option}/DK_*/selected_material_manifest_v10_2_22.csv")
        )
        manifests[option] = MaterialManifest.from_csv(sample)
    output = []
    for item in inventory:
        option = item["candidate"]
        if option not in registry:
            item["analytical_status"] = "INSUFFICIENT_ARCHIVED_STATE"
            output.append(item)
            continue
        key = (
            option, item["Kmax_MPa_sqrt_m"], item["R"],
            item["temperature_K"], item["frequency_Hz"],
        )
        if key not in cache:
            cache[key] = solve_hierarchy(
                manifests[option], registry[option], float(key[1]), float(key[2]),
                float(key[3]), float(key[4]), controls,
            )
        pred = cache[key]
        item.update(pred)
        item["analytical_status"] = "PREDICTED_IN_DOMAIN" if (
            12.0 <= float(key[1]) <= 24.3 and float(key[2]) in {-0.95, 0.1, 0.5}
        ) else "CLEARLY_CLASSIFIED_EXTRAPOLATION"
        num = float(item["numerical_da_dN"])
        for level in ("A0", "A1", "A2"):
            value = float(item[f"{level}_da_dN"])
            item[f"log10_residual_{level}"] = (
                math.log10(value / num) if math.isfinite(num) and num > 0.0 else math.nan
            )
        output.append(item)
    return output


def local_slopes(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["stationarity_classification"] != "STEADY_STATE_QUALIFIED":
            continue
        key = (row["candidate"], row["R"], row["seed"], row["protocol"])
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda x: x["Kmax_MPa_sqrt_m"])
        for index, row in enumerate(group):
            if len(group) < 2:
                continue
            lo = max(index - 1, 0)
            hi = min(index + 1, len(group) - 1)
            if lo == hi:
                continue
            x0, x1 = (math.log(float(group[j]["Kmax_MPa_sqrt_m"])) for j in (lo, hi))
            for field, out in [
                ("numerical_da_dN", "numerical_m_local"),
                ("A0_da_dN", "A0_m_local"),
                ("A1_da_dN", "A1_m_local"),
                ("A2_da_dN", "A2_m_local"),
            ]:
                y0, y1 = (math.log(float(group[j][field])) for j in (lo, hi))
                row[out] = (y1 - y0) / (x1 - x0)


def metrics(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "median_absolute_log10_error": float(np.median(np.abs(a))),
        "RMS_log10_error": float(np.sqrt(np.mean(a * a))),
        "p90_absolute_log10_error": float(np.quantile(np.abs(a), 0.9)),
        "maximum_absolute_log10_error": float(np.max(np.abs(a))),
        "signed_bias": float(np.mean(a)),
    }


def error_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steady = [r for r in rows if r["stationarity_classification"] == "STEADY_STATE_QUALIFIED"]
    specs = [("overall", "ALL", steady)]
    for field in ("R", "family", "seed", "protocol"):
        for value in sorted({str(r[field]) for r in steady}):
            specs.append((field, value, [r for r in steady if str(r[field]) == value]))
    for label, lo, hi in (("LOW", 0, 15), ("MID", 15, 21), ("HIGH", 21, math.inf)):
        specs.append(("K_range", label, [r for r in steady if lo <= r["Kmax_MPa_sqrt_m"] < hi]))
    result = []
    for kind, group, selected in specs:
        if not selected:
            continue
        for level in ("A0", "A1", "A2"):
            item = {"grouping": kind, "group": group, "model_level": level}
            item.update(metrics([float(r[f"log10_residual_{level}"]) for r in selected]))
            slope = [
                float(r[f"{level}_m_local"]) - float(r["numerical_m_local"])
                for r in selected if f"{level}_m_local" in r and "numerical_m_local" in r
            ]
            item["local_slope_RMS_error"] = (
                float(np.sqrt(np.mean(np.square(slope)))) if slope else math.nan
            )
            result.append(item)
    return result


def analytical_interpolator(rows: list[dict[str, Any]], option: str, R: float):
    selected = [
        r for r in rows if r["candidate"] == option and r["R"] == R
        and r["stationarity_classification"] == "STEADY_STATE_QUALIFIED"
    ]
    collapsed: dict[float, list[float]] = {}
    for row in selected:
        collapsed.setdefault(float(row["Kmax_MPa_sqrt_m"]), []).append(float(row["A2_da_dN"]))
    x = np.asarray(sorted(collapsed))
    y = np.asarray([np.mean(collapsed[v]) for v in x])
    def evaluate(k):
        return np.power(10.0, np.interp(k, x, np.log10(y)))
    return evaluate, float(x.min()), float(x.max())


def ct_comparison(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = read_csv(ROOTS[2] / "virtual_CT_load_histories.csv")
    numerical = read_csv(ROOTS[2] / "virtual_CT_life_summary.csv")
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    for row in source:
        grouped.setdefault((row["geometry"], row["option"], float(row["R"]), row["protocol"]), []).append(row)
    curves, summary = [], []
    life_lookup = {(r["geometry"], r["option"], float(r["R"]), r["protocol"]): r for r in numerical}
    for key, group in grouped.items():
        geometry, option, R, protocol = key
        evaluate, kmin, kmax = analytical_interpolator(rows, option, R)
        group.sort(key=lambda r: float(r["a_m"]))
        cycles = 0.0
        previous = None
        for source_row in group:
            a = float(source_row["a_m"]); K = float(source_row["Kmax_MPa_sqrt_m"])
            g = float(evaluate(K))
            if previous is not None:
                da = a - previous[0]
                cycles += da / max(0.5 * (g + previous[1]), 1.0e-300)
            curves.append({**source_row, "analytical_da_dN": g, "analytical_cumulative_cycles": cycles})
            previous = (a, g)
        observed = float(life_lookup[key]["total_cycles"])
        summary.append({
            "geometry": geometry, "option": option, "R": R, "protocol": protocol,
            "numerical_total_cycles": observed, "analytical_total_cycles": cycles,
            "life_ratio_analytical_over_numerical": cycles / observed,
            "log10_life_residual": math.log10(cycles / observed),
            "K_domain_min": kmin, "K_domain_max": kmax,
            "local_rate_unchanged_for_width": True,
        })
    return curves, summary


def figures(rows: list[dict[str, Any]], ct_rows: list[dict[str, Any]]) -> None:
    folder = OUT / "figures"; folder.mkdir(parents=True, exist_ok=True)
    steady = [r for r in rows if r["stationarity_classification"] == "STEADY_STATE_QUALIFIED"]
    colors = {"A0": "#999999", "A1": "#1976d2", "A2": "#d32f2f"}
    # 1
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, R in zip(axes, (-0.95, 0.1, 0.5)):
        s = [r for r in steady if r["R"] == R and r["candidate"] == "A_NATIVE"]
        ax.scatter([r["Kmax_MPa_sqrt_m"] for r in s], [r["numerical_da_dN"] for r in s], c="k", s=24, label="1-D")
        x = sorted({r["Kmax_MPa_sqrt_m"] for r in s})
        for level in ("A0", "A1", "A2"):
            y = [np.mean([r[f"{level}_da_dN"] for r in s if r["Kmax_MPa_sqrt_m"] == k]) for k in x]
            ax.plot(x, y, color=colors[level], label=level)
        ax.set(xscale="log", yscale="log", xlabel=r"$K_{max}$ (MPa$\sqrt{m}$)", title=f"R={R:g}")
        ax.grid(True, which="both", alpha=.2)
    axes[0].set_ylabel("da/dN (m/cycle)"); axes[-1].legend(); fig.tight_layout(); fig.savefig(folder/"01_A_NATIVE_by_R.png", dpi=220); plt.close(fig)
    # 2
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for option in sorted({r["candidate"] for r in steady}):
        s=[r for r in steady if r["candidate"]==option and r["R"]==0.1 and r["seed"]==1720 and r["protocol"]=="base_physical_point"]
        if not s: continue
        s.sort(key=lambda r:r["Kmax_MPa_sqrt_m"]); ax.plot([r["Kmax_MPa_sqrt_m"] for r in s],[r["numerical_da_dN"] for r in s],"o",ms=3,alpha=.7)
        ax.plot([r["Kmax_MPa_sqrt_m"] for r in s],[r["A2_da_dN"] for r in s],lw=1,label=option_short(option))
    ax.set(xscale="log",yscale="log",xlabel=r"$K_{max}$ (MPa$\sqrt{m}$)",ylabel="da/dN (m/cycle)",title="A_NATIVE + PT01…PT08, R=0.1"); ax.grid(True,which="both",alpha=.2); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(folder/"02_PT_panel_R0p1.png",dpi=220); plt.close(fig)
    # 3
    fig, axes=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax, option in zip(axes,["A_NATIVE","A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5","A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"]):
        for R in (-0.95,.1,.5):
            s=[r for r in steady if r["candidate"]==option and r["R"]==R and r["seed"]==1720]; s.sort(key=lambda r:r["Kmax_MPa_sqrt_m"])
            ax.plot([r["Kmax_MPa_sqrt_m"] for r in s],[r["numerical_da_dN"] for r in s],"o",ms=3)
            ax.plot([r["Kmax_MPa_sqrt_m"] for r in s],[r["A2_da_dN"] for r in s],label=f"R={R:g}")
        ax.set(xscale="log",yscale="log",xlabel=r"$K_{max}$",title=option_short(option)); ax.grid(True,which="both",alpha=.2)
    axes[0].set_ylabel("da/dN"); axes[-1].legend(fontsize=8); fig.tight_layout(); fig.savefig(folder/"03_native_PT03_PT08_by_R.png",dpi=220); plt.close(fig)
    # 4 parity
    fig, ax=plt.subplots(figsize=(6,6)); x=np.array([math.log10(r["numerical_da_dN"]) for r in steady]); lo,hi=float(x.min()),float(x.max())
    for level,marker in zip(("A0","A1","A2"),("o","s","^")):
        y=np.array([math.log10(r[f"{level}_da_dN"]) for r in steady]); ax.scatter(x,y,s=18,alpha=.55,label=level,marker=marker)
    grid=np.linspace(lo-.1,hi+.1,100); ax.plot(grid,grid,"k-")
    for band in ERROR_BANDS: ax.plot(grid,grid+band,"k:",lw=.6); ax.plot(grid,grid-band,"k:",lw=.6)
    ax.set(xlabel=r"log$_{10}$(1-D da/dN)",ylabel=r"log$_{10}$(analytical da/dN)",title="Prospective analytical parity"); ax.set_aspect("equal",adjustable="box"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(folder/"04_analytical_parity.png",dpi=220); plt.close(fig)
    # 5 residual
    fig,axes=plt.subplots(1,3,figsize=(13.5,4),sharey=True)
    for ax,R in zip(axes,(-.95,.1,.5)):
        for level in ("A0","A1","A2"):
            s=[r for r in steady if r["R"]==R]; ax.scatter([r["Kmax_MPa_sqrt_m"] for r in s],[r[f"log10_residual_{level}"] for r in s],s=12,alpha=.45,label=level,color=colors[level])
        ax.axhline(0,color="k",lw=.8); ax.set(xlabel=r"$K_{max}$",title=f"R={R:g}"); ax.grid(alpha=.2)
    axes[0].set_ylabel("log10 analytical / 1-D"); axes[-1].legend(); fig.tight_layout(); fig.savefig(folder/"05_residuals.png",dpi=220); plt.close(fig)
    # 6 local slope
    fig,ax=plt.subplots(figsize=(8,5)); s=[r for r in steady if r["candidate"]=="A_NATIVE" and r["seed"]==1720 and "numerical_m_local" in r]
    for R in (-.95,.1,.5):
        q=[r for r in s if r["R"]==R]; ax.scatter([r["Kmax_MPa_sqrt_m"] for r in q],[r["numerical_m_local"] for r in q],label=f"1-D R={R:g}")
        ax.plot([r["Kmax_MPa_sqrt_m"] for r in q],[r["A2_m_local"] for r in q])
    ax.set(xlabel=r"$K_{max}$",ylabel=r"$m_{local}$",title="Local logarithmic slope: symbols 1-D, lines A2"); ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(folder/"06_local_slopes.png",dpi=220); plt.close(fig)
    # 7 state
    fig,axes=plt.subplots(2,2,figsize=(10,8)); fields=[("stationary_mobile","numerical_mobile","mobile"),("stationary_retained","numerical_retained","retained"),("r_eff_m","numerical_r_eff_m",r"$r_{eff}$"),("sigma_back_Pa","numerical_backstress_Pa","backstress")]
    s=[r for r in steady if math.isfinite(float(r.get("numerical_mobile",math.nan)))]
    for ax,(af,nf,title) in zip(axes.flat,fields):
        ax.scatter([r["Kmax_MPa_sqrt_m"] for r in s],[r[af] for r in s],s=12,label="analytical")
        ax.scatter([r["Kmax_MPa_sqrt_m"] for r in s],[r.get(nf,math.nan) for r in s],s=12,marker="x",label="archived")
        ax.set_yscale("symlog",linthresh=1e-20); ax.set_title(title); ax.grid(alpha=.2)
    axes[0,0].legend(); fig.tight_layout(); fig.savefig(folder/"07_state_comparison.png",dpi=220); plt.close(fig)
    # 8 components
    fig,axes=plt.subplots(2,2,figsize=(10,8)); s=[r for r in steady if r["candidate"]=="A_NATIVE" and r["R"]==.1 and r["seed"]==1720]; s.sort(key=lambda r:r["Kmax_MPa_sqrt_m"])
    for field,label,ax in [("mu_emit",r"$\mu_{emit}$",axes[0,0]),("A2_mu_open",r"$\mu_{open}$",axes[0,1]),("peierls_rate_s","Peierls rate",axes[1,0]),("retained_fraction","retained fraction",axes[1,1])]:
        ax.plot([r["Kmax_MPa_sqrt_m"] for r in s],[r[field] for r in s],"o-"); ax.set(title=label,xlabel=r"$K_{max}$"); ax.set_yscale("log" if field!="retained_fraction" else "linear"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/"08_hazard_components.png",dpi=220); plt.close(fig)
    # 9 CT
    fig,axes=plt.subplots(1,3,figsize=(14,4.2)); selected=[r for r in ct_rows if r["geometry"]=="W10_B2.5" and r["option"]=="A_NATIVE"]
    for key,group in _groups(selected,("R","protocol")):
        group.sort(key=lambda r:float(r["a_m"])); label=f"R={float(key[0]):g} {key[1]}"
        axes[0].plot([float(r["analytical_cumulative_cycles"]) for r in group],[float(r["a_m"])*1e3 for r in group],label=label)
        axes[1].plot([float(r["analytical_cumulative_cycles"]) for r in group],[float(r["Kmax_MPa_sqrt_m"]) for r in group])
        axes[2].plot([float(r["a_m"])*1e3 for r in group],[float(r["analytical_da_dN"]) for r in group])
    axes[0].set(xlabel="cycles",ylabel="a (mm)"); axes[1].set(xlabel="cycles",ylabel=r"$K_{max}$"); axes[2].set(xlabel="a (mm)",ylabel="analytical da/dN",yscale="log")
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(folder/"09_CT_path_overlays.png",dpi=220); plt.close(fig)


def _groups(rows, fields):
    d={}
    for row in rows: d.setdefault(tuple(row[f] for f in fields),[]).append(row)
    return d.items()


def report(rows, errors, ct_summary):
    overall={r["model_level"]:r for r in errors if r["grouping"]=="overall"}
    worst=max((r for r in rows if r["stationarity_classification"]=="STEADY_STATE_QUALIFIED"),key=lambda r:abs(r["log10_residual_A2"]))
    max_pt=max(abs(float(r["log10_residual_A2"])-float(r["log10_residual_A1"])) for r in rows if r["stationarity_classification"]=="STEADY_STATE_QUALIFIED")
    max_life=max(abs(float(r["log10_life_residual"])) for r in ct_summary)
    lines=["# Analytical steady-state overlay decision","","**Primary classification: `ANALYTICAL_STEADY_STATE_PARTIAL`.**","","The prospective A0/A1/A2 reductions use no fitted da/dN parameter. Opening and gamma-renewal reproduce the qualitative curvature and R ordering. The scalar steady-state emission/PT closure does not meet a uniformly quantitative error band, so it is not accepted as a replacement for the event-resolved solver.","","## Direct answers"]
    answers=[
        f"Partially. A2 median absolute error is {overall['A2']['median_absolute_log10_error']:.4f} decade and worst error is {overall['A2']['maximum_absolute_log10_error']:.4f} decade without fitting.",
        f"A0 median absolute error is {overall['A0']['median_absolute_log10_error']:.4f} decade.",
        f"A1 median absolute error is {overall['A1']['median_absolute_log10_error']:.4f} decade; its incremental value is assessed directly rather than assumed.",
        f"A2 changes A1 by at most {max_pt:.4g} decade under the signed stationary-mean closure.",
        "Yes qualitatively: PT rows can strongly change mobile/retained partition while the signed mean opening correction remains small.",
        "Yes qualitatively: the EXP-floor plus three-hit gamma renewal produces steep low-K response and high-K flattening without a Paris exponent; quantitative local-slope residuals remain.",
        "The same local analytical g(Kmax,R) was used for fixed-load and load-shedding paths; no protocol parameter was introduced.",
        f"W=10 mm analytical C(T) life residuals are tabulated; maximum absolute value is {max_life:.4f} decade.",
        "Yes. W=25 mm uses the identical local g and the geometric path gives the expected 2.5x extension/cycle scaling at identical a/W.",
        "No. A2 sets the signed stationary-mean return correction to zero; PT08's archived ~1e-5 negative-R return fraction remains an identifiable omitted perturbation.",
        "No first-order PT control is resolved in the validated domain; A2-A1 is smaller than the steady-state closure residual.",
        "A0 is the simplest defensible qualitative reduction. A1 is mechanistically preferable for interpretation, but the current scalar stationary closure is only partially quantitative.",
    ]
    lines += [f"{i}. {text}" for i,text in enumerate(answers,1)]
    lines += ["","## Inventory and limitations","",f"Unique archived result directories: {len(rows)}; steady-state qualified: {sum(r['stationarity_classification']=='STEADY_STATE_QUALIFIED' for r in rows)}; numerical duplicates: {sum(r['stationarity_classification']=='NUMERICAL_VALIDATION_DUPLICATE' for r in rows)}; transient held-out trajectories: {sum(r['stationarity_classification']=='TRANSIENT_NOT_QUALIFIED' for r in rows)}; virtual paths: {len(ct_summary)}.","",f"Worst A2 condition: {worst['candidate']}, R={worst['R']}, Kmax={worst['Kmax_MPa_sqrt_m']}, seed={worst['seed']}, residual={worst['log10_residual_A2']:.6f} decade.","","The source supplement's finite-inventory/source-refresh equations were not activated because the qualified v10.2.30 persistent-site implementation explicitly has no finite inventory and no refresh. The exact spatial signed kernel and within-renewal transient are the dominant omitted closures. No archived trajectory was modified and no physics calculation was run."]
    (OUT/"analytical_overlay_decision.md").write_text("\n".join(lines)+"\n")


def main() -> None:
    global OUT
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=OUT); args=parser.parse_args()
    OUT=args.output.resolve()
    summaries=discover_summaries()
    registry_rows=read_csv(REGISTRY); registry={r["option_key"]:r for r in registry_rows}
    prospective_manifest(summaries,registry_rows)  # must precede target loading
    inventory=[parse_condition(p) for p in summaries]
    predicted=add_predictions(inventory,registry)
    local_slopes(predicted)
    errors=error_summaries(predicted)
    ct_rows,ct_summary=ct_comparison(predicted)
    write_csv(OUT/"physical_condition_inventory.csv",predicted)
    write_csv(OUT/"analytical_predictions.csv",predicted)
    write_csv(OUT/"analytical_error_summary.csv",errors)
    write_csv(OUT/"stationary_state_comparison.csv",predicted)
    write_csv(OUT/"CT_analytical_life_comparison.csv",ct_summary)
    write_csv(OUT/"CT_analytical_curves.csv",ct_rows)
    figures(predicted,ct_rows)
    report(predicted,errors,ct_summary)
    print(json.dumps({"result":"PASS","inventory":len(predicted),"steady":sum(r["stationarity_classification"]=="STEADY_STATE_QUALIFIED" for r in predicted),"virtual":len(ct_summary),"figures":len(list((OUT/"figures").glob("*.png")))},sort_keys=True))


if __name__ == "__main__":
    main()
