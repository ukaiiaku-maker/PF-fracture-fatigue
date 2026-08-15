#!/usr/bin/env python3
"""Merge historic v9.13/v9.14 results and rank material-like fatigue knees."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


V913 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
V914 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search")
TEMP_TABLE = V913 / "runs/v9_13_zeroD_promoted_1d_384_50um_v2/one_d_screen/ranked_candidates.csv"
GLOBAL_REGISTRY = V914 / "runtime_inputs/v914/endurance_knee_global_300K_1024.csv"
MECHANISMS = V914 / "runs/v914_endurance_knee_mechanism_classification_475/mechanism_classification.csv"
CURRENT_CONTROLS = {
    "v914_endurance_knee_0462": "A", "v914_endurance_knee_0658": "B",
    "v914_endurance_knee_0554": "C", "v914_endurance_knee_0133": "D",
}
CANONICAL_TEMP = {
    "v913_zeroD_sobol_0202500": "DBTT",
    "v913_zeroD_sobol_0242980": "PEAK_T",
    "v913_zeroD_sobol_0129902": "WEAK_T",
    "v913_zeroD_sobol_0077080": "CERAMIC_LIKE",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def temp_class(row: pd.Series) -> str:
    dbtt = str(row.get("y__directional_dbtt_ge_threshold_1d", "")).lower() == "true"
    peak = str(row.get("y__peak_like_1d", "")).lower() == "true"
    if dbtt and peak:
        return "DBTT_PEAK_T"
    if dbtt:
        return "DBTT"
    if peak:
        return "PEAK_T"
    return "OTHER_OR_UNCLASSIFIED"


def collect_fatigue(roots: list[Path]) -> pd.DataFrame:
    records = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("fatigue_result.json"):
            # Prefer completed target-reaching records when duplicate case names exist.
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            cid = data.get("candidate_id")
            if not cid:
                continue
            fraction = finite(data.get("fraction"))
            seed = data.get("seed")
            key = (cid, fraction, seed, finite(data.get("loading", {}).get("deltaK_MPa_sqrt_m")))
            status = str(data.get("status", "unknown"))
            rate = finite(data.get("developed_da_dN_m_per_cycle"))
            ext = finite(data.get("final_extension_m"))
            cycles = finite(data.get("final_cycles"))
            rate_basis = "developed"
            # Historic v9.14 screening trajectories intentionally stopped at
            # 10--25 um and therefore did not populate the later 100-um
            # developed-rate field.  Their complete target-reaching secant
            # rate is valid for scouting, but is labelled distinctly.
            if not math.isfinite(rate) and status in {"target_extension_reached", "growth_target_reached"} and ext > 0 and cycles > 0:
                rate = ext / cycles
                rate_basis = "complete_short_trajectory_secant"
            dk = finite(data.get("loading", {}).get("deltaK_MPa_sqrt_m"))
            if not math.isfinite(dk):
                ref = finite(data.get("reference_deltaK_MPa_sqrt_m"))
                dk = ref * fraction
            record = {
                "candidate_id": cid, "fraction": fraction, "deltaK_MPa_sqrt_m": dk,
                "da_dN_m_per_cycle": rate, "status": status,
                "rate_basis": rate_basis,
                "final_extension_um": ext * 1e6, "final_cycles": cycles,
                "seed": seed, "source_path": str(path), "git_head": data.get("git_head", ""),
                "monotonic_K50_300K_MPa_sqrt_m": finite(data.get("monotonic_K50_300K_MPa_sqrt_m")),
                "reference_deltaK_MPa_sqrt_m": finite(data.get("reference_deltaK_MPa_sqrt_m")),
                "action_per_cycle": finite(data.get("current_action_per_cycle")),
            }
            quality = (math.isfinite(rate), ext, path.stat().st_mtime)
            if key not in seen:
                records.append((key, quality, record)); seen.add(key)
            else:
                for i, (k, q, _) in enumerate(records):
                    if k == key and quality > q:
                        records[i] = (key, quality, record)
                        break
    return pd.DataFrame([x[2] for x in records])


def curve_metrics(group: pd.DataFrame) -> dict:
    g = group[np.isfinite(group.da_dN_m_per_cycle) & (group.da_dN_m_per_cycle > 0)].copy()
    g = g.sort_values("deltaK_MPa_sqrt_m").drop_duplicates("deltaK_MPa_sqrt_m")
    result = {"finite_points": len(g), "maximum_calculated_rate": np.nan,
              "dynamic_fatigue_rate_range_decades": np.nan, "knee_location_MPa_sqrt_m": np.nan,
              "knee_width_MPa_sqrt_m": np.nan, "m_low": np.nan, "m_knee": np.nan,
              "m_high": np.nan, "R_recovery": np.nan, "knee_quality": np.nan,
              "highK_rate_slope": np.nan}
    if g.empty:
        return result
    x = g.deltaK_MPa_sqrt_m.to_numpy(float)
    y = np.log10(g.da_dN_m_per_cycle.to_numpy(float))
    result["maximum_calculated_rate"] = 10 ** np.nanmax(y)
    result["dynamic_fatigue_rate_range_decades"] = np.nanmax(y) - np.nanmin(y)
    if len(g) < 4:
        return result
    slopes = np.diff(y) / np.diff(x)
    centers = (x[:-1] + x[1:]) / 2
    # Select the interior weak-slope interval that is surrounded by the
    # strongest two-sided recovery.  A terminal plateau is therefore not
    # misidentified as a localized knee.
    candidates = []
    for j in range(1, len(slopes) - 1):
        ml = float(np.median(np.abs(slopes[:j])))
        mk = float(abs(slopes[j]))
        mh = float(np.median(np.abs(slopes[j + 1:])))
        candidates.append((min(ml, mh) / max(mk, 1e-12), j, ml, mk, mh))
    if not candidates:
        return result
    _, k, ml, mk, mh = max(candidates)
    result.update(knee_location_MPa_sqrt_m=float(centers[k]),
                  knee_width_MPa_sqrt_m=float(x[min(k + 1, len(x)-1)] - x[k]),
                  m_low=ml, m_knee=mk, m_high=mh,
                  R_recovery=mh / max(mk, 1e-12),
                  knee_quality=ml / max(mk, 1e-12), highK_rate_slope=mh)
    return result


def mechanism_letter(value: str) -> str:
    value = str(value)
    return value[0] if value and value[0] in "ABCD" else "SMOOTH_ARRHENIUS"


def savefig(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(curves, info, ids, base, title=None, panels=False):
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        for ax, letter in zip(axes.flat, "ABCD"):
            chosen = [c for c in ids if info.loc[c, "mechanism_letter"] == letter][:10]
            draw_curves(ax, curves, info, chosen)
            ax.text(.02, .96, letter, transform=ax.transAxes, va="top", fontsize=15, weight="bold")
        for ax in axes[:, 0]: ax.set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
        for ax in axes[-1, :]: ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_curves(ax, curves, info, ids)
        ax.set_xlabel(r"$\Delta K$ (MPa $\sqrt{m}$)")
        ax.set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
    if title: fig.suptitle(title)
    savefig(fig, base)


def draw_curves(ax, curves, info, ids):
    styles = {"DBTT": "-", "PEAK_T": "--", "DBTT_PEAK_T": "-.", "OTHER_OR_UNCLASSIFIED": ":"}
    for rank, cid in enumerate(ids):
        g = curves[(curves.candidate_id == cid) & np.isfinite(curves.da_dN_m_per_cycle)].sort_values("deltaK_MPa_sqrt_m")
        if g.empty: continue
        row = info.loc[cid]
        label = f"{cid.replace('v914_endurance_knee_','').replace('v913_zeroD_sobol_','')} [{row.temperature_class}/{row.mechanism_letter}]"
        ax.plot(g.deltaK_MPa_sqrt_m, g.da_dN_m_per_cycle, styles.get(row.temperature_class, ":"),
                marker="o", ms=4, lw=2.2 if rank < 10 else 1.0, alpha=1 if rank < 10 else .48, label=label)
        cens = curves[(curves.candidate_id == cid) & ~np.isfinite(curves.da_dN_m_per_cycle)]
        if not cens.empty:
            floor = max(1e-20, np.nanmin(g.da_dN_m_per_cycle) / 5)
            ax.scatter(cens.deltaK_MPa_sqrt_m, np.full(len(cens), floor), marker="v", facecolors="none", edgecolors="0.35", s=30)
    ax.set_yscale("log"); ax.grid(False); ax.tick_params(labelsize=11)
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-fatigue-root", action="append", type=Path, default=[])
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)

    temp = pd.read_csv(TEMP_TABLE, dtype=str)
    temp["temperature_class"] = temp.apply(temp_class, axis=1)
    mech = pd.read_csv(MECHANISMS)
    local_mech_path = out.parent / "v913_state_classification/mechanism_classification.csv"
    if local_mech_path.exists():
        mech = pd.concat([mech, pd.read_csv(local_mech_path)], ignore_index=True).drop_duplicates("candidate_id", keep="last")
    registry = pd.read_csv(GLOBAL_REGISTRY, dtype=str)
    roots = [V914 / "runs", Path("runs/v10_2_31_endurance_knee_ABCD_high_deltaK_v1")] + args.extra_fatigue_root
    curves = collect_fatigue(roots)
    curves.to_csv(out / "candidate_fatigue_curves.csv", index=False)

    metrics = pd.DataFrame([{"candidate_id": cid, **curve_metrics(g)} for cid, g in curves.groupby("candidate_id")])
    mechanism_map = dict(zip(mech.candidate_id, mech.mechanism_class))
    rows = []
    for _, r in temp.iterrows():
        rows.append({
            "candidate_id": r.candidate_id, "candidate_source_database": "V913_ORIGINAL_TEMP_SEARCH",
            "temperature_class": r.temperature_class,
            "original_DBTT_score": finite(r.get("y__directional_dbtt_gain")),
            "original_Peak_T_score": finite(r.get("y__peak_prominence")),
            "low_temperature_resistance": finite(r.get("y__low_temperature_baseline")),
            "high_temperature_resistance": finite(r.get("y__high_temperature_plateau")),
            "peak_temperature_K": finite(r.get("y__peak_temperature_K")),
            "mechanism_class": mechanism_map.get(r.candidate_id, "MISSING"),
            "provenance_fingerprint": digest(TEMP_TABLE),
        })
    existing = {x["candidate_id"] for x in rows}
    for _, r in registry.iterrows():
        if r.candidate_id in existing: continue
        rows.append({"candidate_id": r.candidate_id, "candidate_source_database": "V914_GLOBAL_KNEE_SEARCH",
                     "temperature_class": CANONICAL_TEMP.get(r.candidate_id, "OTHER_OR_UNCLASSIFIED"),
                     "original_DBTT_score": np.nan, "original_Peak_T_score": np.nan,
                     "mechanism_class": mechanism_map.get(r.candidate_id, "MISSING"),
                     "provenance_fingerprint": digest(GLOBAL_REGISTRY)})
    # Include local/canonical fatigue candidates absent from the two broad registries.
    for cid in sorted(set(curves.candidate_id) - {x["candidate_id"] for x in rows}):
        rows.append({"candidate_id": cid, "candidate_source_database": "V914_EXISTING_FATIGUE_SCREEN",
                     "temperature_class": CANONICAL_TEMP.get(cid, "OTHER_OR_UNCLASSIFIED"),
                     "original_DBTT_score": np.nan, "original_Peak_T_score": np.nan,
                     "mechanism_class": mechanism_map.get(cid, CURRENT_CONTROLS.get(cid, "MISSING")),
                     "provenance_fingerprint": "per-result git_head/registry_sha256"})
    master = pd.DataFrame(rows).merge(metrics, on="candidate_id", how="left")
    scales = curves.groupby("candidate_id").agg(monotonic_K50_300K_MPa_sqrt_m=("monotonic_K50_300K_MPa_sqrt_m", "max"),
                                                  existing_fatigue_curve_points=("candidate_id", "size"),
                                                  censor_count=("da_dN_m_per_cycle", lambda x: int(x.isna().sum())),
                                                  maximum_action_per_cycle=("action_per_cycle", "max"),
                                                  maximum_deltaK_MPa_sqrt_m=("deltaK_MPa_sqrt_m", "max")).reset_index()
    master = master.merge(scales, on="candidate_id", how="left")
    master["mechanism_letter"] = master.mechanism_class.map(mechanism_letter)
    # Unit-exponential renewals and the preserved mean 5-um proposal imply
    # E[da/dN] = da_phys*dH/dN before geometry/energy truncation.  This is a
    # constitutive hazard ceiling estimate, not a replacement fatigue law.
    master["predicted_high_rate_ceiling"] = 5e-6 * master.maximum_action_per_cycle
    # Kmax=DeltaK/(1-R), r0=1 um, sigma=K/sqrt(2*pi*r); production clips at 30 GPa.
    stress_cap_deltaK = .9 * 30e9 * math.sqrt(2 * math.pi * 1e-6) / 1e6
    master["stress_cap_encountered_at_max_load"] = master.maximum_deltaK_MPa_sqrt_m >= stress_cap_deltaK
    master["high_rate_limitation"] = np.where(master.stress_cap_encountered_at_max_load, "STRESS_CAP_LIMITED",
        np.where(master.maximum_calculated_rate >= 1e-3, "NEAR_MONOTONIC",
        np.where(master.m_high.fillna(np.inf) < 0.01, "BARRIER_OR_EVENT_LAW_LIMITED", "BARRIER_LIMITED")))
    master["R_recovery"] = master.R_recovery.replace([np.inf, -np.inf], np.nan)
    high_rate_score = (np.log10(master.maximum_calculated_rate.clip(lower=1e-30)) + 6).clip(0, 6)
    master["compromise_score"] = (master.knee_quality.clip(upper=20).fillna(0) +
                                   2 * master.R_recovery.clip(upper=20).fillna(0) +
                                   3 * high_rate_score.fillna(0) +
                                   master.dynamic_fatigue_rate_range_decades.fillna(0) +
                                   master.temperature_class.isin(["DBTT", "PEAK_T", "DBTT_PEAK_T"]) * 5 -
                                   (master.R_recovery.fillna(0) <= 1) * 12 -
                                   (master.maximum_calculated_rate.fillna(0) < 1e-6) * 12)
    master["recommendation_category"] = np.where(master.finite_points.fillna(0) < 4, "NUMERICALLY_UNRESOLVED",
        np.where(master.maximum_calculated_rate.fillna(0) < 1e-6, "LOW_RATE_CEILING",
        np.where(master.R_recovery.fillna(0) <= 1, "NO_CLEAR_KNEE",
        np.where(master.temperature_class.str.contains("DBTT"), "STRONG_DBTT_KNEE_HIGHK",
        np.where(master.temperature_class.eq("PEAK_T"), "STRONG_PEAK_KNEE_HIGHK", "STRONG_KNEE_OTHER_TEMP")))))
    master.to_csv(out / "candidate_rerank_master.csv", index=False)
    eligible = master[master.finite_points.fillna(0) >= 4].sort_values("compromise_score", ascending=False)
    dbtt = master[master.temperature_class.str.contains("DBTT")].sort_values(["compromise_score", "original_DBTT_score"], ascending=False)
    peak = master[master.temperature_class.str.contains("PEAK")].sort_values(["compromise_score", "original_Peak_T_score"], ascending=False)
    knee = eligible.sort_values(["recommendation_category", "compromise_score", "R_recovery"],
                                key=lambda s: s.map({"STRONG_DBTT_KNEE_HIGHK": 0, "STRONG_PEAK_KNEE_HIGHK": 1,
                                                     "STRONG_KNEE_OTHER_TEMP": 2, "MECHANISTIC_CONTROL": 3,
                                                     "NO_CLEAR_KNEE": 4, "LOW_RATE_CEILING": 5,
                                                     "NUMERICALLY_UNRESOLVED": 6}) if s.name == "recommendation_category" else -s,
                                ascending=True)
    abcd = eligible.sort_values(["mechanism_letter", "compromise_score"], ascending=[True, False])
    for name, frame in [("candidate_rank_DBTT.csv", dbtt), ("candidate_rank_peak.csv", peak),
                        ("candidate_rank_knee_highK.csv", knee), ("candidate_rank_ABCD.csv", abcd)]:
        frame.to_csv(out / name, index=False)
    # Bounded diverse shortlist: authoritative finite curves now; missing historic rows remain in master/rank tables.
    selected = sorted(set(curves[curves.source_path.str.contains("v914_endurance_knee_rerank_DBTT_highK_v1")].candidate_id)) if args.extra_fatigue_root else []
    for cid in CURRENT_CONTROLS:
        if cid in set(eligible.candidate_id): selected.append(cid)
    for klass, n in [("A", 8), ("B", 8), ("C", 8), ("D", 8), ("SMOOTH_ARRHENIUS", 32)]:
        selected += [c for c in abcd[abcd.mechanism_letter == klass].candidate_id if c not in selected][:n]
    for cid in ["v913_zeroD_sobol_0202500", "v913_zeroD_sobol_0242980"]:
        if cid in set(eligible.candidate_id) and cid not in selected: selected.append(cid)
    selected = selected[:50]
    shortlist = master.set_index("candidate_id").loc[selected].reset_index()
    shortlist.insert(0, "shortlist_rank", range(1, len(shortlist) + 1))
    shortlist.to_csv(out / "candidate_shortlist.csv", index=False)
    shortlist.to_csv(out / "candidate_highK_metrics.csv", index=False)
    registry_sources = (sorted((V914 / "runs").glob("**/stageB_registry*.csv")) +
                        [GLOBAL_REGISTRY] +
                        [V914 / "runtime_inputs/v914/local_fracture_manifold_256.csv",
                        V914 / "runtime_inputs/v914/canonical_four_class_registry.csv",
                        V914 / "runtime_inputs/v914/fracture_equivalence_registry.csv"])
    source_rows = pd.concat([pd.read_csv(p, dtype=str) for p in registry_sources], ignore_index=True, sort=False)
    source_rows = source_rows.drop_duplicates("candidate_id", keep="first")
    launch_registry = source_rows[source_rows.candidate_id.isin(selected)].copy()
    launch_registry.to_csv(out / "candidate_shortlist_launch_registry.csv", index=False)
    info = master.set_index("candidate_id")
    plot_curves(curves, info, selected, out / "top_candidates_da_dN_vs_deltaK")
    plot_curves(curves, info, [c for c in dbtt.candidate_id if c in set(curves.candidate_id)][:15], out / "top_DBTT_da_dN_vs_deltaK")
    plot_curves(curves, info, [c for c in peak.candidate_id if c in set(curves.candidate_id)][:12], out / "top_peak_da_dN_vs_deltaK")
    plot_curves(curves, info, list(abcd.candidate_id), out / "top_ABCD_da_dN_vs_deltaK_four_panel", panels=True)
    finalists = list(knee.candidate_id[:10])
    plot_curves(curves, info, finalists, out / "candidate_finalists_da_dN_vs_deltaK")
    fig, ax = plt.subplots(figsize=(9, 7))
    for letter, marker in zip(["A", "B", "C", "D", "SMOOTH_ARRHENIUS"], ["o", "s", "^", "D", "x"]):
        g = shortlist[shortlist.mechanism_letter == letter]
        ax.scatter(g.knee_quality, g.R_recovery, marker=marker, s=55, label=letter)
    ax.set(xlabel=r"knee quality $m_{low}/m_{knee}$", ylabel=r"high-$K$ recovery $m_{high}/m_{knee}$", xscale="log", yscale="log")
    ax.legend(frameon=False); savefig(fig, out / "candidate_knee_vs_highK_recovery")
    fig, ax = plt.subplots(figsize=(11, 7)); q = shortlist.sort_values("maximum_calculated_rate")
    ax.scatter(range(len(q)), q.maximum_calculated_rate, c=pd.factorize(q.mechanism_letter)[0], s=50)
    ax.set_yscale("log"); ax.set_ylabel(r"maximum calculated $da/dN$ (m cycle$^{-1}$)"); ax.set_xticks(range(len(q)), q.candidate_id.str.replace("v914_endurance_knee_", "", regex=False), rotation=90, fontsize=7)
    savefig(fig, out / "candidate_high_rate_capacity")
    new_cases = int(sum(1 for p in args.extra_fatigue_root for _ in p.rglob("fatigue_result.json")))
    summary = {"schema": "v914_endurance_knee_rerank_DBTT_highK_v1", "historic_temperature_candidates": len(temp),
               "mechanism_probe_candidates": len(mech), "fatigue_candidates_reused": curves.candidate_id.nunique(),
               "shortlist_size": len(shortlist), "new_highK_1d_cases": new_cases, "new_2d_runs": 0,
               "limitations": ["historic v9.13 rows without saved 300 K K50 remain explicit missing values",
                               "v9.14 local-family names are not treated as measured temperature classes",
                               "high-rate ceiling is a hazard/event-length estimate before energy truncation"]}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
