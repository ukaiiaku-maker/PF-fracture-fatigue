#!/usr/bin/env python3
"""Audit and plot the prospective v9.14 eight-model selection screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "v914_model_selection_screen_analysis_v1"
KNOWN_STATUSES = {
    "growth_target_reached",
    "cycle_horizon_reached",
    "work_budget_exhausted",
}
REQUIRED_HEAD = "2e7c9752f599a7ee88bed7865bbb3ab726ea533e"
LAUNCHER_HEAD = "a1216242df52f5570417c64683f179205a62e063"
VALID_PRODUCTION_HEADS = {LAUNCHER_HEAD, REQUIRED_HEAD}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _expected_dir(root: Path, case: dict[str, Any]) -> Path:
    family = str(case["family"]).replace("-", "_")
    suffix = str(case["candidate"]).split("_")[-1]
    return root / f"{family}_{suffix}" / (
        f"R{_slug(float(case['R']))}_Kmax{_slug(float(case['Kmax_MPa_sqrt_m']))}"
    )


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite: {value!r}")
    return number


def _terminal_class(status: str, events: int, extension: float) -> tuple[str, str]:
    if status == "growth_target_reached":
        return "completed_growth", "none"
    if status == "work_budget_exhausted":
        return "numerical_partial", "work_budget_exhausted"
    if events == 0 and extension == 0.0:
        return "physical_censor", "no_event_by_cycle_horizon"
    return "physical_horizon_with_growth", "cycle_horizon_with_nonzero_growth"


def audit_screen(
    root: Path,
    *,
    require_complete_ab: bool = True,
    required_head: str = REQUIRED_HEAD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return audited terminal rows and missing manifest cases, failing closed."""
    manifest_path = root / "screen_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if len(cases) != 64:
        raise ValueError(f"expected 64 manifest cases, found {len(cases)}")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    expected_results: set[Path] = set()
    for index, case in enumerate(cases):
        run_dir = _expected_dir(root, case)
        result_path = run_dir / "result.json"
        expected_results.add(result_path.resolve())
        if not result_path.is_file():
            missing.append({"manifest_index": index, **case, "run_directory": str(run_dir.resolve())})
            continue
        contract_path = run_dir / "run_contract.json"
        if not contract_path.is_file():
            raise ValueError(f"terminal result lacks run contract: {result_path}")
        result = json.loads(result_path.read_text())
        contract = json.loads(contract_path.read_text())
        status = str(result.get("status"))
        if status not in KNOWN_STATUSES:
            raise ValueError(f"unknown terminal status {status!r}: {result_path}")
        candidate = str(contract.get("candidate"))
        if candidate != case["candidate"]:
            raise ValueError(f"candidate mismatch at {result_path}: {candidate} != {case['candidate']}")
        R = _finite(contract.get("R"), "R")
        delta_k = _finite(contract.get("deltaK_MPa_sqrt_m"), "DeltaK")
        kmax = delta_k / (1.0 - R)
        if not math.isclose(R, float(case["R"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"R mismatch at {result_path}")
        if not math.isclose(delta_k, float(case["deltaK_MPa_sqrt_m"]), rel_tol=1e-12):
            raise ValueError(f"DeltaK mismatch at {result_path}")
        if not math.isclose(kmax, float(case["Kmax_MPa_sqrt_m"]), rel_tol=1e-12):
            raise ValueError(f"Kmax mismatch at {result_path}")
        restart_count = int(result.get("restart_count", -1))
        if restart_count != 0:
            raise ValueError(f"restart_count={restart_count} is not production-qualified: {result_path}")
        repository_head = contract.get("repository_head")
        if repository_head not in VALID_PRODUCTION_HEADS:
            raise ValueError(f"unqualified repository HEAD at {result_path}: {repository_head}")
        if case["family"] in {"A", "B"} and repository_head != required_head:
            raise ValueError(f"A/B repository HEAD mismatch at {result_path}")
        if contract.get("repository_clean") is not True:
            raise ValueError(f"run contract was not clean: {result_path}")
        frequency = _finite(contract.get("frequency_Hz"), "frequency_Hz")
        final_time = _finite(result.get("final_time_s"), "final_time_s")
        exact_cycles = final_time * frequency
        if exact_cycles <= 0.0:
            raise ValueError(f"nonpositive exact cycle coordinate at {result_path}")
        extension = _finite(result.get("cumulative_extension_m"), "cumulative_extension_m")
        if extension < 0.0:
            raise ValueError(f"negative extension at {result_path}")
        events = int(result.get("event_count", len(result.get("events", []))))
        if events != len(result.get("events", [])):
            raise ValueError(f"event count mismatch at {result_path}")
        if status == "growth_target_reached" and events:
            event_cycles = _finite(result["events"][-1]["cumulative_cycles"], "final event cycles")
            if not math.isclose(exact_cycles, event_cycles, rel_tol=2e-10, abs_tol=2e-10):
                raise ValueError(
                    f"exact cycles {exact_cycles:.17g} disagree with final event {event_cycles:.17g}: {result_path}"
                )
        identity = (candidate, R, kmax)
        if identity in seen:
            raise ValueError(f"duplicate case identity: {identity}")
        seen.add(identity)
        terminal_class, censor_type = _terminal_class(status, events, extension)
        observed = extension / exact_cycles if extension > 0.0 else 0.0
        rows.append({
            "schema": SCHEMA,
            "manifest_index": index,
            "family": case["family"],
            "candidate": candidate,
            "configuration_label": contract.get("configuration_label"),
            "R": R,
            "Kmax_MPa_sqrt_m": kmax,
            "Kmin_MPa_sqrt_m": R * kmax,
            "deltaK_MPa_sqrt_m": delta_k,
            "status": status,
            "terminal_class": terminal_class,
            "censor_type": censor_type,
            "frequency_Hz": frequency,
            "temperature_K": _finite(contract.get("temperature_K"), "temperature_K"),
            "seed": int(contract.get("seed")),
            "final_time_s": final_time,
            "exact_cycles": exact_cycles,
            "completed_physical_cycles": int(result.get("completed_physical_cycles", 0)),
            "event_count": events,
            "cumulative_extension_m": extension,
            "cumulative_extension_um": extension * 1e6,
            "da_dN_observed_m_per_cycle": observed,
            "da_dN_observed_um_per_cycle": observed * 1e6,
            "physical_censor_bound_m_per_cycle": math.nan,
            "censor_bound_basis": "not_derived",
            "plot_rate_m_per_cycle": observed if observed > 0.0 else math.nan,
            "plot_rate_basis": "observed_extension_over_exact_cycles" if observed > 0.0 else "pending_visualization_ordinate",
            "restart_count": restart_count,
            "total_cycle_map_evaluations": int(result.get("total_cycle_map_evaluations", 0)),
            "accepted_cycle_map_evaluations": int(result.get("accepted_cycle_map_evaluations", 0)),
            "accepted_block_count": int(result.get("accepted_block_count", 0)),
            "rejected_block_count": int(result.get("rejected_block_count", 0)),
            "dmd_map_refresh_count": int(result.get("dmd_map_refresh_count", 0)),
            "dmd_map_reuse_count": int(result.get("dmd_map_reuse_count", 0)),
            "dmd_map_invalidation_count": int(result.get("dmd_map_invalidation_count", 0)),
            "fallback_exact_cycles": int(result.get("fallback_exact_cycles", 0)),
            "maximum_accepted_stride": int(result.get("maximum_accepted_stride", 0)),
            "event_guard_halvings": int(result.get("event_guard_halvings", 0)),
            "run_directory": str(run_dir.resolve()),
            "result_path": str(result_path.resolve()),
            "result_sha256": _sha256(result_path),
            "run_contract_path": str(contract_path.resolve()),
            "run_contract_sha256": _sha256(contract_path),
            "repository_branch": contract.get("repository_branch"),
            "repository_head": contract.get("repository_head"),
            "repository_clean": contract.get("repository_clean"),
            "registry_sha256": contract.get("registry_sha256"),
            "physics_sha256": contract.get("physics_sha256"),
            "engine_id": contract.get("engine_id"),
            "state_model_id": contract.get("state_model_id"),
        })

    unexpected = sorted(
        path.resolve() for path in root.rglob("result.json") if path.resolve() not in expected_results
    )
    if unexpected:
        raise ValueError(f"unexpected result files outside manifest paths: {unexpected}")
    ab_missing = [row for row in missing if row["family"] in {"A", "B"}]
    if require_complete_ab and ab_missing:
        detail = ", ".join(str(row["manifest_index"]) for row in ab_missing)
        raise ValueError(f"A/B screen incomplete; missing manifest indices: {detail}")

    positive = [float(row["da_dN_observed_m_per_cycle"]) for row in rows if row["da_dN_observed_m_per_cycle"] > 0.0]
    floor = min(positive) / 10.0 if positive else 1e-20
    for row in rows:
        if row["da_dN_observed_m_per_cycle"] == 0.0:
            row["plot_rate_m_per_cycle"] = floor
            row["plot_rate_basis"] = "visualization_only_one_decade_below_smallest_positive_screen_rate"
    family_order = ["Peak-T", "DBTT", "weak-T", "ceramic-like", "A", "B", "C", "D"]
    for row in rows:
        family_index = family_order.index(str(row["family"])) if str(row["family"]) in family_order else 3.5
        offset = (family_index - 3.5) * 0.035
        row["plot_deltaK_MPa_sqrt_m"] = float(row["deltaK_MPa_sqrt_m"]) + offset
        row["plot_deltaK_basis"] = "visualization_only_family_dodge_0p035_MPa_sqrt_m"
    rows.sort(key=lambda row: int(row["manifest_index"]))
    return rows, missing


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def summarize_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["candidate"], float(row["R"]))].append(row)
    output = []
    for (family, candidate, R), group in sorted(groups.items()):
        group.sort(key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
        growth = [row for row in group if row["terminal_class"] == "completed_growth"]
        censors = [row for row in group if row["censor_type"] == "no_event_by_cycle_horizon"]
        partials = [row for row in group if row["terminal_class"] == "numerical_partial"]
        rates = [float(row["da_dN_observed_m_per_cycle"]) for row in growth if row["da_dN_observed_m_per_cycle"] > 0]
        measured_pairs = [(float(row["deltaK_MPa_sqrt_m"]), float(row["da_dN_observed_m_per_cycle"])) for row in growth if row["da_dN_observed_m_per_cycle"] > 0]
        monotone = all(right[1] >= left[1] for left, right in zip(measured_pairs, measured_pairs[1:]))
        highest_censor = max((float(row["deltaK_MPa_sqrt_m"]) for row in censors), default=math.nan)
        lowest_growth = min((float(row["deltaK_MPa_sqrt_m"]) for row in growth), default=math.nan)
        bracketed = math.isfinite(highest_censor) and math.isfinite(lowest_growth) and highest_censor < lowest_growth
        if partials:
            interpretation = "numerically_unresolved"
        elif growth and censors:
            interpretation = "censor_to_growth_transition"
        elif growth:
            interpretation = "growth_at_all_terminal_loads"
        else:
            interpretation = "all_terminal_points_censored"
        output.append({
            "family": family,
            "candidate": candidate,
            "R": R,
            "terminal_points": len(group),
            "completed_growth_points": len(growth),
            "zero_event_cycle_horizon_censors": len(censors),
            "work_budget_numerical_partials": len(partials),
            "minimum_deltaK_MPa_sqrt_m": min(float(row["deltaK_MPa_sqrt_m"]) for row in group),
            "maximum_deltaK_MPa_sqrt_m": max(float(row["deltaK_MPa_sqrt_m"]) for row in group),
            "lowest_deltaK_with_measured_growth": lowest_growth,
            "highest_deltaK_with_zero_event_censor": highest_censor,
            "transition_bracketed": bracketed,
            "transition_interval_deltaK": f"({highest_censor:g}, {lowest_growth:g}]" if bracketed else "",
            "minimum_completed_growth_da_dN_m_per_cycle": min(rates, default=math.nan),
            "maximum_completed_growth_da_dN_m_per_cycle": max(rates, default=math.nan),
            "completed_growth_monotone_non_decreasing": monotone if len(measured_pairs) >= 2 else "insufficient_points",
            "total_cycle_map_evaluations": sum(int(row["total_cycle_map_evaluations"]) for row in group),
            "interpretation": interpretation,
        })
    return output


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    families = ["Peak-T", "DBTT", "weak-T", "ceramic-like", "A", "B", "C", "D"]
    palette = dict(zip(families, plt.get_cmap("tab10").colors[:8]))

    def panel(ax: Any, subset: list[dict[str, Any]], title: str) -> None:
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            by_family[str(row["family"])].append(row)
        for family in families:
            group = sorted(by_family.get(family, []), key=lambda row: float(row["deltaK_MPa_sqrt_m"]))
            color = palette[family]
            completed = [row for row in group if row["terminal_class"] == "completed_growth"]
            # Connect only adjacent completed manifest anchors; never bridge a censor/partial.
            for left, right in zip(group, group[1:]):
                if left["terminal_class"] == right["terminal_class"] == "completed_growth":
                    ax.plot(
                    [left["plot_deltaK_MPa_sqrt_m"], right["plot_deltaK_MPa_sqrt_m"]],
                        [left["plot_rate_m_per_cycle"], right["plot_rate_m_per_cycle"]],
                        color=color, linewidth=1.2, alpha=0.75,
                    )
            if completed:
                ax.scatter(
                    [row["plot_deltaK_MPa_sqrt_m"] for row in completed],
                    [row["plot_rate_m_per_cycle"] for row in completed],
                    color=color, marker="o", s=42, zorder=4,
                )
            censors = [row for row in group if row["censor_type"] == "no_event_by_cycle_horizon"]
            if censors:
                ax.scatter(
                    [row["plot_deltaK_MPa_sqrt_m"] for row in censors],
                    [row["plot_rate_m_per_cycle"] for row in censors],
                    edgecolor=color, facecolor="none", marker="v", s=55, linewidth=1.4, zorder=4,
                )
            partials = [row for row in group if row["terminal_class"] == "numerical_partial"]
            if partials:
                ax.scatter(
                    [row["plot_deltaK_MPa_sqrt_m"] for row in partials],
                    [row["plot_rate_m_per_cycle"] for row in partials],
                    edgecolor=color, facecolor="none", marker="s", s=48, linewidth=1.4, zorder=4,
                )
            horizon_growth = [row for row in group if row["terminal_class"] == "physical_horizon_with_growth"]
            if horizon_growth:
                ax.scatter(
                    [row["plot_deltaK_MPa_sqrt_m"] for row in horizon_growth],
                    [row["plot_rate_m_per_cycle"] for row in horizon_growth],
                    edgecolor=color, facecolor="none", marker="D", s=48, linewidth=1.4, zorder=4,
                )
        ax.set_yscale("log")
        ax.set_xlabel(r"Imposed $\Delta K$ (MPa $\sqrt{m}$)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)

    family_handles = [Line2D([0], [0], color=palette[f], marker="o", linewidth=1.2, label=f) for f in families]
    status_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="none", label="completed growth"),
        Line2D([0], [0], color="black", marker="v", markerfacecolor="none", linestyle="none", label="zero-event horizon censor*"),
        Line2D([0], [0], color="black", marker="s", markerfacecolor="none", linestyle="none", label="work-budget partial"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    for ax, R in zip(axes, (0.1, -0.5)):
        panel(ax, [row for row in rows if math.isclose(float(row["R"]), R)], f"R = {R:+g}")
    axes[0].set_ylabel(r"Observed $da/dN$ (m cycle$^{-1}$)")
    axes[1].legend(handles=family_handles + status_handles, fontsize=8, ncol=2, loc="best")
    fig.suptitle("v9.14 model-selection screen: every valid terminal trajectory")
    fig.text(0.5, 0.025, "* Censor ordinate and ±0.123 MPa√m maximum family dodge are visualization-only; physical values remain unchanged in the CSV.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.86, wspace=0.06)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"v914_model_selection_da_dN_vs_DeltaK_by_R.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    markers = {0.1: "o", -0.5: "^"}
    status_markers = {"completed_growth": "o", "physical_censor": "v", "numerical_partial": "s", "physical_horizon_with_growth": "D"}
    for family in families:
        for R in (0.1, -0.5):
            for terminal_class, marker in status_markers.items():
                group = [row for row in rows if row["family"] == family and row["terminal_class"] == terminal_class and math.isclose(float(row["R"]), R)]
                if group:
                    hollow = terminal_class in {"physical_censor", "numerical_partial", "physical_horizon_with_growth"}
                    ax.scatter(
                        [row["plot_deltaK_MPa_sqrt_m"] for row in group],
                        [row["plot_rate_m_per_cycle"] for row in group],
                        edgecolor=palette[family], facecolor="none" if hollow else palette[family],
                        marker=marker, s=42 if R == 0.1 else 68, linewidth=1.3,
                        alpha=0.95 if R == 0.1 else 0.55,
                    )
    ax.set_yscale("log")
    ax.set(xlabel=r"Imposed $\Delta K$ (MPa $\sqrt{m}$)", ylabel=r"Observed $da/dN$ (m cycle$^{-1}$)", title="All terminal screen data (color=model, marker=R)")
    ax.grid(True, which="both", alpha=0.22)
    handles = family_handles + status_handles + [
        Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=5, label="R=+0.1 (smaller/opaque)"),
        Line2D([0], [0], color="black", marker="o", linestyle="none", markersize=8, alpha=0.55, label="R=-0.5 (larger/faded)"),
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"v914_model_selection_da_dN_vs_DeltaK_all.{suffix}", dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    statuses = Counter(str(row["status"]) for row in rows)
    families = Counter(str(row["family"]) for row in rows)
    lines = [
        "# v9.14 Model-Selection Screen",
        "",
        f"- Production repository HEAD: `{REQUIRED_HEAD}`",
        f"- Audited terminal rows: {len(rows)}",
        f"- Status counts: {dict(sorted(statuses.items()))}",
        f"- Family counts: {dict(sorted(families.items()))}",
        f"- Missing manifest cases: {len(missing)}",
        "",
        "## Rate and censor policy",
        "",
        "`exact_cycles = final_time_s * frequency_Hz` and `da/dN = cumulative_extension_m / exact_cycles`.",
        "Zero-event horizon censors retain an observed rate of exactly zero. Their log-plot ordinate is visualization-only and is stored separately in `plot_rate_m_per_cycle`.",
        "Work-budget exhaustion is a numerical partial, not an endurance censor; its accumulated physical rate is retained but excluded from completed-growth interpretation.",
        "Interrupted attempts without terminal `result.json` files are excluded and never spliced into fresh runs.",
        "",
        "## Candidate/R summary",
        "",
        "| Family | R | Growth | Censors | Numerical partials | Transition | Interpretation |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['family']} | {float(row['R']):+g} | {row['completed_growth_points']} | "
            f"{row['zero_event_cycle_horizon_censors']} | {row['work_budget_numerical_partials']} | "
            f"{row['transition_interval_deltaK'] or '—'} | {row['interpretation']} |"
        )
    if missing:
        lines += ["", "## Missing manifest cases", ""]
        lines += [f"- index {row['manifest_index']}: {row['family']} R={row['R']:+g}, Kmax={row['Kmax_MPa_sqrt_m']:g}" for row in missing]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete-ab", action="store_true")
    args = parser.parse_args()
    root = args.screen_root.resolve()
    output = (args.output_dir or (root / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = audit_screen(root, require_complete_ab=not args.allow_incomplete_ab)
    summaries = summarize_models(rows)
    _write_csv(output / "model_screen.csv", rows)
    _write_csv(output / "model_screen_summary.csv", summaries)
    _write_csv(output / "missing_manifest_cases.csv", missing)
    write_plots(rows, output)
    write_report(output / "V914_MODEL_SELECTION_SCREEN_SUMMARY.md", rows, summaries, missing)
    audit = {
        "schema": SCHEMA,
        "terminal_rows": len(rows),
        "missing_manifest_cases": len(missing),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "all_ab_terminal": not any(row["family"] in {"A", "B"} for row in missing),
    }
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
