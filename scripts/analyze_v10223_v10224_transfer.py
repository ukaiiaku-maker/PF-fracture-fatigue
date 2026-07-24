#!/usr/bin/env python3
"""Summarize v10.2.23 peak and v10.2.24 upper-shelf 1-D/2-D transfer.

The analyzer can read the two run directories directly or the compact ZIP
created by the campaign handoff command. It uses only the Python standard
library for the numerical analysis; matplotlib plots are optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import zipfile


TEMPERATURE_RE = re.compile(r"K50_T(?P<T>[0-9]+)K_MPa_sqrt_m$")
SUMMARY_NAME = "v10_2_22_dbtt_50um_screen_summary.csv"


def number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def finite(value: object) -> bool:
    return math.isfinite(number(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.median(clean) if clean else float("nan")


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else float("nan")


def rmse(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return math.sqrt(sum(value * value for value in clean) / len(clean)) if clean else float("nan")


def fmt(value: object, digits: int = 3) -> str:
    x = number(value)
    return f"{x:.{digits}f}" if math.isfinite(x) else "NA"


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def locate_family_root(base: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    for summary in base.rglob(SUMMARY_NAME):
        lower = str(summary).lower()
        if any(name.lower() in lower for name in names):
            return summary.parent
    raise FileNotFoundError(f"could not locate family root for {names} below {base}")


def locate_summary(root: Path) -> Path:
    direct = root / SUMMARY_NAME
    if direct.is_file():
        return direct
    matches = list(root.rglob(SUMMARY_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {SUMMARY_NAME} below {root}; found {len(matches)}"
        )
    return matches[0]


def read_reference(path: Path, rank_field: str) -> tuple[list[dict[str, str]], list[int]]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"empty 1-D reference: {path}")
    temperatures = []
    for name in rows[0]:
        match = TEMPERATURE_RE.match(name)
        if match:
            temperatures.append(int(match.group("T")))
    temperatures = sorted(set(temperatures))
    if not temperatures:
        raise RuntimeError(f"no K50 temperature columns in {path}")
    for row in rows:
        if not row.get("candidate_id"):
            raise RuntimeError(f"reference row missing candidate_id in {path}")
        if not finite(row.get(rank_field)):
            raise RuntimeError(f"reference row missing {rank_field}: {row.get('candidate_id')}")
    return rows, temperatures


def summary_lookup(path: Path) -> tuple[dict[tuple[str, int], dict[str, str]], list[str]]:
    rows = read_csv(path)
    if not rows:
        return {}, ["summary CSV is empty"]
    required = {"candidate_id", "temperature_K", "K_50um_MPa_sqrt_m"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(f"summary missing columns {missing}: {path}")
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    errors: list[str] = []
    for row in rows:
        candidate = str(row.get("candidate_id", ""))
        temperature_value = number(row.get("temperature_K"))
        if not candidate or not math.isfinite(temperature_value):
            errors.append(f"invalid summary identity row: {row}")
            continue
        key = (candidate, int(round(temperature_value)))
        if key in lookup:
            errors.append(f"duplicate summary key: {key}")
            continue
        lookup[key] = row
    return lookup, errors


def ordered_rehardening(points: list[tuple[int, float]]) -> tuple[float, float, bool]:
    """Return post-transition drop, later rebound, and a rehardening flag.

    The transition reference is the largest K between 850 and 1100 K. A true
    rehardening response requires both a subsequent drop and a later recovery
    of at least 3 MPa sqrt(m). A monotonic shelf is therefore not rehardening.
    """
    transition = [(t, k) for t, k in points if 850 <= t <= 1100 and math.isfinite(k)]
    if not transition:
        return float("nan"), float("nan"), False
    transition_t, transition_k = max(transition, key=lambda pair: pair[1])
    post = [(t, k) for t, k in points if t > transition_t and math.isfinite(k)]
    if len(post) < 2:
        return 0.0, 0.0, False
    minimum_index = min(range(len(post)), key=lambda index: post[index][1])
    minimum_value = post[minimum_index][1]
    later = [value for _, value in post[minimum_index + 1 :]]
    rebound = max(later) - minimum_value if later else 0.0
    drop = transition_k - minimum_value
    return drop, rebound, drop >= 3.0 and rebound >= 3.0


def curve_metrics(points: list[tuple[int, float]]) -> dict[str, object]:
    points = sorted((int(t), float(k)) for t, k in points if math.isfinite(k))
    if not points:
        return {
            "peak_temperature_K": float("nan"),
            "peak_K50_MPa_sqrt_m": float("nan"),
            "low_temperature_baseline_MPa_sqrt_m": float("nan"),
            "high_temperature_shelf_MPa_sqrt_m": float("nan"),
            "directional_dbtt_gain_MPa_sqrt_m": float("nan"),
            "peak_prominence_MPa_sqrt_m": float("nan"),
            "shelf_persistence": float("nan"),
            "post_transition_drop_MPa_sqrt_m": float("nan"),
            "late_rebound_MPa_sqrt_m": float("nan"),
            "rehardening": False,
            "peak_like": False,
            "upper_shelf_gate": False,
            "classic_upper_shelf": False,
        }
    peak_t, peak_k = max(points, key=lambda pair: pair[1])
    low = points[0][1]
    high_points = [k for t, k in points if t >= 1200]
    if len(high_points) < 3:
        high_points = [k for _, k in points[-3:]]
    shelf = median(high_points)
    gain = shelf - low
    prominence = max(0.0, min(peak_k - low, peak_k - shelf))
    persistence = shelf / peak_k if peak_k > 0.0 else float("nan")
    drop, rebound, rehardening = ordered_rehardening(points)
    peak_like = 850 <= peak_t <= 1100 and prominence >= 5.0
    upper_shelf_gate = gain >= 5.0 and prominence < 5.0 and persistence >= 0.70
    classic_upper_shelf = upper_shelf_gate and not rehardening
    return {
        "peak_temperature_K": peak_t,
        "peak_K50_MPa_sqrt_m": peak_k,
        "low_temperature_baseline_MPa_sqrt_m": low,
        "high_temperature_shelf_MPa_sqrt_m": shelf,
        "directional_dbtt_gain_MPa_sqrt_m": gain,
        "peak_prominence_MPa_sqrt_m": prominence,
        "shelf_persistence": persistence,
        "post_transition_drop_MPa_sqrt_m": drop,
        "late_rebound_MPa_sqrt_m": rebound,
        "rehardening": rehardening,
        "peak_like": peak_like,
        "upper_shelf_gate": upper_shelf_gate,
        "classic_upper_shelf": classic_upper_shelf,
    }


def inspect_status_files(root: Path) -> dict[str, object]:
    statuses = list(root.rglob("stage3_case_status.json"))
    complete = 0
    invalid = 0
    for path in statuses:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            invalid += 1
            continue
        if payload.get("complete") is True:
            complete += 1
    failed = [str(path.relative_to(root).parent) for path in root.rglob("RUN_FAILED")]
    disk_full_hits = 0
    for name in ("driver.log", "driver.resume.log"):
        for path in root.rglob(name):
            try:
                disk_full_hits += path.read_text(errors="replace").count("No space left on device")
            except OSError:
                pass
    return {
        "status_files": len(statuses),
        "complete_status_files": complete,
        "invalid_status_files": invalid,
        "run_failed_markers": len(failed),
        "failed_case_directories": failed,
        "disk_full_log_hits": disk_full_hits,
    }


def analyze_family(
    *,
    family: str,
    root: Path,
    reference_path: Path,
    rank_field: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    reference_rows, temperatures = read_reference(reference_path, rank_field)
    summary_path = locate_summary(root)
    lookup, summary_errors = summary_lookup(summary_path)
    expected = {
        (str(row["candidate_id"]), temperature)
        for row in reference_rows
        for temperature in temperatures
    }
    observed = set(lookup)
    missing_keys = sorted(expected - observed)
    unexpected_keys = sorted(observed - expected)

    reference_by_candidate = {str(row["candidate_id"]): row for row in reference_rows}
    temperature_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for candidate, reference in reference_by_candidate.items():
        points_1d: list[tuple[int, float]] = []
        points_2d: list[tuple[int, float]] = []
        errors: list[float] = []
        absolute_errors: list[float] = []
        for temperature in temperatures:
            one_d = number(reference.get(f"K50_T{temperature}K_MPa_sqrt_m"))
            summary = lookup.get((candidate, temperature))
            two_d = number(summary.get("K_50um_MPa_sqrt_m")) if summary else float("nan")
            if math.isfinite(one_d):
                points_1d.append((temperature, one_d))
            if math.isfinite(two_d):
                points_2d.append((temperature, two_d))
            delta = two_d - one_d if math.isfinite(one_d) and math.isfinite(two_d) else float("nan")
            if math.isfinite(delta):
                errors.append(delta)
                absolute_errors.append(abs(delta))
            temperature_rows.append(
                {
                    "family": family,
                    "candidate_id": candidate,
                    "rank": int(round(number(reference[rank_field]))),
                    "temperature_K": temperature,
                    "K50_1d_MPa_sqrt_m": one_d,
                    "K50_2d_MPa_sqrt_m": two_d,
                    "delta_2d_minus_1d_MPa_sqrt_m": delta,
                    "absolute_error_MPa_sqrt_m": abs(delta) if math.isfinite(delta) else float("nan"),
                }
            )

        metrics_1d = curve_metrics(points_1d)
        metrics_2d = curve_metrics(points_2d)
        intended_retained = (
            bool(metrics_2d["peak_like"])
            if family == "peak"
            else bool(metrics_2d["classic_upper_shelf"])
        )
        candidate_rows.append(
            {
                "family": family,
                "candidate_id": candidate,
                "rank": int(round(number(reference[rank_field]))),
                "matched_temperatures": len(errors),
                "bias_2d_minus_1d_MPa_sqrt_m": mean(errors),
                "MAE_MPa_sqrt_m": mean(absolute_errors),
                "RMSE_MPa_sqrt_m": rmse(errors),
                "maximum_absolute_error_MPa_sqrt_m": max(absolute_errors) if absolute_errors else float("nan"),
                "peak_temperature_1d_K": metrics_1d["peak_temperature_K"],
                "peak_temperature_2d_K": metrics_2d["peak_temperature_K"],
                "peak_temperature_shift_2d_minus_1d_K": (
                    number(metrics_2d["peak_temperature_K"]) - number(metrics_1d["peak_temperature_K"])
                    if finite(metrics_2d["peak_temperature_K"]) and finite(metrics_1d["peak_temperature_K"])
                    else float("nan")
                ),
                "peak_prominence_1d_MPa_sqrt_m": metrics_1d["peak_prominence_MPa_sqrt_m"],
                "peak_prominence_2d_MPa_sqrt_m": metrics_2d["peak_prominence_MPa_sqrt_m"],
                "directional_dbtt_gain_1d_MPa_sqrt_m": metrics_1d["directional_dbtt_gain_MPa_sqrt_m"],
                "directional_dbtt_gain_2d_MPa_sqrt_m": metrics_2d["directional_dbtt_gain_MPa_sqrt_m"],
                "upper_shelf_1d_MPa_sqrt_m": metrics_1d["high_temperature_shelf_MPa_sqrt_m"],
                "upper_shelf_2d_MPa_sqrt_m": metrics_2d["high_temperature_shelf_MPa_sqrt_m"],
                "shelf_persistence_1d": metrics_1d["shelf_persistence"],
                "shelf_persistence_2d": metrics_2d["shelf_persistence"],
                "post_transition_drop_2d_MPa_sqrt_m": metrics_2d["post_transition_drop_MPa_sqrt_m"],
                "late_rebound_2d_MPa_sqrt_m": metrics_2d["late_rebound_MPa_sqrt_m"],
                "rehardening_2d": bool(metrics_2d["rehardening"]),
                "peak_like_2d": bool(metrics_2d["peak_like"]),
                "upper_shelf_gate_2d": bool(metrics_2d["upper_shelf_gate"]),
                "classic_upper_shelf_2d": bool(metrics_2d["classic_upper_shelf"]),
                "intended_topology_retained": intended_retained,
            }
        )

    integrity = inspect_status_files(root)
    all_errors = [number(row["delta_2d_minus_1d_MPa_sqrt_m"]) for row in temperature_rows]
    all_absolute = [abs(value) for value in all_errors if math.isfinite(value)]
    summary = {
        "family": family,
        "root": str(root),
        "reference": str(reference_path),
        "expected_cases": len(expected),
        "summary_rows": len(observed),
        "missing_cases": len(missing_keys),
        "missing_case_keys": [f"{candidate}:T{temperature}K" for candidate, temperature in missing_keys],
        "unexpected_cases": len(unexpected_keys),
        "summary_schema_errors": summary_errors,
        "global_bias_2d_minus_1d_MPa_sqrt_m": mean(all_errors),
        "global_MAE_MPa_sqrt_m": mean(all_absolute),
        "global_RMSE_MPa_sqrt_m": rmse(all_errors),
        "intended_topology_retained_count": sum(bool(row["intended_topology_retained"]) for row in candidate_rows),
        "peak_like_2d_count": sum(bool(row["peak_like_2d"]) for row in candidate_rows),
        "upper_shelf_gate_2d_count": sum(bool(row["upper_shelf_gate_2d"]) for row in candidate_rows),
        "classic_upper_shelf_2d_count": sum(bool(row["classic_upper_shelf_2d"]) for row in candidate_rows),
        "rehardening_2d_count": sum(bool(row["rehardening_2d"]) for row in candidate_rows),
        **integrity,
    }
    return summary, candidate_rows, temperature_rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def create_report(
    summaries: list[dict[str, object]],
    candidates: list[dict[str, object]],
) -> str:
    lines = [
        "# v10.2.23/v10.2.24 1-D to 2-D transfer analysis",
        "",
        "## Classification used by this analyzer",
        "",
        "- `peak_like_2d`: the maximum occurs from 850–1100 K and the two-sided prominence relative to the 700 K baseline and median 1200–1400 K response is at least 5 MPa√m.",
        "- `upper_shelf_gate_2d`: median 1200–1400 K toughness exceeds the 700 K value by at least 5 MPa√m, peak prominence is below 5 MPa√m, and shelf/maximum persistence is at least 0.70.",
        "- `classic_upper_shelf_2d`: the upper-shelf gate passes and there is no ordered post-transition drop and rebound of at least 3 MPa√m.",
        "- `rehardening_2d`: after the strongest 850–1100 K transition response, the curve drops by at least 3 MPa√m and subsequently rebounds by at least 3 MPa√m.",
        "",
        "## Run integrity and aggregate transfer",
        "",
    ]
    integrity_rows = []
    for summary in summaries:
        integrity_rows.append(
            [
                str(summary["family"]),
                f"{summary['summary_rows']}/{summary['expected_cases']}",
                str(summary["missing_cases"]),
                str(summary["run_failed_markers"]),
                fmt(summary["global_bias_2d_minus_1d_MPa_sqrt_m"]),
                fmt(summary["global_MAE_MPa_sqrt_m"]),
                fmt(summary["global_RMSE_MPa_sqrt_m"]),
                str(summary["intended_topology_retained_count"]),
                str(summary["rehardening_2d_count"]),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Family",
                "2-D rows",
                "Missing",
                "RUN_FAILED",
                "Bias",
                "MAE",
                "RMSE",
                "Topology retained",
                "Rehardening",
            ],
            integrity_rows,
        )
    )

    for family in ("peak", "upper_shelf"):
        local = [row for row in candidates if row["family"] == family]
        if family == "peak":
            local.sort(key=lambda row: number(row["peak_prominence_2d_MPa_sqrt_m"]), reverse=True)
        else:
            local.sort(
                key=lambda row: (
                    bool(row["classic_upper_shelf_2d"]),
                    number(row["directional_dbtt_gain_2d_MPa_sqrt_m"]),
                    number(row["shelf_persistence_2d"]),
                ),
                reverse=True,
            )
        lines.extend(["", f"## {family.replace('_', ' ').title()} candidates", ""])
        candidate_table = []
        for row in local:
            candidate_table.append(
                [
                    str(row["rank"]),
                    str(row["candidate_id"]),
                    str(row["matched_temperatures"]),
                    fmt(row["MAE_MPa_sqrt_m"]),
                    fmt(row["peak_temperature_2d_K"], 0),
                    fmt(row["peak_prominence_2d_MPa_sqrt_m"]),
                    fmt(row["directional_dbtt_gain_2d_MPa_sqrt_m"]),
                    fmt(row["shelf_persistence_2d"]),
                    "yes" if row["intended_topology_retained"] else "no",
                    "yes" if row["rehardening_2d"] else "no",
                ]
            )
        lines.append(
            markdown_table(
                [
                    "Rank",
                    "Candidate",
                    "Matched T",
                    "MAE",
                    "2-D peak T",
                    "2-D prominence",
                    "2-D shelf gain",
                    "2-D persistence",
                    "Retained",
                    "Rehardening",
                ],
                candidate_table,
            )
        )

    missing = [
        (str(summary["family"]), key)
        for summary in summaries
        for key in summary["missing_case_keys"]
    ]
    if missing:
        lines.extend(["", "## Missing cases", ""])
        lines.extend(f"- {family}: `{key}`" for family, key in missing)
    lines.append("")
    return "\n".join(lines)


def optional_plots(out_dir: Path, temperatures: list[dict[str, object]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written = []
    for family in ("peak", "upper_shelf"):
        local = [row for row in temperatures if row["family"] == family and finite(row["K50_2d_MPa_sqrt_m"])]
        by_candidate: dict[str, list[dict[str, object]]] = {}
        for row in local:
            by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
        if not by_candidate:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        for candidate, rows in by_candidate.items():
            rows.sort(key=lambda row: int(row["temperature_K"]))
            ax.plot(
                [int(row["temperature_K"]) for row in rows],
                [number(row["K50_2d_MPa_sqrt_m"]) for row in rows],
                marker="o",
                label=candidate.rsplit("_", 1)[-1],
            )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"$K_{50\,\mu m}$ (MPa$\sqrt{m}$)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        name = f"{family}_2d_K50_temperature.png"
        fig.savefig(out_dir / name, dpi=240)
        plt.close(fig)
        written.append(name)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--compact-zip", type=Path)
    source.add_argument("--peak-root", type=Path)
    parser.add_argument("--shelf-root", type=Path)
    parser.add_argument(
        "--peak-reference",
        type=Path,
        default=Path("arrhenius_fracture/data/materials/v10_2_23_v913_top10_1d_reference.csv"),
    )
    parser.add_argument(
        "--shelf-reference",
        type=Path,
        default=Path("arrhenius_fracture/data/materials/v10_2_24_v913_top10_upper_shelf_1d_reference.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/v10223_v10224_transfer_analysis_v1"),
    )
    args = parser.parse_args()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.compact_zip is not None:
        temporary = tempfile.TemporaryDirectory(prefix="v10223_v10224_analysis_")
        extracted = Path(temporary.name)
        safe_extract(args.compact_zip, extracted)
        peak_root = locate_family_root(extracted, ("peak", "v10_2_23"))
        shelf_root = locate_family_root(extracted, ("upper_shelf", "shelf", "v10_2_24"))
    else:
        if args.shelf_root is None:
            parser.error("--shelf-root is required with --peak-root")
        peak_root = args.peak_root
        shelf_root = args.shelf_root

    args.out_dir.mkdir(parents=True, exist_ok=True)
    peak_summary, peak_candidates, peak_temperature = analyze_family(
        family="peak",
        root=peak_root,
        reference_path=args.peak_reference,
        rank_field="search_rank",
    )
    shelf_summary, shelf_candidates, shelf_temperature = analyze_family(
        family="upper_shelf",
        root=shelf_root,
        reference_path=args.shelf_reference,
        rank_field="shelf_rank",
    )
    summaries = [peak_summary, shelf_summary]
    candidates = peak_candidates + shelf_candidates
    temperatures = peak_temperature + shelf_temperature

    write_csv(args.out_dir / "candidate_transfer_metrics.csv", candidates)
    write_csv(args.out_dir / "temperature_transfer_points.csv", temperatures)
    report = create_report(summaries, candidates)
    (args.out_dir / "analysis_report.md").write_text(report + "\n")
    payload = {
        "schema": "v10.2.23_v10.2.24_transfer_analysis_v1",
        "families": summaries,
        "candidate_count": len(candidates),
        "temperature_point_count": len(temperatures),
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=lambda value: None) + "\n"
    )
    plots = optional_plots(args.out_dir, temperatures)

    bundle = args.out_dir / "v10223_v10224_transfer_analysis_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "analysis_report.md",
            "analysis_summary.json",
            "candidate_transfer_metrics.csv",
            "temperature_transfer_points.csv",
            *plots,
        ):
            path = args.out_dir / name
            if path.is_file():
                archive.write(path, name)

    print(report)
    print(f"\nANALYSIS DIRECTORY: {args.out_dir}")
    print(f"UPLOAD BUNDLE: {bundle}")
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
