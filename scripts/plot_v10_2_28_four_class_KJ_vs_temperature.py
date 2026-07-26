#!/usr/bin/env python3
"""Plot initial and long-propagation K/J versus temperature for v10.2.28.

The script reads the completed per-case ``steps_*K.csv`` files directly.  The
long-propagation response is a crack-extension-weighted average over the final
window of propagation, rather than an arithmetic average over accepted solver
steps or a two-checkpoint approximation.

For the default 1000 um campaign, ``--tail-length-um 200`` and
``--tail-fraction 0.20`` describe essentially the same final interval.  The
``maximum`` policy uses the larger of those two widths so that the requested
window remains at least 200 um and at least 20 percent of realized propagation.

The step files store ``KJ_Pa_sqrtm`` (the equivalent stress intensity obtained
from the directional J integral), not a separate J column.  J is reconstructed
as ``J = KJ**2 / Eprime``.  Eprime is derived from the mechanically locked cubic
elastic constants when they are isotropic (Zener ratio one), or may be supplied
explicitly with ``--Eprime-GPa``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_RE = re.compile(
    r"T(?P<T>\d+(?:\.\d+)?)K_th(?P<theta>[-+0-9.]+)_seed(?P<seed>\d+)$"
)

OPTION_ORDER = (
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0129902_persistent_sites",
    "v913_paper_ceramic01_0077080_persistent_sites",
)

OPTION_LABELS = {
    OPTION_ORDER[0]: "Peak",
    OPTION_ORDER[1]: "DBTT",
    OPTION_ORDER[2]: "Weak-T",
    OPTION_ORDER[3]: "Ceramic",
}

SUMMARY_SCHEMA = "v10.2.28_four_class_KJ_temperature_response_v1"
AUDIT_SCHEMA = "v10.2.28_four_class_KJ_temperature_plot_audit_v1"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _isotropic_moduli_from_cubic(
    C11_Pa: float,
    C12_Pa: float,
    C44_Pa: float,
    *,
    tolerance: float = 1.0e-9,
) -> tuple[float, float]:
    """Return isotropic E and nu from a Zener-one cubic stiffness tensor."""
    C11 = float(C11_Pa)
    C12 = float(C12_Pa)
    C44 = float(C44_Pa)
    values = (C11, C12, C44)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("cubic elastic constants must be finite and positive")
    residual = abs(C11 - (C12 + 2.0 * C44))
    scale = max(abs(C11), abs(C12) + 2.0 * abs(C44), 1.0)
    if residual > tolerance * scale:
        raise ValueError(
            "locked cubic elasticity is not isotropic; supply --Eprime-GPa for "
            "the desired anisotropic J-to-K conversion"
        )
    lam = C12
    mu = C44
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    if not (math.isfinite(E) and E > 0.0 and -1.0 < nu < 0.5):
        raise ValueError("derived isotropic elastic constants are invalid")
    return float(E), float(nu)


def _family_candidates(outroot: Path, lock: dict[str, Any]) -> Iterable[Path]:
    raw = str(lock.get("kernel_family", "")).strip()
    if raw:
        yield Path(raw).expanduser()
    fingerprint = str(lock.get("kernel_configuration_fingerprint", "")).strip()
    if fingerprint:
        repo_root = Path(__file__).resolve().parents[1]
        yield repo_root / "runs" / "v10_2_28_kernel_cache" / fingerprint / "family.json"
        yield outroot.parent / "v10_2_28_kernel_cache" / fingerprint / "family.json"


def _effective_modulus(
    outroot: Path,
    *,
    Eprime_GPa: float | None,
    constraint: str,
) -> tuple[float, dict[str, Any]]:
    if Eprime_GPa is not None:
        value = float(Eprime_GPa) * 1.0e9
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("--Eprime-GPa must be finite and positive")
        return value, {
            "source": "explicit_cli",
            "constraint": constraint,
            "Eprime_Pa": value,
        }

    lock_path = outroot / "v10_2_28_campaign_kernel_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"missing {lock_path}; supply --Eprime-GPa if no campaign kernel lock exists"
        )
    lock = _load_object(lock_path)
    family_path = next(
        (candidate.resolve() for candidate in _family_candidates(outroot, lock) if candidate.is_file()),
        None,
    )
    if family_path is None:
        raise FileNotFoundError(
            "could not locate the locked direct-kernel family; supply --Eprime-GPa"
        )
    family = _load_object(family_path)
    configuration = family.get("mechanical_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("locked family lacks mechanical_configuration")
    E, nu = _isotropic_moduli_from_cubic(
        configuration["crystal_C11_Pa"],
        configuration["crystal_C12_Pa"],
        configuration["crystal_C44_Pa"],
    )
    Eprime = E / (1.0 - nu * nu) if constraint == "plane_strain" else E
    return float(Eprime), {
        "source": "locked_direct_kernel_family",
        "family": str(family_path),
        "family_sha256": _sha256(family_path),
        "constraint": constraint,
        "E_Pa": E,
        "nu": nu,
        "Eprime_Pa": Eprime,
        "crystal_C11_Pa": float(configuration["crystal_C11_Pa"]),
        "crystal_C12_Pa": float(configuration["crystal_C12_Pa"]),
        "crystal_C44_Pa": float(configuration["crystal_C44_Pa"]),
    }


def _load_steps(case_root: Path) -> tuple[np.ndarray, Path]:
    paths = sorted(case_root.glob("steps_*K.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one steps CSV in {case_root}; found {paths}")
    data = np.atleast_1d(np.genfromtxt(paths[0], delimiter=",", names=True, dtype=float))
    required = {"KJ_Pa_sqrtm", "crack_extension_m", "da_block_m", "n_fire"}
    names = set(data.dtype.names or ())
    missing = required - names
    if missing:
        raise RuntimeError(f"{paths[0]} is missing columns {sorted(missing)}")
    return data, paths[0]


def _event_intervals(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fired = np.asarray(data["n_fire"], dtype=float) > 0.0
    post = 1.0e6 * np.asarray(data["crack_extension_m"], dtype=float)[fired]
    increment = 1.0e6 * np.asarray(data["da_block_m"], dtype=float)[fired]
    K = 1.0e-6 * np.asarray(data["KJ_Pa_sqrtm"], dtype=float)[fired]
    pre = post - increment
    valid = (
        np.isfinite(pre)
        & np.isfinite(post)
        & np.isfinite(K)
        & (increment > 0.0)
        & (post >= pre)
    )
    pre = np.maximum(pre[valid], 0.0)
    post = post[valid]
    K = K[valid]
    order = np.argsort(pre, kind="stable")
    return pre[order], post[order], K[order]


def _tail_width_um(
    achieved_um: float,
    *,
    tail_length_um: float,
    tail_fraction: float,
    policy: str,
) -> float:
    length_width = min(max(float(tail_length_um), 0.0), achieved_um)
    fraction_width = min(max(float(tail_fraction), 0.0) * achieved_um, achieved_um)
    if policy == "length":
        width = length_width
    elif policy == "fraction":
        width = fraction_width
    elif policy == "maximum":
        width = max(length_width, fraction_width)
    else:
        raise ValueError(f"unknown tail policy: {policy}")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("tail averaging window has zero or invalid width")
    return float(width)


def _weighted_tail_response(
    pre_um: np.ndarray,
    post_um: np.ndarray,
    values: np.ndarray,
    *,
    tail_start_um: float,
    tail_end_um: float,
) -> tuple[float, float, int]:
    """Return propagation-distance-weighted mean, covered length, event count."""
    pre = np.asarray(pre_um, dtype=float)
    post = np.asarray(post_um, dtype=float)
    response = np.asarray(values, dtype=float)
    overlap = np.maximum(
        np.minimum(post, float(tail_end_um)) - np.maximum(pre, float(tail_start_um)),
        0.0,
    )
    selected = overlap > 0.0
    covered = float(np.sum(overlap[selected]))
    if covered <= 0.0:
        raise ValueError("no fracture-event propagation overlaps the requested tail window")
    mean = float(np.sum(response[selected] * overlap[selected]) / covered)
    return mean, covered, int(np.count_nonzero(selected))


def _case_metadata(case_root: Path, option: str) -> tuple[str, str]:
    transfer_path = case_root / "v10_2_27_paper_four_class_parameter_transfer.json"
    transfer = _load_object(transfer_path) if transfer_path.is_file() else {}
    selection = transfer.get("paper_campaign_selection")
    selection = selection if isinstance(selection, dict) else {}
    candidate = str(
        transfer.get("selected_candidate")
        or selection.get("candidate_id")
        or option
    )
    response_class = str(
        selection.get("response_class")
        or transfer.get("source_material_class")
        or OPTION_LABELS.get(option, option)
    )
    return candidate, response_class


def _resolve_case_root(outroot: Path, record: dict[str, Any]) -> Path:
    raw = str(record.get("case_root", "")).strip()
    if raw:
        supplied = Path(raw).expanduser()
        if supplied.is_dir():
            return supplied.resolve()
    option = str(record["option"])
    temperature = float(record["temperature_K"])
    theta = float(record.get("theta_deg", 30.0))
    seed = int(record["seed"])
    return (
        outroot
        / option
        / f"T{temperature:g}K_th{theta:g}_seed{seed}"
    ).resolve()


def _acceptance_records(outroot: Path) -> list[dict[str, Any]]:
    path = outroot / "v10_2_27_campaign_acceptance.json"
    if path.is_file():
        payload = _load_object(path)
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise ValueError("campaign acceptance records must be a list")
        return [dict(record) for record in records]

    records: list[dict[str, Any]] = []
    for option_root in sorted(path for path in outroot.iterdir() if path.is_dir()):
        for case_root in sorted(path for path in option_root.iterdir() if path.is_dir()):
            match = CASE_RE.fullmatch(case_root.name)
            if not match:
                continue
            records.append(
                {
                    "option": option_root.name,
                    "temperature_K": float(match.group("T")),
                    "theta_deg": float(match.group("theta")),
                    "seed": int(match.group("seed")),
                    "case_root": str(case_root.resolve()),
                    "complete": (case_root / "COMPLETE").is_file(),
                }
            )
    return records


def _analyze_case(
    outroot: Path,
    record: dict[str, Any],
    *,
    Eprime_Pa: float,
    tail_length_um: float,
    tail_fraction: float,
    tail_policy: str,
    minimum_tail_coverage_fraction: float,
) -> dict[str, Any]:
    case_root = _resolve_case_root(outroot, record)
    if not case_root.is_dir():
        raise FileNotFoundError(f"case directory is missing: {case_root}")
    if not (case_root / "COMPLETE").is_file():
        raise RuntimeError(f"case is not marked COMPLETE: {case_root}")
    data, steps_path = _load_steps(case_root)
    pre, post, K = _event_intervals(data)
    if K.size == 0:
        raise RuntimeError(f"case contains no finite fracture events: {case_root}")

    achieved_um = float(np.max(post))
    width_um = _tail_width_um(
        achieved_um,
        tail_length_um=tail_length_um,
        tail_fraction=tail_fraction,
        policy=tail_policy,
    )
    tail_start_um = max(achieved_um - width_um, 0.0)
    J = (K * 1.0e6) ** 2 / float(Eprime_Pa) / 1000.0
    K_tail, covered_um, n_tail = _weighted_tail_response(
        pre,
        post,
        K,
        tail_start_um=tail_start_um,
        tail_end_um=achieved_um,
    )
    J_tail, J_covered_um, J_n_tail = _weighted_tail_response(
        pre,
        post,
        J,
        tail_start_um=tail_start_um,
        tail_end_um=achieved_um,
    )
    if not math.isclose(covered_um, J_covered_um, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("K and J tail coverage are inconsistent")
    if n_tail != J_n_tail:
        raise RuntimeError("K and J tail event counts are inconsistent")
    coverage_fraction = covered_um / width_um
    if coverage_fraction + 1.0e-12 < float(minimum_tail_coverage_fraction):
        raise RuntimeError(
            f"tail event coverage is incomplete for {case_root}: "
            f"{coverage_fraction:.6g} < {minimum_tail_coverage_fraction:.6g}"
        )

    option = str(record["option"])
    candidate, response_class = _case_metadata(case_root, option)
    return {
        "schema": SUMMARY_SCHEMA,
        "option_key": option,
        "plot_label": OPTION_LABELS.get(option, option),
        "candidate_id": candidate,
        "response_class": response_class,
        "temperature_K": float(record["temperature_K"]),
        "theta_deg": float(record.get("theta_deg", 30.0)),
        "seed": int(record["seed"]),
        "case_root": str(case_root),
        "steps_file": str(steps_path.resolve()),
        "steps_sha256": _sha256(steps_path),
        "n_fracture_event_rows": int(K.size),
        "initial_K_MPa_sqrt_m": float(K[0]),
        "initial_J_kJ_m2": float(J[0]),
        "achieved_extension_um": achieved_um,
        "tail_policy": tail_policy,
        "requested_tail_length_um": float(tail_length_um),
        "requested_tail_fraction": float(tail_fraction),
        "tail_start_extension_um": tail_start_um,
        "tail_end_extension_um": achieved_um,
        "tail_window_width_um": width_um,
        "tail_covered_extension_um": covered_um,
        "tail_coverage_fraction": coverage_fraction,
        "tail_event_row_count": n_tail,
        "tail_average_K_MPa_sqrt_m": K_tail,
        "tail_average_J_kJ_m2": J_tail,
        "Eprime_GPa": float(Eprime_Pa) / 1.0e9,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty response table")
    fieldnames = list(records[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _ordered_options(records: list[dict[str, Any]]) -> list[str]:
    present = {str(record["option_key"]) for record in records}
    ordered = [option for option in OPTION_ORDER if option in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _save_plot(
    records: list[dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for option in _ordered_options(records):
        rows = sorted(
            (record for record in records if record["option_key"] == option),
            key=lambda record: float(record["temperature_K"]),
        )
        ax.plot(
            [float(record["temperature_K"]) for record in rows],
            [float(record[value_key]) for record in rows],
            marker="o",
            label=str(rows[0]["plot_label"]),
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--tail-length-um", type=float, default=200.0)
    parser.add_argument("--tail-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tail-policy",
        choices=("maximum", "length", "fraction"),
        default="maximum",
    )
    parser.add_argument("--minimum-tail-coverage-fraction", type=float, default=0.95)
    parser.add_argument("--constraint", choices=("plane_strain", "plane_stress"), default="plane_strain")
    parser.add_argument("--Eprime-GPa", type=float)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    outroot = args.outroot.expanduser().resolve()
    if not outroot.is_dir():
        raise FileNotFoundError(f"campaign output root is missing: {outroot}")
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else outroot / "temperature_response"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    Eprime, elastic_audit = _effective_modulus(
        outroot,
        Eprime_GPa=args.Eprime_GPa,
        constraint=args.constraint,
    )
    source_records = _acceptance_records(outroot)
    selected = [record for record in source_records if record.get("complete", True) is True]
    if not selected:
        raise RuntimeError("no complete campaign cases were found")

    records = [
        _analyze_case(
            outroot,
            record,
            Eprime_Pa=Eprime,
            tail_length_um=args.tail_length_um,
            tail_fraction=args.tail_fraction,
            tail_policy=args.tail_policy,
            minimum_tail_coverage_fraction=args.minimum_tail_coverage_fraction,
        )
        for record in selected
    ]
    option_rank = {option: index for index, option in enumerate(OPTION_ORDER)}
    records.sort(
        key=lambda record: (
            option_rank.get(str(record["option_key"]), len(option_rank)),
            float(record["temperature_K"]),
            int(record["seed"]),
        )
    )

    csv_path = plot_dir / "v10_2_28_four_class_KJ_temperature_response.csv"
    json_path = plot_dir / "v10_2_28_four_class_KJ_temperature_response.json"
    _write_csv(csv_path, records)
    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    generated: list[Path] = [csv_path, json_path]
    if not args.no_plots:
        generated.extend(
            _save_plot(
                records,
                value_key="initial_K_MPa_sqrt_m",
                ylabel="K (MPa√m)",
                title="Initial fracture resistance",
                output_stem=plot_dir / "K_initial_vs_temperature",
                dpi=args.dpi,
            )
        )
        generated.extend(
            _save_plot(
                records,
                value_key="tail_average_K_MPa_sqrt_m",
                ylabel="K (MPa√m)",
                title="Tail-average fracture resistance",
                output_stem=plot_dir / "K_tail_average_vs_temperature",
                dpi=args.dpi,
            )
        )
        generated.extend(
            _save_plot(
                records,
                value_key="initial_J_kJ_m2",
                ylabel="J (kJ/m²)",
                title="Initial energy-release response",
                output_stem=plot_dir / "J_initial_vs_temperature",
                dpi=args.dpi,
            )
        )
        generated.extend(
            _save_plot(
                records,
                value_key="tail_average_J_kJ_m2",
                ylabel="J (kJ/m²)",
                title="Tail-average energy-release response",
                output_stem=plot_dir / "J_tail_average_vs_temperature",
                dpi=args.dpi,
            )
        )

    audit = {
        "schema": AUDIT_SCHEMA,
        "outroot": str(outroot),
        "plot_dir": str(plot_dir),
        "case_count": len(records),
        "option_count": len({record["option_key"] for record in records}),
        "temperature_count": len({record["temperature_K"] for record in records}),
        "tail_length_um": float(args.tail_length_um),
        "tail_fraction": float(args.tail_fraction),
        "tail_policy": args.tail_policy,
        "minimum_tail_coverage_fraction": float(args.minimum_tail_coverage_fraction),
        "tail_average_weighting": "fracture-event overlap length in crack-extension space",
        "J_definition": "eventwise KJ_Pa_sqrtm squared divided by Eprime",
        "elastic_conversion": elastic_audit,
        "generated_files": [str(path.resolve()) for path in generated],
        "generated_sha256": {
            str(path.resolve()): _sha256(path) for path in generated
        },
    }
    audit_path = plot_dir / "v10_2_28_four_class_KJ_temperature_plot_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
