"""Persist accepted-step configurational-J and energy-balance diagnostics.

The legacy sharp-front driver already maintains accepted-step histories for
external work, stored elastic energy, bulk plastic work, and tip-emission work,
but only a subset was written to the production step CSV.  This overlay wraps
``numpy.savetxt`` only while the audited entry point is running and augments
each ``steps_*K.csv`` table with those histories.

The implementation reads the already accepted ``hist`` arrays at final output
time.  It therefore excludes rejected adaptive trial steps.  Newer fixed-point
paths may additionally provide ``hist['W_p_constitutive']``; when present, the
CSV retains that older constitutive estimate beside the primary endpoint-path
work ledger rather than silently replacing it.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "v10.2.27_accepted_step_energy_ledger_v2"
_EXTRA_COLUMNS = (
    "J_effective_direct_J_per_m2",
    "J_signed_direct_J_per_m2",
    "W_ext_cumulative_J_per_m",
    "U_elastic_J_per_m",
    "W_bulk_plastic_cumulative_J_per_m",
    "W_tip_emit_cumulative_J_per_m",
    "W_fracture_residual_cumulative_J_per_m",
)
_OPTIONAL_PATH_WORK_COLUMNS = (
    "W_bulk_plastic_constitutive_cumulative_J_per_m",
    "W_bulk_plastic_path_minus_constitutive_cumulative_J_per_m",
    "W_fracture_residual_constitutive_cumulative_J_per_m",
)

_original_savetxt = None
_install_depth = 0
_records: list[dict[str, Any]] = []


def _as_2d(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("step table must be one- or two-dimensional")
    return array


def _history_array(hist: dict[str, Any], key: str, nrows: int) -> np.ndarray:
    values = np.asarray(hist.get(key, []), dtype=float).reshape(-1)
    if values.size != nrows:
        raise RuntimeError(
            f"accepted-step history {key!r} has {values.size} rows; expected {nrows}"
        )
    return values


def _direct_j_from_front_rows(
    steps: np.ndarray,
    fronts_rows: Any,
    effective_modulus_pa: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    nrows = steps.shape[0]
    effective = np.full(nrows, np.nan, dtype=float)
    signed = np.full(nrows, np.nan, dtype=float)

    front_array = np.asarray(fronts_rows, dtype=float)
    if front_array.size:
        if front_array.ndim == 1:
            front_array = front_array.reshape(1, -1)
        if front_array.shape[1] >= 19:
            root_by_step: dict[int, tuple[float, float]] = {}
            for row in front_array:
                if int(round(float(row[1]))) != 0:
                    continue
                step = int(round(float(row[0])))
                root_by_step[step] = (float(row[17]), float(row[16]))
            for index, row in enumerate(steps):
                item = root_by_step.get(int(round(float(row[0])))
                if item is not None:
                    effective[index], signed[index] = item

    missing = ~np.isfinite(effective)
    if np.any(missing):
        if not np.isfinite(effective_modulus_pa) or effective_modulus_pa <= 0.0:
            raise RuntimeError("cannot recover direct J without a positive Eprime")
        effective[missing] = (
            np.maximum(steps[missing, 3], 0.0) ** 2 / effective_modulus_pa
        )
        signed[missing] = effective[missing]
        provenance = (
            "root_front_direct_J_with_exact_nondeflecting_KJ_inverse_fallback"
        )
    else:
        provenance = "root_front_direct_J"
    return effective, signed, provenance


def augment_steps_table(
    values: Any,
    header: str,
    hist: dict[str, Any],
    fronts_rows: Any,
    effective_modulus_pa: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Return an augmented accepted-step table and audit record."""
    steps = _as_2d(values)
    if "J_effective_direct_J_per_m2" in header:
        return steps, header, {"already_augmented": True}
    if steps.shape[1] < 15:
        raise RuntimeError("unexpected legacy steps table width")

    nrows = steps.shape[0]
    w_ext = _history_array(hist, "W_ext", nrows)
    u_el = _history_array(hist, "U_el", nrows)
    w_bulk = _history_array(hist, "W_p", nrows)
    w_emit = _history_array(hist, "W_emit", nrows)
    j_effective, j_signed, provenance = _direct_j_from_front_rows(
        steps, fronts_rows, effective_modulus_pa
    )
    residual = w_ext - u_el - w_bulk - w_emit

    extra_arrays = [
        j_effective,
        j_signed,
        w_ext,
        u_el,
        w_bulk,
        w_emit,
        residual,
    ]
    columns = list(_EXTRA_COLUMNS)
    optional_path_work = False
    if "W_p_constitutive" in hist:
        w_bulk_constitutive = _history_array(
            hist, "W_p_constitutive", nrows
        )
        residual_constitutive = (
            w_ext - u_el - w_bulk_constitutive - w_emit
        )
        extra_arrays.extend(
            [
                w_bulk_constitutive,
                w_bulk - w_bulk_constitutive,
                residual_constitutive,
            ]
        )
        columns.extend(_OPTIONAL_PATH_WORK_COLUMNS)
        optional_path_work = True

    extra = np.column_stack(extra_arrays)
    augmented = np.column_stack([steps, extra])
    augmented_header = header.rstrip() + "," + ",".join(columns)
    audit = {
        "schema": SCHEMA,
        "row_count": int(nrows),
        "direct_J_provenance": provenance,
        "accepted_step_histories": True,
        "adaptive_rejected_trials_excluded": True,
        "supports_tip_only_and_full_field_bulk_plasticity": True,
        "energy_balance_definition": (
            "W_fracture_residual=W_ext-U_elastic-W_bulk_plastic-W_tip_emit; "
            "residual contains fracture-surface work plus discretization error"
        ),
        "primary_bulk_plastic_history": "hist.W_p",
        "constitutive_comparison_history_present": optional_path_work,
        "columns_added": columns,
    }
    return augmented, augmented_header, audit


def _find_driver_locals(frame, nrows: int) -> dict[str, Any] | None:
    current = frame
    while current is not None:
        local = current.f_locals
        hist = local.get("hist")
        if isinstance(hist, dict):
            try:
                if len(hist.get("Uapp", [])) == nrows:
                    return local
            except Exception:
                pass
        current = current.f_back
    return None


def install_energy_ledger_output() -> None:
    """Install the temporary ``numpy.savetxt`` output overlay."""
    global _original_savetxt, _install_depth
    if _install_depth > 0:
        _install_depth += 1
        return
    _install_depth = 1
    _records.clear()
    _original_savetxt = np.savetxt

    def wrapped_savetxt(fname, X, *args, **kwargs):
        path = Path(str(fname))
        header = str(kwargs.get("header", ""))
        is_steps = path.name.startswith("steps_") and path.suffix == ".csv"
        if not is_steps or not header:
            return _original_savetxt(fname, X, *args, **kwargs)

        values = _as_2d(X)
        local = _find_driver_locals(inspect.currentframe().f_back, values.shape[0])
        if local is None:
            raise RuntimeError(
                f"could not locate accepted-step histories while writing {path}"
            )
        mat = local.get("mat")
        effective_modulus_pa = float(getattr(mat, "Eprime", np.nan))
        augmented, new_header, audit = augment_steps_table(
            values,
            header,
            local["hist"],
            local.get("fronts_rows", []),
            effective_modulus_pa,
        )
        call_kwargs = dict(kwargs)
        call_kwargs["header"] = new_header
        result = _original_savetxt(fname, augmented, *args, **call_kwargs)
        audit["steps_csv"] = str(path.resolve())
        audit["effective_modulus_Pa_used_only_for_exact_inverse_fallback"] = (
            effective_modulus_pa
        )
        _records.append(audit)
        return result

    np.savetxt = wrapped_savetxt


def restore_energy_ledger_output() -> None:
    global _original_savetxt, _install_depth
    if _install_depth <= 0:
        return
    _install_depth -= 1
    if _install_depth == 0 and _original_savetxt is not None:
        np.savetxt = _original_savetxt
        _original_savetxt = None


def write_energy_ledger_audit(outroot: str | Path) -> Path:
    root = Path(outroot).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "v10_2_27_energy_ledger_output_audit.json"
    payload = {
        "schema": SCHEMA,
        "records": list(_records),
        "record_count": len(_records),
        "direct_configurational_J_is_not_total_absorbed_energy": True,
        "bulk_plasticity_accounting": (
            "W_bulk_plastic is the accepted cumulative FEM plastic-work ledger; "
            "newer fixed-point paths use endpoint-average stress contracted with "
            "the actual accepted plastic-strain increment"
        ),
        "constitutive_comparison_accounting": (
            "when present, W_bulk_plastic_constitutive retains the older local "
            "constitutive estimate for diagnosis and is not the primary ledger"
        ),
        "tip_emission_accounting": (
            "W_tip_emit is the accepted cumulative process-zone emission-work ledger"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
