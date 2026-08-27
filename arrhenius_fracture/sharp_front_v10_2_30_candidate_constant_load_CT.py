"""Candidate-registry entry for exact constant-load virtual C(T) trajectories."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

from . import fatigue_v1
from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_1 as _legacy
from . import sharp_front_v10_2_30_energy_gated_fatigue as _energy
from .hazard_energy_event_gate_v10230 import set_latest_probe_K
from .virtual_ct_v10230 import (
    CompactTensionGeometry, ct_geometry_factor, ct_k_from_load_pa_sqrt_m,
    ct_load_from_k_pa_sqrt_m,
)

MODEL_ID = "v10.2.30_candidate_constant_load_virtual_CT_v1"


def _projected_extension(root: Path) -> float:
    path = root / "stochastic_avalanche_geometry_events.json"
    if not path.is_file():
        return 0.0
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        return 0.0
    return max(float(rows[-1]["x1"]) - float(rows[0]["x0"]), 0.0)


def _run(args: list[str], root: Path, geometry: CompactTensionGeometry,
         initial_kmax_mpa: float) -> object:
    R = float(_legacy._option_value(args, "--R", "0.1") or 0.1)
    if not -1.0 <= R < 1.0:
        raise SystemExit("constant-load C(T) mode requires -1 <= R < 1")
    initial_kmax_pa = initial_kmax_mpa*1.0e6
    pmax = ct_load_from_k_pa_sqrt_m(initial_kmax_pa, geometry, geometry.initial_crack_m)
    original = fatigue_v1.FatigueWaveform
    audit = {"waveforms_created": 0, "minimum_Kmax_driver_Pa_sqrt_m": None,
             "maximum_Kmax_driver_Pa_sqrt_m": None, "maximum_R_error": 0.0}

    def factory(*factory_args, **factory_kwargs):
        incoming = factory_kwargs.get("Kmax", factory_args[0] if factory_args else None)
        set_latest_probe_K(incoming)
        extension = _projected_extension(root)
        crack = geometry.initial_crack_m + extension
        kmax = ct_k_from_load_pa_sqrt_m(pmax, geometry, crack)
        kwargs = dict(factory_kwargs)
        kwargs.update({"Kmax": kmax, "R": R})
        if R < 0.0:
            kwargs["closure_clip"] = False
        wave = original(**kwargs)
        audit["waveforms_created"] += 1
        lo = audit["minimum_Kmax_driver_Pa_sqrt_m"]
        hi = audit["maximum_Kmax_driver_Pa_sqrt_m"]
        audit["minimum_Kmax_driver_Pa_sqrt_m"] = kmax if lo is None else min(lo, kmax)
        audit["maximum_Kmax_driver_Pa_sqrt_m"] = kmax if hi is None else max(hi, kmax)
        audit["maximum_R_error"] = max(audit["maximum_R_error"], abs(wave.R-R))
        return wave

    factory._prescribed_fixed_deltaK_control = True
    fatigue_v1.FatigueWaveform = factory
    try:
        with _legacy._allow_right_censored_stochastic_summary():
            result = _energy.main(args)
    finally:
        fatigue_v1.FatigueWaveform = original
        set_latest_probe_K(None)

    final_extension = _projected_extension(root)
    final_crack = geometry.initial_crack_m + final_extension
    final_kmax = ct_k_from_load_pa_sqrt_m(pmax, geometry, final_crack)
    payload = {
        "schema": MODEL_ID, "fatigue_control_mode": "constant_load_virtual_CT",
        "deltaK_semantics": "LOCAL_TIP_EFFECTIVE_K_ENERGY_EQUIVALENT_TO_NOMINAL",
        "parameter_option": _legacy._option_value(args, "--parameter-option"),
        "R": R, "frequency_Hz": float(_legacy._option_value(args, "--frequency-Hz", "1000") or 1000),
        "cleavage_hazard_seed": int(os.environ.get("CLEAVAGE_HAZARD_SEED", "0")),
        "cycles_max": float(_legacy._option_value(args, "--cycles-max", "0") or 0),
        "W_m": geometry.width_m, "B_m": geometry.thickness_m,
        "a0_m": geometry.initial_crack_m, "a0_over_W": geometry.initial_crack_m/geometry.width_m,
        "Pmax_N": pmax, "Pmin_N": R*pmax,
        "initial_Kmax_nominal_MPa_sqrt_m": initial_kmax_mpa,
        "initial_Kmin_nominal_MPa_sqrt_m": R*initial_kmax_mpa,
        "initial_deltaK_nominal_MPa_sqrt_m": (1-R)*initial_kmax_mpa,
        "final_projected_extension_m": final_extension, "final_a_macro_m": final_crack,
        "final_Kmax_nominal_MPa_sqrt_m": final_kmax/1e6,
        "final_Kmin_nominal_MPa_sqrt_m": R*final_kmax/1e6,
        "final_deltaK_nominal_MPa_sqrt_m": (1-R)*final_kmax/1e6,
        "K_driver_equals_K_nominal_by_J_equivalence": True,
        "tip_radius_used_in_nominal_K": False, "resumed": False,
        "acceleration_mode": "explicit_only", **audit,
    }
    (root/"v10_2_30_constant_load_CT_control.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    # Compatibility metadata used only by the existing developed-rate extractor.
    compatibility = dict(payload)
    compatibility.update({"schema": MODEL_ID, "target_deltaK_MPa_sqrt_m": (1-R)*initial_kmax_mpa,
                          "target_Kmax_MPa_sqrt_m": initial_kmax_mpa,
                          "target_Kmin_MPa_sqrt_m": R*initial_kmax_mpa})
    (root/"v10_2_30_fixed_deltaK_control.json").write_text(json.dumps(compatibility, indent=2, sort_keys=True)+"\n")
    return result


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    initial_delta = float(_legacy._pop_value(args, "--target-deltaK-MPa-sqrt-m"))
    R = float(_legacy._option_value(args, "--R", "0.1") or 0.1)
    initial_kmax = float(os.environ.get("V10230_CT_INITIAL_KMAX_MPA_SQRT_M", initial_delta/(1-R)))
    out = _legacy._option_value(args, "--out")
    if not out:
        raise SystemExit("constant-load C(T) mode requires --out")
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    geometry = CompactTensionGeometry(
        float(os.environ.get("V10230_CT_W_M", "0.01")),
        float(os.environ.get("V10230_CT_B_M", "0.0025")),
        float(os.environ.get("V10230_CT_A0_M", "0.005")),
    ).validate()
    if "--fatigue-cycles" not in args: args.append("--fatigue-cycles")
    _legacy._ensure_toggle(args, "--no-cyclic-mechanics", "--cyclic-mechanics")
    _legacy._ensure_toggle(args, "--fatigue-hold-load", "--no-fatigue-hold-load")

    registry = Path(os.environ.get("V10230_CANDIDATE_REGISTRY", "")).resolve()
    selection = Path(os.environ.get("V10230_CANDIDATE_SELECTION", "")).resolve()
    if not registry.is_file() or not selection.is_file():
        raise SystemExit("constant-load C(T) mode requires immutable candidate registry and selection")
    with registry.open(newline="") as stream: rows = list(csv.DictReader(stream))
    valid = {row["option_key"]: row["candidate_id"] for row in rows}
    old = (_paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS)
    _paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS = registry, selection, valid
    try:
        return _run(args, root, geometry, initial_kmax)
    finally:
        _paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS = old


if __name__ == "__main__":
    main()
