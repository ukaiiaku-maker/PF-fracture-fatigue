"""Fixed-DeltaK entry for v10.2.30 hazard-energy-gated fatigue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import fatigue_v1
from .fixed_deltaK_v1021 import (
    fixed_deltaK_audit_payload,
    install_fixed_deltaK_waveform,
)
from . import sharp_front_v10_2_1 as _legacy_fixed
from . import sharp_front_v10_2_30_energy_gated_fatigue as _energy
from .hazard_energy_event_gate_v10230 import (
    audit_payload as energy_gate_audit_payload,
    set_latest_probe_K,
)


MODEL_ID = "v10.2.30_persistent_site_fixed_deltaK_hazard_energy_gate"


def _write_audit(args: list[str], target_deltaK: float) -> dict:
    out = _legacy_fixed._option_value(args, "--out")
    if not out:
        return {}
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    R = float(_legacy_fixed._option_value(args, "--R", "0.1") or 0.1)
    target_Kmax = target_deltaK / (1.0 - R)
    target_Kmin = R * target_Kmax
    semantics = _legacy_fixed._normalize_output_semantics(root, target_deltaK, R)

    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    energy_path = root / "hazard_energy_gated_events_v10_2_30.json"
    events = []
    if geometry_path.is_file():
        data = json.loads(geometry_path.read_text())
        if isinstance(data, list):
            events = data
    energy_events = []
    if energy_path.is_file():
        data = json.loads(energy_path.read_text())
        if isinstance(data, list):
            energy_events = data

    payload = fixed_deltaK_audit_payload()
    payload.update(
        {
            "schema": MODEL_ID,
            "base_fixed_deltaK_control": "v10.2.1",
            "fatigue_engine": (
                "v10.2.30_transactional_persistent_site_hazard_energy_gate"
            ),
            "parameter_option": _legacy_fixed._option_value(
                args, "--parameter-option"
            ),
            "cleavage_hazard_seed": int(
                os.environ.get("CLEAVAGE_HAZARD_SEED", "0")
            ),
            "target_deltaK_MPa_sqrt_m": float(target_deltaK),
            "target_Kmax_MPa_sqrt_m": float(target_Kmax),
            "target_Kmin_MPa_sqrt_m": float(target_Kmin),
            "R": R,
            "frequency_Hz": float(
                _legacy_fixed._option_value(
                    args, "--frequency-Hz", "1000"
                )
                or 1000.0
            ),
            "cycles_max": float(
                _legacy_fixed._option_value(args, "--cycles-max", "0")
                or 0.0
            ),
            "fatigue_control_mode": "prescribed_fixed_local_deltaK",
            "fem_loading_mode": "held_nonzero_geometry_tensor_probe",
            "probe_KJ_is_fatigue_driving_K": False,
            "probe_energy_rescaled_to_event_K": True,
            "probe_energy_scale_expression": "(K_event/K_probe)**2",
            "persistent_site_source": True,
            "finite_source_inventory": False,
            "source_depletion": False,
            "source_refresh": False,
            "explicit_recovery": False,
            "state_coupled_cleavage_hazard": True,
            "cleavage_first_passage_only_trigger": True,
            "transactional_tip_translation": True,
            "stochastic_geometry_events": len(events),
            "hazard_energy_gated_events": len(energy_events),
            "censor_status": (
                "propagated" if energy_events else "right_censored_no_event"
            ),
            "Gc0_athermal_active": False,
            "independent_fracture_energy_active": False,
            **semantics,
            "energy_gate": energy_gate_audit_payload(),
        }
    )
    target_pa = float(target_deltaK) * 1.0e6
    error_pa = float(
        payload.get("maximum_abs_target_error_Pa_sqrt_m", float("inf"))
    )
    payload["fixed_deltaK_exact_within_relative_1e-12"] = bool(
        error_pa <= max(1.0e-6, 1.0e-12 * target_pa)
    )
    (root / "v10_2_30_fixed_deltaK_control.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    return payload


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    target_deltaK = float(
        _legacy_fixed._pop_value(args, "--target-deltaK-MPa-sqrt-m")
    )
    if target_deltaK <= 0.0:
        raise SystemExit("--target-deltaK-MPa-sqrt-m must be positive")
    if "--fatigue-cycles" not in args:
        args.append("--fatigue-cycles")
    _legacy_fixed._ensure_toggle(
        args, "--no-cyclic-mechanics", "--cyclic-mechanics"
    )
    _legacy_fixed._ensure_toggle(
        args, "--fatigue-hold-load", "--no-fatigue-hold-load"
    )

    R = float(_legacy_fixed._option_value(args, "--R", "0.1") or 0.1)
    if not 0.0 <= R < 1.0:
        raise SystemExit("v10.2.30 fixed-DeltaK mode requires 0 <= R < 1")
    target_Kmax = target_deltaK / (1.0 - R)
    print(
        "  v10.2.30 persistent-site fixed-DeltaK energy-gated fatigue: "
        f"DeltaK={target_deltaK:g} MPa*sqrt(m) "
        f"Kmax={target_Kmax:g} MPa*sqrt(m) R={R:g}"
    )

    with install_fixed_deltaK_waveform(target_deltaK):
        fixed_factory = fatigue_v1.FatigueWaveform

        def capture_probe(*factory_args, **factory_kwargs):
            incoming = factory_kwargs.get(
                "Kmax", factory_args[0] if factory_args else None
            )
            set_latest_probe_K(incoming)
            return fixed_factory(*factory_args, **factory_kwargs)

        fatigue_v1.FatigueWaveform = capture_probe
        try:
            with _legacy_fixed._allow_right_censored_stochastic_summary():
                with _legacy_fixed._fixed_deltaK_console_semantics(
                    target_deltaK, R
                ):
                    result = _energy.main(args)
        finally:
            fatigue_v1.FatigueWaveform = fixed_factory
            set_latest_probe_K(None)

    audit = _write_audit(args, target_deltaK)
    print(
        "  fixed-DeltaK v10.2.30 audit: "
        f"events={audit.get('hazard_energy_gated_events', 0)} "
        f"status={audit.get('censor_status', 'unknown')}"
    )
    return result


if __name__ == "__main__":
    main()
