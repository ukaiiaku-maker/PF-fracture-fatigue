"""v10.1.7.4 entry point: anisotropic emission plus stochastic avalanche length."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import continuum_source_tip
from . import crystal as _crystal
from . import fem as _fem
from . import sharp_front_v10_1_5 as _campaign
from . import sharp_front_v10_1_7_3 as _avalanche
from .anisotropic_emission_v10174 import (
    MODEL_ID,
    AnisotropicEmissionConfig,
    AnisotropicStochasticAvalancheTipEngine,
    wrap_assemble_mechanics,
    wrap_near_tip_stress_tensor,
)


def _option_value(args: list[str], name: str, default: str | None = None):
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _config(args: list[str]) -> AnisotropicEmissionConfig:
    theta = float(
        _option_value(
            args,
            "--crystal-theta-deg",
            os.environ.get("ANISOTROPIC_CRYSTAL_THETA_DEG", "45"),
        )
    )
    return AnisotropicEmissionConfig(
        enabled=_env_bool("ANISOTROPIC_EMISSION_ENABLED", True),
        crystal_theta_deg=theta,
        probe_radius_m=float(
            os.environ.get("ANISOTROPIC_PROBE_RADIUS_M", "1e-5")
        ),
        sector_half_angle_deg=float(
            os.environ.get(
                "ANISOTROPIC_PROBE_HALF_ANGLE_DEG", "25"
            )
        ),
        damage_cutoff=float(
            os.environ.get("ANISOTROPIC_PROBE_DAMAGE_CUTOFF", "0.85")
        ),
        min_elements=int(
            os.environ.get("ANISOTROPIC_PROBE_MIN_ELEMENTS", "3")
        ),
        schmid_reference=float(
            os.environ.get("ANISOTROPIC_SCHMID_REFERENCE", "0.5")
        ),
        shared_forest_density=_env_bool(
            "ANISOTROPIC_SHARED_FOREST_DENSITY", True
        ),
        require_reliable_probe=_env_bool(
            "ANISOTROPIC_REQUIRE_RELIABLE_PROBE", True
        ),
    ).validate()


def _rewrite_audits(
    args: list[str],
    config: AnisotropicEmissionConfig,
    use_avalanche_backend: bool,
) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)

    payload = AnisotropicStochasticAvalancheTipEngine.audit_payload()
    (root / "anisotropic_emission_audit_v10174.json").write_text(
        json.dumps(payload, indent=2)
    )

    mode_path = root / "v10_1_driver_modes.json"
    mode = json.loads(mode_path.read_text()) if mode_path.exists() else {}
    mode.update(
        {
            "schema": "v10.1.7.4_anisotropic_emission_pilot",
            "anisotropic_emission_model_id": MODEL_ID,
            "anisotropic_emission_enabled": bool(config.enabled),
            "anisotropic_emission_two_reduced_slip_traces": True,
            "anisotropic_emission_full_3d_crystal_plasticity": False,
            "anisotropic_emission_factors_normalized": False,
            "anisotropic_emission_factors_clipped": False,
            "anisotropic_emission_post_hazard_weighting": False,
            "anisotropic_emission_factor_application": (
                "sigma_emit_s=max(f_s*sigma_open-sigma_back_s,0)"
            ),
            "anisotropic_transport_channel_resolved": True,
            "anisotropic_opening_stress_unshielded": True,
            "anisotropic_cleavage_shielding_only": True,
            "anisotropic_taylor_backstress_emission_only": True,
            "anisotropic_probe_radius_m": float(config.probe_radius_m),
            "anisotropic_probe_half_angle_deg": float(
                config.sector_half_angle_deg
            ),
            "anisotropic_probe_damage_cutoff": float(
                config.damage_cutoff
            ),
            "anisotropic_probe_min_elements": int(config.min_elements),
            "anisotropic_schmid_reference": float(
                config.schmid_reference
            ),
            "anisotropic_shared_forest_density": bool(
                config.shared_forest_density
            ),
            "anisotropic_use_avalanche_backend": bool(
                use_avalanche_backend
            ),
            "stochastic_geometry_transaction_modified": False,
        }
    )
    mode_path.write_text(json.dumps(mode, indent=2))

    source_path = root / "v10_1_1_source_model.json"
    source = (
        json.loads(source_path.read_text())
        if source_path.exists()
        else {}
    )
    source.update(
        {
            "schema": "v10.1.7.4_anisotropic_source_model",
            "anisotropic_emission_model_id": MODEL_ID,
            "tip_source_state": (
                "bounded continuum capacity per reduced slip-trace channel"
            ),
            "emission_drive": (
                "finite-radius tensor shape times unshielded sharp-tip "
                "opening stress"
            ),
            "directional_factor_applied_after_hazard": False,
            "transport": (
                "channel-resolved Peierls-Taylor with shared forest density"
                if config.shared_forest_density
                else "channel-resolved Peierls-Taylor with self-channel forest"
            ),
        }
    )
    source_path.write_text(json.dumps(source, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    config = _config(args)
    if not config.enabled:
        raise SystemExit(
            "v10.1.7.4 requires ANISOTROPIC_EMISSION_ENABLED=1"
        )
    if "--crystal-aniso" not in args:
        raise SystemExit(
            "v10.1.7.4 anisotropic emission requires --crystal-aniso"
        )
    if "--material-class" not in args and "--material-manifest" not in args:
        raise SystemExit(
            "v10.1.7.4 requires --material-class or --material-manifest"
        )

    use_avalanche_backend = _env_bool(
        "ANISOTROPIC_USE_AVALANCHE_BACKEND", True
    )

    engine = AnisotropicStochasticAvalancheTipEngine
    engine.configure_campaign(
        _campaign.BACKSTRESS_SCALE,
        _campaign.REFRESH_SCALE,
    )
    engine.configure_hazard(
        _avalanche.HAZARD_MODE,
        _avalanche.HAZARD_SEED,
        _avalanche.HAZARD_MIN_THRESHOLD,
    )
    engine.configure_avalanche(
        _avalanche.EVENT_LENGTH_MODE,
        _avalanche.EVENT_MIN_FACTOR,
        _avalanche.EVENT_MAX_FACTOR,
        _avalanche.EVENT_SUBSEGMENT_FRACTION,
    )
    engine.configure_anisotropic_emission(config)

    original_continuum = (
        continuum_source_tip.ContinuumSourceKineticTipEngine
    )
    original_protected = (
        _campaign._protected.ContinuumSourceKineticTipEngine
    )
    original_assemble = _fem.assemble_mechanics
    original_near_tip = _crystal.near_tip_stress_tensor

    continuum_source_tip.ContinuumSourceKineticTipEngine = engine
    _campaign._protected.ContinuumSourceKineticTipEngine = engine
    _fem.assemble_mechanics = wrap_assemble_mechanics(
        original_assemble
    )
    _crystal.near_tip_stress_tensor = wrap_near_tip_stress_tensor(
        original_near_tip,
        config,
    )

    try:
        print(
            "  v10.1.7.4 anisotropic emission: "
            f"theta={config.crystal_theta_deg:g}deg "
            f"probe={config.probe_radius_m*1e6:g}um "
            f"half_angle={config.sector_half_angle_deg:g}deg "
            f"damage_cutoff={config.damage_cutoff:g} "
            f"schmid_ref={config.schmid_reference:g} "
            "channels=2 factor=inside_barrier "
            "post_hazard_weight=0 "
            f"avalanche_backend={int(use_avalanche_backend)}"
        )
        if use_avalanche_backend:
            result = _avalanche.main(args)
        else:
            result = _campaign.main(args)
        _rewrite_audits(args, config, use_avalanche_backend)
        return result
    finally:
        continuum_source_tip.ContinuumSourceKineticTipEngine = (
            original_continuum
        )
        _campaign._protected.ContinuumSourceKineticTipEngine = (
            original_protected
        )
        _fem.assemble_mechanics = original_assemble
        _crystal.near_tip_stress_tensor = original_near_tip


if __name__ == "__main__":
    main()
