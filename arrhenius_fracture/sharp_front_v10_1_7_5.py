"""v10.1.7.5 entry point for controlled anisotropic-transport comparison.

Both transport modes use the same tensor-resolved anisotropic emission law and
the same campaign-calibrated finite source budget.  The only switch is the
active-zone Peierls--Taylor transport operator:

``validated_scalar``
    Preserve the inherited, campaign-validated common-stress/common-velocity
    transport operator.  Only emission is channel resolved.

``channel_resolved``
    Use the v10.1.7.4 channel-specific transport stresses, rates, and velocities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Callable

from . import anisotropic_emission_v10174 as _anisotropic
from . import sharp_front_v10_1_7_4 as _v10174


MODEL_ID = "v10.1.7.5_anisotropic_transport_comparison"
VALIDATED_SCALAR_TRANSPORT = "validated_scalar"
CHANNEL_RESOLVED_TRANSPORT = "channel_resolved"
_VALID_MODES = {
    VALIDATED_SCALAR_TRANSPORT,
    CHANNEL_RESOLVED_TRANSPORT,
}


def normalize_transport_mode(value: str | None) -> str:
    raw = (value or VALIDATED_SCALAR_TRANSPORT).strip().lower().replace("-", "_")
    aliases = {
        "scalar": VALIDATED_SCALAR_TRANSPORT,
        "validated": VALIDATED_SCALAR_TRANSPORT,
        "legacy": VALIDATED_SCALAR_TRANSPORT,
        "emission_only": VALIDATED_SCALAR_TRANSPORT,
        "anisotropic": CHANNEL_RESOLVED_TRANSPORT,
        "channel": CHANNEL_RESOLVED_TRANSPORT,
        "resolved": CHANNEL_RESOLVED_TRANSPORT,
        "full": CHANNEL_RESOLVED_TRANSPORT,
    }
    mode = aliases.get(raw, raw)
    if mode not in _VALID_MODES:
        allowed = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"invalid ANISOTROPIC_TRANSPORT_MODE={value!r}; choose {allowed}")
    return mode


def make_transport_installer(
    original_install: Callable,
    mode: str,
) -> Callable:
    """Wrap the v10.1.7.4 installer while changing only the transport operator."""
    selected = normalize_transport_mode(mode)

    def install(state, config) -> None:
        inherited_evolve = state.evolve
        original_install(state, config)
        if selected == VALIDATED_SCALAR_TRANSPORT:
            # The inherited operator still calls state._emit dynamically, so it
            # uses the anisotropic emitter installed immediately above while
            # preserving the validated common transport stress and velocity.
            state.evolve = inherited_evolve
        state._anisotropic_transport_mode = selected
        state._anisotropic_transport_channel_resolved = bool(
            selected == CHANNEL_RESOLVED_TRANSPORT
        )

    return install


def make_diagnostics_wrapper(original: Callable, mode: str) -> Callable:
    selected = normalize_transport_mode(mode)

    def diagnostics(self):
        payload = original(self)
        payload.update(
            {
                "anisotropic_transport_mode": selected,
                "anisotropic_transport_channel_resolved": bool(
                    selected == CHANNEL_RESOLVED_TRANSPORT
                ),
                "anisotropic_transport_validated_common_operator": bool(
                    selected == VALIDATED_SCALAR_TRANSPORT
                ),
            }
        )
        return payload

    return diagnostics


def _option_value(args: list[str], name: str, default: str | None = None):
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _rewrite_outputs(args: list[str], mode: str) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    selected = normalize_transport_mode(mode)
    channel_resolved = selected == CHANNEL_RESOLVED_TRANSPORT

    transport_payload = {
        "schema": MODEL_ID,
        "transport_mode": selected,
        "anisotropic_emission_identical_between_transport_modes": True,
        "campaign_source_budget_preserved": True,
        "campaign_geometry_refresh_preserved": True,
        "campaign_active_shielding_cap_preserved": True,
        "temporal_source_recycling": False,
        "channel_resolved_transport": channel_resolved,
        "validated_common_transport_operator": not channel_resolved,
    }
    (root / "v10_1_7_5_transport_mode.json").write_text(
        json.dumps(transport_payload, indent=2)
    )

    mode_path = root / "v10_1_driver_modes.json"
    driver = json.loads(mode_path.read_text()) if mode_path.exists() else {}
    driver.update(transport_payload)
    driver["schema"] = "v10.1.7.5_transport_comparison"
    driver["anisotropic_transport_channel_resolved"] = channel_resolved
    driver["anisotropic_transport_validated_common_operator"] = not channel_resolved
    mode_path.write_text(json.dumps(driver, indent=2))

    source_path = root / "v10_1_1_source_model.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    source.update(
        {
            "schema": "v10.1.7.5_anisotropic_transport_comparison",
            "transport_mode": selected,
            "transport": (
                "channel-resolved Peierls-Taylor stresses, rates, and velocities"
                if channel_resolved
                else "validated common Peierls-Taylor transport with anisotropic emission"
            ),
            "source_budget_changed_between_transport_modes": False,
            "geometry_refresh_changed_between_transport_modes": False,
        }
    )
    source_path.write_text(json.dumps(source, indent=2))

    audit_path = root / "anisotropic_emission_audit_v10174.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit["transport_comparison"] = transport_payload
        audit_path.write_text(json.dumps(audit, indent=2))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    mode = normalize_transport_mode(os.environ.get("ANISOTROPIC_TRANSPORT_MODE"))

    original_install = _anisotropic.install_anisotropic_campaign_emission
    engine = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_diagnostics = engine._anisotropic_diagnostics

    _anisotropic.install_anisotropic_campaign_emission = make_transport_installer(
        original_install,
        mode,
    )
    engine._anisotropic_diagnostics = make_diagnostics_wrapper(
        original_diagnostics,
        mode,
    )

    try:
        print(
            "  v10.1.7.5 transport comparison: "
            f"mode={mode} "
            "emission=tensor_resolved_campaign_budget "
            f"channel_transport={int(mode == CHANNEL_RESOLVED_TRANSPORT)}"
        )
        result = _v10174.main(args)
        _rewrite_outputs(args, mode)
        return result
    finally:
        _anisotropic.install_anisotropic_campaign_emission = original_install
        engine._anisotropic_diagnostics = original_diagnostics


if __name__ == "__main__":
    main()
