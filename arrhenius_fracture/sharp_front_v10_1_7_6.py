"""Four-option parameter overlay on the unchanged v10.1.7.5 monotonic 2-D model.

This entry adds no mechanics, kernel, source, shielding, transport, or geometry
model.  It selects one exact v9.11.1 row, writes it in the existing material
manifest format, applies that row's MPZ grid, and invokes v10.1.7.5.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# These are the production stochastic defaults developed in v10.1.7.2/3.
os.environ.setdefault("CLEAVAGE_HAZARD_MODE", "exponential")
os.environ.setdefault("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
os.environ.setdefault("CLEAVAGE_EVENT_MIN_FACTOR", "0.5")
os.environ.setdefault("CLEAVAGE_EVENT_MAX_FACTOR", "4.0")
os.environ.setdefault("CLEAVAGE_EVENT_SUBSEGMENT_FRACTION", "0.1")
os.environ.setdefault("ANISOTROPIC_USE_AVALANCHE_BACKEND", "1")
os.environ.setdefault("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")

from . import sharp_front_v10_1_7_3 as _avalanche  # noqa: E402
from . import sharp_front_v10_1_7_5 as _working_2d  # noqa: E402
from .parameter_registry_v9111 import (  # noqa: E402
    default_registry_path,
    select_option,
    write_material_manifest,
    write_selection_audit,
)

MODEL_ID = "v10.1.7.6_four_option_stochastic_parameter_overlay"
WORKING_ENTRY = "arrhenius_fracture.sharp_front_v10_1_7_5"


def _pop_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return default


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _remove_value(args: list[str], name: str) -> None:
    prefix = name + "="
    kept: list[str] = []
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(prefix):
            continue
        kept.append(token)
    args[:] = kept


def _set_value(args: list[str], name: str, value) -> None:
    _remove_value(args, name)
    args.extend([name, str(value)])


def _require_stochastic_configuration() -> int:
    hazard = os.environ.get("CLEAVAGE_HAZARD_MODE", "").strip().lower()
    event = os.environ.get("CLEAVAGE_EVENT_LENGTH_MODE", "").strip().lower()
    if hazard != "exponential":
        raise SystemExit(f"v10.1.7.6 requires CLEAVAGE_HAZARD_MODE=exponential; got {hazard!r}")
    if event != "threshold_scaled":
        raise SystemExit(
            f"v10.1.7.6 requires CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled; got {event!r}"
        )
    raw_seed = os.environ.get("CLEAVAGE_HAZARD_SEED", "").strip()
    if not raw_seed:
        raise SystemExit("v10.1.7.6 requires explicit CLEAVAGE_HAZARD_SEED")
    return int(raw_seed)


def _zero_event_safe_summary(args: list[str]) -> None:
    """Preserve a valid no-fracture run as right-censored postprocessing output."""
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    summary_path = root / "summary.json"
    if not geometry_path.is_file() or not summary_path.is_file():
        return
    geometry = json.loads(geometry_path.read_text())
    if geometry:
        _ORIGINAL_REWRITE(args)
        return
    summary = json.loads(summary_path.read_text())
    if isinstance(summary, list) and summary:
        summary[0].update(
            {
                "n_geometry_events": 0,
                "n_equivalent_checkpoints_exact": 0.0,
                "n_equivalent_checkpoints_rounded": 0,
                "geometry_path_length_m": 0.0,
                "geometry_projected_extension_m": 0.0,
                "n_advances_semantics": "rounded_path_length_over_nominal_checkpoint",
                "n_geometry_events_semantics": "accepted_cleavage_renewals_and_geometry_commits",
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


_ORIGINAL_REWRITE = _avalanche._rewrite_summary_event_semantics


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    seed = _require_stochastic_configuration()
    out_value = _option_value(args, "--out")
    if not out_value:
        raise SystemExit("v10.1.7.6 requires --out")
    option_key = _pop_value(args, "--parameter-option", os.environ.get("PARAMETER_OPTION"))
    if not option_key:
        raise SystemExit("v10.1.7.6 requires --parameter-option")
    registry = _pop_value(
        args,
        "--parameter-registry",
        os.environ.get("PARAMETER_REGISTRY", str(default_registry_path())),
    )
    selected = select_option(option_key, registry)
    root = Path(out_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = write_material_manifest(selected, root / "selected_material_manifest_v9_11_1.csv")
    write_selection_audit(selected, root / "v10_1_7_6_parameter_selection.json", manifest)
    _remove_value(args, "--material-class")
    _set_value(args, "--material-manifest", manifest)
    _set_value(args, "--mpz-length-um", selected.mpz_length_um)
    _set_value(args, "--mpz-n-bins", selected.mpz_n_bins)

    original_rewrite = _avalanche._rewrite_summary_event_semantics
    _avalanche._rewrite_summary_event_semantics = _zero_event_safe_summary
    try:
        print(
            "  v10.1.7.6 parameter overlay: "
            f"entry={WORKING_ENTRY} option={selected.option_key} "
            f"candidate={selected.candidate_id} "
            f"mpz={selected.mpz_length_um:g}um/{selected.mpz_n_bins}bins "
            f"hazard=exponential event_length=threshold_scaled seed={seed}"
        )
        result = _working_2d.main(args)
    finally:
        _avalanche._rewrite_summary_event_semantics = original_rewrite

    audit = {
        "schema": MODEL_ID,
        "working_2d_entry": WORKING_ENTRY,
        "parameter_overlay_only": True,
        "mechanics_or_kernel_promotion": False,
        "atlas_used": False,
        "source_model_changed": False,
        "shielding_model_changed": False,
        "anisotropic_emission_changed": False,
        "transport_changed": False,
        "geometry_backend_changed": False,
        "cleavage_hazard_mode": "exponential",
        "cleavage_event_length_mode": "threshold_scaled",
        "cleavage_hazard_seed": seed,
        "selected_option": selected.option_key,
        "candidate_id": selected.candidate_id,
        "mpz_length_um": selected.mpz_length_um,
        "mpz_n_bins": selected.mpz_n_bins,
    }
    (root / "v10_1_7_6_parameter_overlay_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    main()
