"""Exact V2-row adapter for the accepted v10.2.28 single-crack 2-D stack.

The adapter changes parameter binding only.  FEM mechanics, J evaluation,
stochastic event handling, geometry transactions, and checkpointing remain in
the audited v10.2.28 entrypoint.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

from . import sharp_front as _driver
from . import sharp_front_v10_1_5 as _campaign
from . import sharp_front_v10_2_28_audited as _audited
from .canonical_v2_registry_v10230 import surface_adapters
from .material_manifest import MaterialManifest
from . import parameter_registry_v9111 as _registry
from .v2_named_parameterizations import NAMED_ALIASES, load_exact_candidate, load_for

MODEL_ID = "v10.2.30_v2_named_parameter_single_crack_parameter_adapter_v1"


def _value(args: list[str], option: str) -> str | None:
    prefix = option + "="
    for i, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and i + 1 < len(args):
            return args[i + 1]
    return None


def _force(args: list[str], option: str, value) -> None:
    prefix = option + "="
    i = 0
    while i < len(args):
        if args[i].startswith(prefix):
            del args[i]
        elif args[i] == option:
            del args[i:i + 2]
        else:
            i += 1
    args.extend([option, str(value)])


def _remove(args: list[str], option: str) -> None:
    prefix = option + "="
    i = 0
    while i < len(args):
        if args[i].startswith(prefix):
            del args[i]
        elif args[i] == option:
            del args[i:i + 2]
        else:
            i += 1


def _finite(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"V2 row field {name!r} is missing or nonnumeric") from exc
    if not math.isfinite(value):
        raise SystemExit(f"V2 row field {name!r} is nonfinite")
    return value


def _runtime_registry(out: Path, alias: str, row: dict[str, str]) -> tuple[Path, Path]:
    parent = json.loads(row["parent_complete_registry_row_json"])
    parent["option_key"] = alias
    parent["candidate_id"] = row["candidate_id"]
    parent["role"] = row.get("role", parent.get("role", "prospective V2 transfer"))
    registry = out / "v2_runtime_parameter_registry.csv"
    with registry.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parent))
        writer.writeheader(); writer.writerow(parent)
    selection = out / "v2_runtime_selection.json"
    selection.write_text(json.dumps({
        "schema": MODEL_ID,
        "canonical_option_order": [alias],
        "primary_candidates": [{"candidate_id": row["candidate_id"], "alias": alias}],
        "installed_registry_sha256": None,
    }, indent=2, sort_keys=True) + "\n")
    return registry, selection


def _exact_manifest(base: MaterialManifest, row: dict[str, str]) -> MaterialManifest:
    opening, emission = surface_adapters(row)
    return replace(
        base,
        name=str(row["candidate_id"]),
        candidate_id=str(row["candidate_id"]),
        cleavage=opening,
        emission=emission,
    )


def _select_v2_option(option_key, registry_path, *, canonical_stage3_only=False):
    """Select a complete direct-surface V2 row under its own exact contract."""
    source = Path(registry_path).resolve()
    matches = [r for r in _registry.read_registry(source) if r["option_key"] == option_key]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one V2 runtime row; found {len(matches)}")
    row = matches[0]
    required_exact = {
        "Tref_K": 300.0, "exact_spatial_Tref_active": 1.0,
        "n_slip_channels": 2.0, "rho_forest_floor_m2": 5.0e12,
        "peierls_stress_fraction": 1.0 / math.sqrt(3.0),
        "taylor_stress_fraction": 1.0 / math.sqrt(3.0),
        "mobile_shield_fraction": 0.0, "source_recovery_rate_s": 0.0,
    }
    for name, expected in required_exact.items():
        value = float(row[name])
        if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"V2 spatial contract mismatch for {name}: {value!r}")
    bins = float(row["n_bins_recommended"]); length = float(row["L_pz_um_recommended"])
    if not bins.is_integer() or bins < 4 or not math.isfinite(length) or length <= 0:
        raise ValueError("invalid V2 process-zone discretization")
    return _registry.SelectedResponseOption(
        option_key=option_key, candidate_id=row["candidate_id"].strip(),
        material_class=row["material_class"].strip(), role=row.get("role", "").strip(),
        mechanism_summary=row.get("mechanism_summary", "").strip(),
        validation_status=row.get("validation_status", "").strip(),
        mpz_length_um=length, mpz_n_bins=int(bins), row=dict(row),
        registry_path=str(source), registry_sha256=_registry.sha256_file(source))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    alias = _value(args, "--v2-parameter-alias") or os.environ.get("V2_PARAMETER_ALIAS", "")
    if alias not in NAMED_ALIASES:
        raise SystemExit("--v2-parameter-alias must name one of: " + ", ".join(NAMED_ALIASES))
    _remove(args, "--v2-parameter-alias")
    out_raw = _value(args, "--out")
    if not out_raw:
        raise SystemExit("V2 2-D adapter requires --out")
    out = Path(out_raw).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    record = load_for(alias, "PF_sharp_front")
    exact = load_exact_candidate(record.source_candidate_id)
    if dict(record.full_precision_row) != dict(exact.full_precision_row):
        raise SystemExit("alias and exact-candidate lookup differ")
    row = dict(record.full_precision_row)
    parent_row = json.loads(row["parent_complete_registry_row_json"])
    def physical(name: str) -> float:
        return _finite(row if name in row else parent_row, name)
    if row["complete_bound_row_sha256"] != record.complete_bound_row_sha256:
        raise SystemExit("V2 row hash binding mismatch")
    runtime_registry, runtime_selection = _runtime_registry(out, alias, row)

    _force(args, "--parameter-option", alias)
    _force(args, "--parameter-registry", runtime_registry)
    _force(args, "--multihit-m", format(physical("physics__cleavage_hits"), ".17g"))
    _force(args, "--multihit-tau", format(physical("physics__cleavage_correlation_time_s"), ".17g"))
    _force(args, "--nu0-cleave", format(_finite(row, "opening_attempt_frequency_s"), ".17g"))
    _force(args, "--nu0-emit", format(_finite(row, "emission_attempt_frequency_s"), ".17g"))
    _force(args, "--c-blunt", format(physical("c_blunt"), ".17g"))
    _force(args, "--mpz-blunting-length-um", format(1e6 * physical("physics__blunting_length_m"), ".17g"))
    _force(args, "--kinetic-velocity-scale", format(physical("physics__mobile_transport_velocity_scale"), ".17g"))
    _force(args, "--pt-taylor-phi-max", format(physical("physics__taylor_phi_max"), ".17g"))
    # Consumed by the production registry adapter; retained explicitly in the
    # audit because the current 2-D state law has no independent slip-fraction
    # CLI (c_blunt is its immutable runtime blunting coefficient).
    blunting_slip_fraction = physical("physics__blunting_slip_fraction")

    v227 = _audited._entry._base
    original_registry, original_selection = v227.DEFAULT_REGISTRY, v227.SELECTION_RECORD
    original_options = v227.VALID_OPTIONS
    original_source_selector = v227._SOURCE_SELECT_OPTION
    original_manifest_symbol = _driver.MaterialManifest
    original_backstress = _campaign.BACKSTRESS_SCALE
    bound_factory = MaterialManifest.from_csv

    class ExactV2MaterialManifest(MaterialManifest):
        @classmethod
        def from_csv(cls, path):
            return _exact_manifest(bound_factory(path), row)

    v227.DEFAULT_REGISTRY = runtime_registry
    v227.SELECTION_RECORD = runtime_selection
    v227.VALID_OPTIONS = {alias: record.source_candidate_id}
    v227._SOURCE_SELECT_OPTION = _select_v2_option
    _driver.MaterialManifest = ExactV2MaterialManifest
    _campaign.BACKSTRESS_SCALE = physical("physics__persistent_backstress_scale")
    try:
        result = _audited.main(args)
    finally:
        v227.DEFAULT_REGISTRY, v227.SELECTION_RECORD = original_registry, original_selection
        v227.VALID_OPTIONS = original_options
        v227._SOURCE_SELECT_OPTION = original_source_selector
        _driver.MaterialManifest = original_manifest_symbol
        _campaign.BACKSTRESS_SCALE = original_backstress

    audit = {
        "schema": MODEL_ID, "alias": alias,
        "candidate_id": record.source_candidate_id,
        "complete_bound_row_sha256": record.complete_bound_row_sha256,
        "loader": "load_for(alias, PF_sharp_front)",
        "alias_exact_lookup_byte_identical": True,
        "parameter_binding_only": True,
        "base_entry": "arrhenius_fracture.sharp_front_v10_2_28_audited",
        "runtime": {
            "cleavage_hits": physical("physics__cleavage_hits"),
            "cleavage_correlation_time_s": physical("physics__cleavage_correlation_time_s"),
            "opening_attempt_frequency_s": _finite(row, "opening_attempt_frequency_s"),
            "emission_attempt_frequency_s": _finite(row, "emission_attempt_frequency_s"),
            "c_blunt": physical("c_blunt"),
            "blunting_length_m": physical("physics__blunting_length_m"),
            "persistent_backstress_scale": physical("physics__persistent_backstress_scale"),
            "mobile_transport_velocity_scale": physical("physics__mobile_transport_velocity_scale"),
            "taylor_phi_max": physical("physics__taylor_phi_max"),
        },
        "source_design_metadata": {"blunting_slip_fraction": blunting_slip_fraction},
        "direct_opening_surface": True, "direct_emission_surface": True,
        "direct_surface_reference_temperature_K": 300.0,
        "v2_direct_surface_contract_validated": True,
        "mechanics_changed": False, "event_transaction_changed": False,
    }
    (out / "v2_exact_parameter_binding.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    main()
