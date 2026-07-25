"""Four-class paper campaign on the audited v10.2.22 persistent-site model."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from . import sharp_front_v10_2_22 as _base

MODEL_ID = "v10.2.27_paper_four_class_persistent_sites_physical_width"
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "data" / "materials" / "v10_2_27_paper_four_class_registry.csv"
SELECTION_RECORD = Path(__file__).resolve().parent / "data" / "materials" / "v10_2_27_paper_four_class_selection.json"
VALID_OPTIONS = {
    "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
    "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
    "v913_paper_weakT01_0129902_persistent_sites": "v913_zeroD_sobol_0129902",
    "v913_paper_ceramic01_0077080_persistent_sites": "v913_zeroD_sobol_0077080",
}
PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine
_SOURCE_SELECT_OPTION = _base.select_option


def _select_option_four_class(*args, **kwargs):
    selected = _SOURCE_SELECT_OPTION(*args, **kwargs)
    source_class = selected.material_class.strip().lower()
    if source_class not in {"peak", "dbtt", "weakt", "ceramic"}:
        raise ValueError(f"unexpected v10.2.27 material class: {selected.material_class!r}")
    if source_class == "dbtt":
        return selected
    return replace(selected, material_class="DBTT")


def _option_value(args: list[str], option: str, default: str | None = None) -> str | None:
    prefix = option + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and index + 1 < len(args):
            return args[index + 1]
    return default


def _remove_value_option(args: list[str], option: str) -> None:
    prefix = option + "="
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith(prefix):
            del args[index]
            continue
        if token == option:
            del args[index]
            if index < len(args):
                del args[index]
            continue
        index += 1


def _append_cli_mechanics(command: list[str], args: list[str]) -> None:
    mappings = (
        ("--mpz-length-um", "--process-zone-length-um", 1.0),
        ("--mpz-n-bins", "--process-zone-bins", 1.0),
        ("--nx", "--mesh-nx", 1.0),
        ("--ny", "--mesh-ny", 1.0),
        ("--tip-h-fine", "--tip-h-fine-um", 1.0e6),
        ("--tip-ratio", "--tip-ratio", 1.0),
        ("--da-phys", "--da-phys-um", 1.0e6),
    )
    integers = {"--process-zone-bins", "--mesh-nx", "--mesh-ny"}
    for source, destination, scale in mappings:
        raw = _option_value(args, source)
        if raw is None:
            continue
        value = (
            str(int(round(float(raw))))
            if destination in integers
            else f"{float(raw) * scale:.17g}"
        )
        command.extend([destination, value])


def _install_resolved_geometry(family: Path) -> None:
    payload = json.loads(family.read_text())
    configuration = payload.get("mechanical_configuration")
    if not isinstance(configuration, dict):
        raise SystemExit(
            "resolved signed-kernel family lacks explicit mechanical configuration"
        )
    mapping = {
        "specimen_length_x_m": "V10227_SPECIMEN_LX_M",
        "specimen_length_y_m": "V10227_SPECIMEN_LY_M",
        "initial_crack_length_m": "V10227_INITIAL_CRACK_LENGTH_M",
        "notch_half_thickness_m": "V10227_NOTCH_HALF_THICKNESS_M",
    }
    for key, environment_name in mapping.items():
        value = configuration.get(key)
        if value is None:
            raise SystemExit(f"resolved mechanical configuration lacks {key}")
        os.environ[environment_name] = f"{float(value):.17g}"
    os.environ["V10227_MECHANICAL_CONFIGURATION_FINGERPRINT"] = str(
        payload.get("mechanical_configuration_fingerprint", "")
    )


def _resolve_signed_kernel(args: list[str]) -> tuple[str, bool]:
    supplied = _option_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    _remove_value_option(args, "--signed-kernel-family")

    supplied_path = Path(supplied).expanduser() if supplied else None
    mechanical_config = os.environ.get("MECHANICAL_CONFIG", "").strip()
    if not mechanical_config and supplied_path is not None and supplied_path.is_file():
        sibling = supplied_path.resolve().parent / "mechanical_configuration.json"
        if sibling.is_file():
            mechanical_config = str(sibling)

    official_profile = os.environ.get("PARAMETER_CAMPAIGN", "0") == "1"
    if not mechanical_config and not official_profile:
        raise SystemExit(
            "automatic v10.2.27 kernel resolution outside the fixed paper runner requires "
            "MECHANICAL_CONFIG. This prevents an arbitrary invocation from being treated "
            "as the default specimen/mesh/crack configuration."
        )

    theta = float(_option_value(args, "--crystal-theta-deg", os.environ.get("THETA", "30")))
    target = float(
        _option_value(
            args,
            "--target-crack-extension-um",
            os.environ.get("TARGET_EXT_UM", "1000"),
        )
    )
    maximum_fronts = int(_option_value(args, "--max-fronts", "1"))
    branching_mode = (
        "topology_cached"
        if "--crystal-branch" in args or maximum_fronts > 1
        else "single_front"
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "ensure_v10_2_27_signed_kernel.py"),
        "--theta-deg",
        f"{theta:.17g}",
        "--target-extension-um",
        f"{target:.17g}",
        "--branching-mode",
        branching_mode,
        "--maximum-fronts",
        str(maximum_fronts),
        "--mode",
        os.environ.get("KERNEL_RESOLUTION_MODE", "auto"),
    ]
    _append_cli_mechanics(command, args)
    mechanical_profile = os.environ.get("MECHANICAL_PROFILE", "").strip()
    if mechanical_profile:
        command.extend(["--mechanical-profile", mechanical_profile])
    if mechanical_config:
        command.extend(["--mechanical-config", mechanical_config])
    optional = (
        ("KERNEL_BUILD_COMMAND", "--builder-command"),
        ("KERNEL_SNAPSHOT_ARCHIVE", "--snapshot-archive"),
        ("KERNEL_LOAD_INVARIANCE_ARCHIVE", "--load-invariance-archive"),
        ("KERNEL_ATLAS_ANCHOR_SPACING_UM", "--atlas-anchor-spacing-um"),
        ("KERNEL_INTERACTION_LENGTH_UM", "--interaction-length-um"),
        ("KERNEL_MIN_ELEMENTS_PER_PZ", "--minimum-elements-per-process-zone"),
        ("SPECIMEN_LENGTH_X_UM", "--specimen-length-x-um"),
        ("SPECIMEN_LENGTH_Y_UM", "--specimen-length-y-um"),
        ("INITIAL_CRACK_LENGTH_UM", "--initial-crack-length-um"),
        ("NOTCH_HALF_THICKNESS_UM", "--notch-half-thickness-um"),
    )
    for environment_name, command_option in optional:
        value = os.environ.get(environment_name, "").strip()
        if value:
            command.extend([command_option, value])
    if supplied_path is not None:
        if supplied_path.is_file():
            command.extend(["--family-override", str(supplied_path.resolve())])
        elif os.environ.get("KERNEL_STRICT_FAMILY_OVERRIDE", "0") == "1":
            raise SystemExit(f"explicit signed-kernel family is missing: {supplied_path}")
        else:
            print(
                f"Ignoring stale signed-kernel path; recalculating if needed: {supplied_path}",
                file=sys.stderr,
            )

    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(
            "automatic signed-kernel resolution failed:\n"
            + completed.stdout
            + completed.stderr
        )
    family_text = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    family = Path(family_text).expanduser()
    if not family.is_file():
        raise SystemExit(f"kernel resolver did not return a valid family path: {family_text!r}")
    family = family.resolve()
    _install_resolved_geometry(family)
    args.extend(["--signed-kernel-family", str(family)])
    os.environ["SIGNED_KERNEL_FAMILY_JSON"] = str(family)
    automatically = not bool(supplied_path is not None and supplied_path.is_file())
    return str(family), automatically


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not DEFAULT_REGISTRY.is_file() or not SELECTION_RECORD.is_file():
        raise FileNotFoundError(
            "missing generated v10.2.27 registry or selection record; run "
            "python scripts/install_v10_2_27_four_class_registry.py"
        )
    campaign = json.loads(SELECTION_RECORD.read_text())
    installed_order = campaign.get("canonical_option_order")
    expected_order = list(VALID_OPTIONS)
    if installed_order != expected_order:
        raise RuntimeError(f"v10.2.27 option order mismatch: {installed_order!r} != {expected_order!r}")

    resolved_family, resolved_automatically = _resolve_signed_kernel(args)

    original_registry = _base.DEFAULT_REGISTRY
    original_options = _base.VALID_OPTIONS
    original_model_id = _base.MODEL_ID
    original_engine = _base.PersistentSiteStateResolvedTipEngine
    original_select_option = _base.select_option
    _base.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    _base.VALID_OPTIONS = VALID_OPTIONS
    _base.MODEL_ID = MODEL_ID
    _base.PersistentSiteStateResolvedTipEngine = PersistentSiteStateResolvedTipEngine
    _base.select_option = _select_option_four_class
    try:
        result = _base.main(args)
        out = _base._base._option_value(args, "--out")
        if out:
            root = Path(out)
            selection_path = root / "v10_2_22_parameter_selection.json"
            parameter_selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
            selected_candidate = parameter_selection.get("candidate_id")
            selected_metadata = next(
                (row for row in campaign.get("primary_candidates", []) if row.get("candidate_id") == selected_candidate),
                None,
            )
            exact_row = parameter_selection.get("exact_registry_row") or {}
            payload = {
                "schema": MODEL_ID,
                "base_entry": "arrhenius_fracture.sharp_front_v10_2_22",
                "parameter_transfer_only": True,
                "selected_option": parameter_selection.get("option_key"),
                "selected_candidate": selected_candidate,
                "paper_campaign_selection": selected_metadata,
                "paper_campaign_selection_record": str(SELECTION_RECORD),
                "parameter_registry": str(DEFAULT_REGISTRY),
                "parameter_registry_sha256": campaign.get("installed_registry_sha256"),
                "source_active_parameter_fingerprint_sha256": campaign.get(
                    "source_active_parameter_fingerprint_sha256"
                ),
                "source_material_class": exact_row.get("material_class"),
                "base_loader_material_class": parameter_selection.get("material_class"),
                "base_loader_class_normalization_only": True,
                "mechanics_changed": False,
                "source_closure_changed": False,
                "stochastic_cleavage_law_changed": False,
                "persistent_sites": True,
                "finite_source_inventory": False,
                "source_depletion_on_emission": False,
                "source_refresh": False,
                "explicit_recovery": False,
                "front_width_grid_independent": True,
                "signed_kernel_resolved_automatically": resolved_automatically,
                "signed_kernel_family": resolved_family,
                "mechanical_configuration_fingerprint": os.environ.get(
                    "V10227_MECHANICAL_CONFIGURATION_FINGERPRINT"
                ),
            }
            (root / "v10_2_27_paper_four_class_parameter_transfer.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _base.DEFAULT_REGISTRY = original_registry
        _base.VALID_OPTIONS = original_options
        _base.MODEL_ID = original_model_id
        _base.PersistentSiteStateResolvedTipEngine = original_engine
        _base.select_option = original_select_option


if __name__ == "__main__":
    main()
