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
    # The inherited v10.2.22 loader exposes only its DBTT mechanics route. Keep the
    # exact paper class in selected.row while normalizing only the legacy loader tag.
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


def _resolve_signed_kernel(args: list[str]) -> None:
    supplied = _option_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    if supplied and Path(supplied).expanduser().is_file():
        return
    if supplied:
        print(
            f"Ignoring stale signed-kernel path and resolving mechanically: {supplied}",
            file=sys.stderr,
        )
    _remove_value_option(args, "--signed-kernel-family")

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
        "--mechanical-profile",
        os.environ.get(
            "MECHANICAL_PROFILE",
            "v10_2_27_default_single_front_frontfix",
        ),
        "--mode",
        os.environ.get("KERNEL_RESOLUTION_MODE", "auto"),
    ]
    optional = (
        ("MECHANICAL_CONFIG", "--mechanical-config"),
        ("KERNEL_BUILD_COMMAND", "--builder-command"),
        ("KERNEL_SNAPSHOT_ARCHIVE", "--snapshot-archive"),
        ("KERNEL_LOAD_INVARIANCE_ARCHIVE", "--load-invariance-archive"),
    )
    for environment_name, command_option in optional:
        value = os.environ.get(environment_name, "").strip()
        if value:
            command.extend([command_option, value])
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
    family = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    if not family or not Path(family).is_file():
        raise SystemExit(f"kernel resolver did not return a valid family path: {family!r}")
    args.extend(["--signed-kernel-family", family])
    os.environ["SIGNED_KERNEL_FAMILY_JSON"] = family


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

    _resolve_signed_kernel(args)

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
                "signed_kernel_resolved_automatically": True,
                "signed_kernel_family": os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
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
