"""v10.2.27 four-class physics with the v10.2.28 direct kernel provider."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from . import sharp_front_v10_2_27 as _base

MODEL_ID = "v10.2.28_paper_four_class_direct_prescribed_geometry_kernel"


def _resolve_signed_kernel(args: list[str]) -> tuple[str, bool]:
    supplied = _base._option_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    _base._remove_value_option(args, "--signed-kernel-family")

    supplied_path = Path(supplied).expanduser() if supplied else None
    mechanical_config = os.environ.get("MECHANICAL_CONFIG", "").strip()
    if not mechanical_config and supplied_path is not None and supplied_path.is_file():
        sibling = supplied_path.resolve().parent / "mechanical_configuration.json"
        if sibling.is_file():
            mechanical_config = str(sibling)

    official_profile = os.environ.get("PARAMETER_CAMPAIGN", "0") == "1"
    if not mechanical_config and not official_profile:
        raise SystemExit(
            "automatic v10.2.28 kernel resolution outside the fixed paper runner "
            "requires MECHANICAL_CONFIG"
        )

    theta = float(
        _base._option_value(args, "--crystal-theta-deg", os.environ.get("THETA", "30"))
    )
    target = float(
        _base._option_value(
            args,
            "--target-crack-extension-um",
            os.environ.get("TARGET_EXT_UM", "1000"),
        )
    )
    maximum_fronts = int(_base._option_value(args, "--max-fronts", "1"))
    branching_mode = (
        "topology_cached"
        if "--crystal-branch" in args or maximum_fronts > 1
        else "single_front"
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts" / "ensure_v10_2_28_signed_kernel.py"),
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
    if not mechanical_config:
        _base._append_cli_mechanics(command, args)
    mechanical_profile = os.environ.get("MECHANICAL_PROFILE", "").strip()
    if mechanical_profile and not mechanical_config:
        command.extend(["--mechanical-profile", mechanical_profile])
    if mechanical_config:
        command.extend(["--mechanical-config", mechanical_config])

    optional = (
        ("KERNEL_BUILD_COMMAND", "--builder-command"),
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
    forbidden = (
        "KERNEL_CAPTURE_SEED_FAMILY",
        "KERNEL_CAPTURE_PARAMETER_OPTION",
        "KERNEL_CAPTURE_HAZARD_SEED",
        "KERNEL_SNAPSHOT_ARCHIVE",
        "KERNEL_LOAD_INVARIANCE_ARCHIVE",
    )
    supplied_forbidden = [name for name in forbidden if os.environ.get(name, "").strip()]
    if supplied_forbidden:
        raise SystemExit(
            "v10.2.28 direct provider forbids legacy bootstrap/capture inputs: "
            + ", ".join(supplied_forbidden)
        )
    if supplied_path is not None:
        if supplied_path.is_file():
            command.extend(["--family-override", str(supplied_path.resolve())])
        elif os.environ.get("KERNEL_STRICT_FAMILY_OVERRIDE", "0") == "1":
            raise SystemExit(f"explicit signed-kernel family is missing: {supplied_path}")
        else:
            print(
                f"Ignoring stale signed-kernel path; direct recalculation may follow: {supplied_path}",
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
            "automatic direct signed-kernel resolution failed:\n"
            + completed.stdout
            + completed.stderr
        )
    family_text = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    family = Path(family_text).expanduser()
    if not family.is_file():
        raise SystemExit(f"direct kernel resolver returned invalid family path: {family_text!r}")
    family = family.resolve()
    _base._install_resolved_mechanics(family, args)
    args.extend(["--signed-kernel-family", str(family)])
    os.environ["SIGNED_KERNEL_FAMILY_JSON"] = str(family)
    automatically = not bool(supplied_path is not None and supplied_path.is_file())
    return str(family), automatically


def main(argv=None):
    original_resolver = _base._resolve_signed_kernel
    original_model_id = _base.MODEL_ID
    _base._resolve_signed_kernel = _resolve_signed_kernel
    _base.MODEL_ID = MODEL_ID
    try:
        return _base.main(argv)
    finally:
        _base._resolve_signed_kernel = original_resolver
        _base.MODEL_ID = original_model_id


if __name__ == "__main__":
    main()
