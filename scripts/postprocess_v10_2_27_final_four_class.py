#!/usr/bin/env python3
"""Run the v10.2.27 K, direct-J, and energy postprocessors with final labels."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OPTION_ORDER = (
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0129902_persistent_sites",
    "v913_paper_ceramic01_0077080_persistent_sites",
)
SHORT_LABELS = {
    OPTION_ORDER[0]: "Peak 0242980",
    OPTION_ORDER[1]: "DBTT 0202500",
    OPTION_ORDER[2]: "Weak-T/FCC-like 0129902",
    OPTION_ORDER[3]: "Ceramic-like 0077080",
}
SUPERSEDED_OPTIONS = (
    "v913_paper_weakT01_0257068_persistent_sites",
    "v913_paper_ceramic01_0189364_persistent_sites",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEFAULT_OPTION_ORDER = OPTION_ORDER
    module.SHORT_LABELS = SHORT_LABELS
    return module


def _run(module, argv: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [str(module.__file__), *argv]
        result = module.main()
    finally:
        sys.argv = previous
    if result not in (None, 0):
        raise SystemExit(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True)
    parser.add_argument("--target-extension-um", type=float, default=1000.0)
    args = parser.parse_args()

    outroot = Path(args.outroot).expanduser().resolve()
    if not outroot.is_dir():
        raise FileNotFoundError(outroot)
    present_superseded = [name for name in SUPERSEDED_OPTIONS if (outroot / name).exists()]
    if present_superseded:
        raise RuntimeError(
            "archive superseded option directories outside the campaign root before "
            f"postprocessing: {present_superseded}"
        )
    manifest_path = outroot / "v10_2_27_campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if tuple(manifest.get("options", [])) != OPTION_ORDER:
        raise RuntimeError("campaign manifest does not contain the final four options")

    rcurve = _load(
        ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py",
        "v10227_final_rcurves",
    )
    ktemp = _load(
        ROOT / "scripts" / "plot_v10_2_27_paper_four_class_K_vs_temperature.py",
        "v10227_final_ktemp",
    )
    jenergy = _load(
        ROOT / "scripts" / "plot_v10_2_27_paper_four_class_J_energy_vs_temperature.py",
        "v10227_final_jenergy",
    )

    target = f"{args.target_extension_um:g}"
    _run(
        rcurve,
        [
            "--outroot",
            str(outroot),
            "--plot-dir",
            str(outroot / "plots"),
            "--target-extension-um",
            target,
        ],
    )
    _run(
        ktemp,
        [
            "--outroot",
            str(outroot),
            "--plot-dir",
            str(outroot / "plots" / "K_vs_temperature"),
            "--target-extension-um",
            target,
        ],
    )
    _run(
        jenergy,
        [
            "--outroot",
            str(outroot),
            "--plot-dir",
            str(outroot / "plots" / "J_energy_vs_temperature"),
            "--target-extension-um",
            target,
        ],
    )
    print(f"Final four-class K/J/energy postprocessing accepted: {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
