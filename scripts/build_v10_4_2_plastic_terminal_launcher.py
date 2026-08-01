#!/usr/bin/env python3
"""Finalize v10.4.2 inherited-case and terminal campaign verification."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_base_builder():
    path = Path(__file__).with_name("build_v10_4_bulk_rate_orientation_launcher.py")
    spec = importlib.util.spec_from_file_location("v1042_base_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.4.2 base builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_base_builder().transform(source)
    marker = '''command = (root / "command.sh").read_text()
if is_plastic_terminal and "--plastic-flow-terminal" not in command:
'''
    replacement = '''v1042_reuse_path = root / "v10_4_2_reuse_audit.json"
if v1042_reuse_path.is_file():
    from arrhenius_fracture.reuse_v1041_v1042 import verify_materialized_case

    verify_materialized_case(root)
elif bulk_model_audit.get("schema") != "v10.4.2_bulk_detailed_balance_plastic_flow_terminal":
    raise SystemExit(1)

command = (root / "command.sh").read_text()
if is_plastic_terminal and "--plastic-flow-terminal" not in command:
'''
    text = _replace_once(
        text,
        marker,
        replacement,
        "v10.4.2 inherited-case hash verification",
    )

    legacy_plot = '''"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
  --outroot "$OUTROOT" \\
  --target-extension-um "$TARGET_EXT_UM" || {
    echo "ERROR: four-class R-curve postprocessing failed" >&2
    exit 1
  }
'''
    terminal_aware_plot = '''if find "$OUTROOT" -type f -name COMPLETE -print -quit | grep -q .; then
  "$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
    --outroot "$OUTROOT" \\
    --target-extension-um "$TARGET_EXT_UM" || {
      echo "ERROR: four-class R-curve postprocessing failed" >&2
      exit 1
    }
else
  echo "No sharp-fracture COMPLETE cases; skipping fracture-only R-curve postprocessing"
fi
'''
    return _replace_once(
        text,
        legacy_plot,
        terminal_aware_plot,
        "v10.4.2 terminal-aware fracture postprocessing",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
