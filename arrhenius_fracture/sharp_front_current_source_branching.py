"""Current-campaign material adapter for the production multi-front overlay.

The physical engine, signed transport, and material loading are taken from the
canonical PF transfer stack.  The topology transaction overlay is the qualified
atomic production implementation ported onto this clean child branch; the
historical V11 executable and its historical material row are never invoked.
"""
from __future__ import annotations

from pathlib import Path
import sys

from . import sharp_front_v11_branching as topology


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv"
SELECTION = ROOT / "runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json"
OPTION = "v913_paper_weakT01_0129902_persistent_sites"
CANDIDATE = "oneD_v2_focused_weak_T_0016"
MODEL_ID = "pf.current_source.signed_dislocation_atomic_multifront/1"


def main(argv=None):
    # Keep the production-lineage import process-local to execution. Importing
    # the entry module for validation/provenance must not install its legacy
    # module-level process-source overlays into an unrelated caller.
    from . import sharp_front_v10_2_27 as paper

    if not REGISTRY.is_file() or not SELECTION.is_file():
        raise SystemExit("current-source branching transfer inputs are missing")
    args = list(sys.argv[1:] if argv is None else argv)
    # Validate the requested row without changing the forwarded arguments.
    requested = None
    for index, token in enumerate(args):
        if token.startswith("--parameter-option="):
            requested = token.split("=", 1)[1]
        elif token == "--parameter-option" and index + 1 < len(args):
            requested = args[index + 1]
    if requested != OPTION:
        raise SystemExit(f"current-source branching requires --parameter-option {OPTION}")

    original_registry = paper.DEFAULT_REGISTRY
    original_selection = paper.SELECTION_RECORD
    original_options = paper.VALID_OPTIONS
    paper.DEFAULT_REGISTRY = REGISTRY
    paper.SELECTION_RECORD = SELECTION
    paper.VALID_OPTIONS = {
        "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
        "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
        OPTION: CANDIDATE,
        "v913_paper_ceramic01_0077080_persistent_sites": "oneD_v2_focused_ceramic_like_0018",
    }
    try:
        return topology.main(args)
    finally:
        paper.DEFAULT_REGISTRY = original_registry
        paper.SELECTION_RECORD = original_selection
        paper.VALID_OPTIONS = original_options


if __name__ == "__main__":
    raise SystemExit(main())
