"""Exact v9.13 top-ten parameter transfer onto the audited v10.2.22 2-D model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _base


MODEL_ID = "v10.2.23_v913_top10_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_23_v913_top10_persistent_site_registry.csv"
)
VALID_OPTIONS = {
    "v913_top01_0091348_persistent_sites": "v913_zeroD_sobol_0091348",
    "v913_top02_0086763_persistent_sites": "v913_zeroD_sobol_0086763",
    "v913_top03_0115460_persistent_sites": "v913_zeroD_sobol_0115460",
    "v913_top04_0097332_persistent_sites": "v913_zeroD_sobol_0097332",
    "v913_top05_0127508_persistent_sites": "v913_zeroD_sobol_0127508",
    "v913_top06_0116955_persistent_sites": "v913_zeroD_sobol_0116955",
    "v913_top07_0060219_persistent_sites": "v913_zeroD_sobol_0060219",
    "v913_top08_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
    "v913_top09_0103251_persistent_sites": "v913_zeroD_sobol_0103251",
    "v913_top10_0080699_persistent_sites": "v913_zeroD_sobol_0080699",
}

# The audited wrapper temporarily replaces this symbol before calling main().
PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine


def main(argv=None):
    """Run v10.2.22 unchanged except for the exact selected v9.13 row."""
    args = list(sys.argv[1:] if argv is None else argv)
    original_registry = _base.DEFAULT_REGISTRY
    original_options = _base.VALID_OPTIONS
    original_model_id = _base.MODEL_ID
    original_engine = _base.PersistentSiteStateResolvedTipEngine
    _base.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    _base.VALID_OPTIONS = VALID_OPTIONS
    _base.MODEL_ID = MODEL_ID
    _base.PersistentSiteStateResolvedTipEngine = PersistentSiteStateResolvedTipEngine
    try:
        result = _base.main(args)
        out = _base._base._option_value(args, "--out")
        if out:
            root = Path(out)
            selection_path = root / "v10_2_22_parameter_selection.json"
            selection = (
                json.loads(selection_path.read_text())
                if selection_path.is_file()
                else {}
            )
            payload = {
                "schema": MODEL_ID,
                "base_entry": "arrhenius_fracture.sharp_front_v10_2_22",
                "parameter_transfer_only": True,
                "selected_option": selection.get("option_key"),
                "selected_candidate": selection.get("candidate_id"),
                "parameter_registry": str(DEFAULT_REGISTRY),
                "mechanics_changed": False,
                "source_closure_changed": False,
                "stochastic_cleavage_law_changed": False,
                "persistent_sites": True,
                "finite_source_inventory": False,
                "source_refresh": False,
                "explicit_recovery": False,
                "front_width_grid_independent": True,
            }
            (root / "v10_2_23_v913_parameter_transfer.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _base.DEFAULT_REGISTRY = original_registry
        _base.VALID_OPTIONS = original_options
        _base.MODEL_ID = original_model_id
        _base.PersistentSiteStateResolvedTipEngine = original_engine


if __name__ == "__main__":
    main()
