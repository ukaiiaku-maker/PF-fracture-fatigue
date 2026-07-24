"""Exact v9.13 upper-shelf top-ten transfer onto the audited v10.2.22 2-D model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _base


MODEL_ID = "v10.2.24_v913_upper_shelf_top10_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_24_v913_top10_upper_shelf_registry.csv"
)
SELECTION_RECORD = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_24_v913_upper_shelf_selection.json"
)
VALID_OPTIONS = {
    "v913_shelf01_0086420_persistent_sites": "v913_zeroD_sobol_0086420",
    "v913_shelf02_0009771_persistent_sites": "v913_zeroD_sobol_0009771",
    "v913_shelf03_0088403_persistent_sites": "v913_zeroD_sobol_0088403",
    "v913_shelf04_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
    "v913_shelf05_0196804_persistent_sites": "v913_zeroD_sobol_0196804",
    "v913_shelf06_0162507_persistent_sites": "v913_zeroD_sobol_0162507",
    "v913_shelf07_0027268_persistent_sites": "v913_zeroD_sobol_0027268",
    "v913_shelf08_0011131_persistent_sites": "v913_zeroD_sobol_0011131",
    "v913_shelf09_0073460_persistent_sites": "v913_zeroD_sobol_0073460",
    "v913_shelf10_0134035_persistent_sites": "v913_zeroD_sobol_0134035",
}

PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine


def main(argv=None):
    """Run v10.2.22 unchanged except for the selected upper-shelf row."""
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
            shelf_selection = json.loads(SELECTION_RECORD.read_text())
            selected_id = selection.get("candidate_id")
            selected_metadata = next(
                (
                    row
                    for row in shelf_selection.get("selected", [])
                    if row.get("candidate_id") == selected_id
                ),
                None,
            )
            payload = {
                "schema": MODEL_ID,
                "base_entry": "arrhenius_fracture.sharp_front_v10_2_22",
                "parameter_transfer_only": True,
                "response_class": "directional_dbtt_upper_shelf_non_peak",
                "selected_option": selection.get("option_key"),
                "selected_candidate": selected_id,
                "upper_shelf_selection": selected_metadata,
                "upper_shelf_selection_record": str(SELECTION_RECORD),
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
            (root / "v10_2_24_v913_upper_shelf_parameter_transfer.json").write_text(
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
