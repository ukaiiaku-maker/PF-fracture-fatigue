"""Final four-class paper registry on the audited v10.2.22 mechanics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _base


MODEL_ID = "v10.2.27_v913_four_class_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_27_v913_four_class_paper_registry.csv"
)
SELECTION_RECORD = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_27_v913_four_class_paper_selection.json"
)


def load_valid_options(path: Path = DEFAULT_REGISTRY) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4:
        raise RuntimeError(f"v10.2.27 registry must contain four rows, found {len(rows)}")
    options = {str(row["option_key"]): str(row["candidate_id"]) for row in rows}
    if len(options) != 4 or len(set(options.values())) != 4:
        raise RuntimeError("v10.2.27 registry options and candidate IDs must be unique")
    return options


VALID_OPTIONS = load_valid_options()
PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine


def main(argv=None):
    """Run v10.2.22 unchanged with one of the four final paper rows."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not DEFAULT_REGISTRY.is_file() or not SELECTION_RECORD.is_file():
        raise FileNotFoundError(
            "missing installed v10.2.27 registry/selection; run "
            "scripts/install_v913_four_class_paper_selection.py first"
        )
    valid_options = load_valid_options(DEFAULT_REGISTRY)
    campaign = json.loads(SELECTION_RECORD.read_text())
    if campaign.get("candidate_count") != 4:
        raise RuntimeError("v10.2.27 selection record must contain four candidates")

    original_registry = _base.DEFAULT_REGISTRY
    original_options = _base.VALID_OPTIONS
    original_model_id = _base.MODEL_ID
    original_engine = _base.PersistentSiteStateResolvedTipEngine
    _base.DEFAULT_REGISTRY = DEFAULT_REGISTRY
    _base.VALID_OPTIONS = valid_options
    _base.MODEL_ID = MODEL_ID
    _base.PersistentSiteStateResolvedTipEngine = PersistentSiteStateResolvedTipEngine
    try:
        result = _base.main(args)
        out = _base._base._option_value(args, "--out")
        if out:
            root = Path(out)
            parameter_selection_path = root / "v10_2_22_parameter_selection.json"
            parameter_selection = (
                json.loads(parameter_selection_path.read_text())
                if parameter_selection_path.is_file()
                else {}
            )
            selected_candidate = parameter_selection.get("candidate_id")
            selected_metadata = next(
                (
                    row
                    for row in campaign.get("primary_candidates", [])
                    if row.get("candidate_id") == selected_candidate
                ),
                None,
            )
            payload = {
                "schema": MODEL_ID,
                "base_entry": "arrhenius_fracture.sharp_front_v10_2_22",
                "parameter_transfer_only": True,
                "selected_option": parameter_selection.get("option_key"),
                "selected_candidate": selected_candidate,
                "paper_campaign_selection": selected_metadata,
                "paper_campaign_selection_record": str(SELECTION_RECORD),
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
            (root / "v10_2_27_v913_four_class_parameter_transfer.json").write_text(
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
