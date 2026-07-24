"""Weak-T/FCC-like and ceramic-like paper rows on the audited v10.2.22 model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _base


MODEL_ID = "v10.2.26_v913_weakT_ceramic_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_26_v913_weakT_ceramic_registry.csv"
)
SELECTION_RECORD = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_26_v913_weakT_ceramic_selection.json"
)
VALID_OPTIONS = {
    "v913_paper_weakT01_0257068_persistent_sites": "v913_zeroD_sobol_0257068",
    "v913_paper_ceramic01_0189364_persistent_sites": "v913_zeroD_sobol_0189364",
}

PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine


def main(argv=None):
    """Run v10.2.22 unchanged except for the selected installed parameter row."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not DEFAULT_REGISTRY.is_file():
        raise FileNotFoundError(
            f"missing installed v10.2.26 registry: {DEFAULT_REGISTRY}; run "
            "scripts/install_v913_weakT_ceramic_handoff.py first"
        )
    if not SELECTION_RECORD.is_file():
        raise FileNotFoundError(
            f"missing installed v10.2.26 selection record: {SELECTION_RECORD}; run "
            "scripts/install_v913_weakT_ceramic_handoff.py first"
        )

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
            parameter_selection_path = root / "v10_2_22_parameter_selection.json"
            parameter_selection = (
                json.loads(parameter_selection_path.read_text())
                if parameter_selection_path.is_file()
                else {}
            )
            campaign = json.loads(SELECTION_RECORD.read_text())
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
            (root / "v10_2_26_v913_weakT_ceramic_parameter_transfer.json").write_text(
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
