"""Four-class paper campaign on the audited v10.2.22 persistent-site model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_22 as _base


MODEL_ID = "v10.2.27_paper_four_class_persistent_sites_physical_width"
DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_27_paper_four_class_registry.csv"
)
SELECTION_RECORD = (
    Path(__file__).resolve().parent
    / "data"
    / "materials"
    / "v10_2_27_paper_four_class_selection.json"
)
VALID_OPTIONS = {
    "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
    "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
    "v913_paper_weakT01_0257068_persistent_sites": "v913_zeroD_sobol_0257068",
    "v913_paper_ceramic01_0189364_persistent_sites": "v913_zeroD_sobol_0189364",
}

PersistentSiteStateResolvedTipEngine = _base.PersistentSiteStateResolvedTipEngine


def main(argv=None):
    """Run v10.2.22 unchanged except for one exact four-class parameter row."""
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
        raise RuntimeError(
            f"v10.2.27 option order mismatch: {installed_order!r} != {expected_order!r}"
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
                "parameter_registry_sha256": campaign.get("installed_registry_sha256"),
                "mechanics_changed": False,
                "source_closure_changed": False,
                "stochastic_cleavage_law_changed": False,
                "persistent_sites": True,
                "finite_source_inventory": False,
                "source_depletion_on_emission": False,
                "source_refresh": False,
                "explicit_recovery": False,
                "front_width_grid_independent": True,
            }
            (
                root / "v10_2_27_paper_four_class_parameter_transfer.json"
            ).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        _base.DEFAULT_REGISTRY = original_registry
        _base.VALID_OPTIONS = original_options
        _base.MODEL_ID = original_model_id
        _base.PersistentSiteStateResolvedTipEngine = original_engine


if __name__ == "__main__":
    main()
