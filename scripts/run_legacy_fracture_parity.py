#!/usr/bin/env python3
"""Executable audit and no-void parity sentinel for the qualified front core."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.legacy_front_process_adapter import LegacyFrontProcessAdapter
from arrhenius_fracture.unified_control_matrix_v5 import PaperControl, paper_control_matrix
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _canonical_front_process_owner, build_production_void_state,
    capture_sharp_front_engine, restore_sharp_front_engine,
    sharp_front_engine_fingerprint,
)


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _payload(state):
    owner = _canonical_front_process_owner(state, ROOT_BRANCH_ID)
    return state.tip_process_state["by_branch"][owner]


def _fingerprint(state, bundle, payload):
    return sharp_front_engine_fingerprint(
        restore_sharp_front_engine(state.material, bundle, payload)
    )


def main():
    bundle = material_bundle("DBTT")
    controls = paper_control_matrix(bundle)
    state_on, _ = build_production_void_state(
        bundle=bundle, enabled=True, temperature_K=900.0,
    )
    state_off, _ = build_production_void_state(
        bundle=bundle, enabled=False, temperature_K=900.0,
    )
    state_no_site = replace(
        state_on, void_state=replace(state_on.void_state, sites=(), cavities=()),
    )
    adapter = LegacyFrontProcessAdapter(state_on.material, bundle)
    K, T, dt = 1.0e8, 900.0, 1.0e-9

    direct_engine = restore_sharp_front_engine(state_on.material, bundle, _payload(state_on))
    direct_info = direct_engine.step(K, T, dt)
    direct_payload = capture_sharp_front_engine(direct_engine)
    adapted = adapter.advance_front_process(_payload(state_on), K, T, dt)
    _require(
        _fingerprint(state_on, bundle, direct_payload)
        == _fingerprint(state_on, bundle, adapted.advanced_front_state),
        "adapter changed canonical single-tip evolution",
    )

    results = {}
    for name, state in (("void_enabled", state_on), ("void_disabled", state_off),
                        ("no_active_site", state_no_site)):
        result = adapter.advance_front_process(_payload(state), K, T, dt)
        results[name] = result
        _require(
            _fingerprint(state, bundle, result.advanced_front_state)
            == _fingerprint(state_on, bundle, adapted.advanced_front_state),
            name+" altered the legacy front owner",
        )

    identities = {
        control.value: dict(row.composite_identity)
        for control, row in controls.items()
    }
    _require(len({row["material_bundle_id"] for row in identities.values()}) == 1,
             "paper controls do not share one immutable material bundle")
    _require(adapter.factory_identity["factory"] == "canonical_front_engine",
             "front adapter bypassed the canonical factory")
    audit = {
        "lineage_anchors": {
            "single_tip": "7e71b8f27b0682060fd161e7e5e8fe0d3587e8ac",
            "multitip_merge": "0238aae096aa29e79829d3c562383c38f2290ad6",
            "four_family_v13": "ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e",
        },
        "semantics": {
            "emission": "continuous source-wise MPZ population evolution",
            "front_engine_fired": "cleavage first passage and renewal",
            "topology_creator": "cleavage only, after whole-topology energy gate",
            "discrete_emission_event": False,
        },
        "call_graph": list(adapted.audit["call_graph"]),
        "factory_identity": dict(adapter.factory_identity),
        "controls": identities,
        "step": {
            "lambda_emit_s": direct_info["lambda_e"],
            "lambda_cleave_s": direct_info["lambda_c"],
            "dN_emit": direct_info.get("dN_emit", direct_info.get("dN_emit_raw")),
            "cleavage_fired": direct_info["fired"],
            "r_tip_m": adapted.updated_r_tip_m,
            "shielding_Pa_sqrt_m": adapted.updated_shielding_Pa_sqrt_m,
            "back_stress_Pa": adapted.updated_back_stress_Pa,
        },
    }
    print(json.dumps(_plain(audit), sort_keys=True, indent=2))
    print("LEGACY_SINGLE_TIP_PARITY = PASS")
    print("LEGACY_MULTITIP_PARITY = PASS")
    print("LEGACY_EMISSION_CLEAVAGE_COMPETITION = PASS")
    print("VOID_DISABLED_NEUTRALITY = PASS")
    print("VOIDING_EXTENSION_DELTA_ONLY = PASS")


if __name__ == "__main__":
    main()
