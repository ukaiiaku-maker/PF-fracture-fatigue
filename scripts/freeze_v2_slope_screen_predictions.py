"""Freeze analytical single-patch predictions at Kmax=15/18/21 MPa*sqrt(m)
(mission Step 3) BEFORE any new trajectory runs.

Reloads the reversible AND persistent finite-cohesion configs verbatim
from the parent branch's committed frozen_configuration.json (no barrier
re-inversion, no rescaling), and re-evaluates A_on(K), A_off(K), p_B*(K),
K_rebond(K) at each Kmax while holding every kinetics/cohesion parameter
fixed. These are prospective diagnostics, not fitted predictions -- frozen
here, before the 8 new trajectories, so they cannot be tuned after seeing
results.

Usage:
    <pinned interpreter> scripts/freeze_v2_slope_screen_predictions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    SCREEN_KMAX_GRID_Pa_sqrt_m,
    analytical_predictions_at_Kmax,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402

PARENT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
OUT_DIR = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"


def _config_from_frozen(frozen: dict, key: str) -> CrackRebondingControls:
    payload = frozen["configs"][key]
    kwargs = dict(payload)
    kwargs["model_level"] = RebondModelLevel(kwargs["model_level"])
    kwargs["contact_model"] = ContactModel(kwargs["contact_model"])
    kwargs["feedback_mode"] = FeedbackMode(kwargs["feedback_mode"])
    kwargs["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(kwargs["initial_precrack_wake_mode"])
    cfg = CrackRebondingControls(**kwargs)
    expected_hash = frozen["config_hashes"][key]
    actual_hash = cfg.config_hash()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"reloaded config {key!r} hash {actual_hash} does not match the parent "
            f"branch's committed frozen_configuration.json hash {expected_hash}"
        )
    return cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((PARENT_ARTIFACTS_DIR / "frozen_configuration.json").read_text())
    reference_contact_radius_m = frozen["material"]["reference_contact_radius_m"]
    Eprime_Pa = frozen["material"]["Eprime_Pa"]

    # Cross-check against a freshly-built bare engine too (belt and braces --
    # confirms the parent's own recorded material constants are still what
    # this worktree's A_NATIVE engine construction produces).
    bare_engine, _ = build_a_native_engine(None)
    fresh_Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    fresh_r_eff = max(bare_engine.r_eff(), 1.0e-9)
    if abs(fresh_Eprime_Pa - Eprime_Pa) / max(abs(Eprime_Pa), 1e-300) > 1e-9:
        raise RuntimeError("fresh A_NATIVE E' does not match the parent's frozen material constant")
    if abs(fresh_r_eff - reference_contact_radius_m) / max(reference_contact_radius_m, 1e-300) > 1e-9:
        raise RuntimeError("fresh A_NATIVE r_eff does not match the parent's frozen material constant")

    rb2_rev_finite = _config_from_frozen(frozen, "RB2_reversible_finite")
    rb2_pers_finite = _config_from_frozen(frozen, "RB2_persistent_finite")

    predictions = {"reversible": [], "persistent": []}
    for label, cfg in (("reversible", rb2_rev_finite), ("persistent", rb2_pers_finite)):
        for Kmax in SCREEN_KMAX_GRID_Pa_sqrt_m:
            pred = analytical_predictions_at_Kmax(
                cfg, Kmax_Pa_sqrt_m=Kmax, reference_contact_radius_m=reference_contact_radius_m,
                Eprime_Pa=Eprime_Pa,
            )
            pred["S_h_pred_note"] = (
                "S_h_pred is not defined from single-patch A_on/A_off alone (S_h is a "
                "population/rate-ratio quantity from the real engine's cleavage hazard, "
                "not the single-patch bonded-fraction model); p_B_star and "
                "K_rebond_predicted are the frozen prospective diagnostics for this model"
            )
            predictions[label].append(pred)

    payload = {
        "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_frozen_predictions_v1",
        "parent_frozen_configuration_sha256": frozen["frozen_configuration_sha256"],
        "kmax_grid_Pa_sqrt_m": list(SCREEN_KMAX_GRID_Pa_sqrt_m),
        "material": {"Eprime_Pa": Eprime_Pa, "reference_contact_radius_m": reference_contact_radius_m},
        "config_hashes_reused_verbatim": {
            "RB2_reversible_finite": frozen["config_hashes"]["RB2_reversible_finite"],
            "RB2_persistent_finite": frozen["config_hashes"]["RB2_persistent_finite"],
        },
        "predictions": predictions,
        "note": (
            "no barrier re-inversion, no activation-volume search, no Pi_K rescaling "
            "at the new loads -- every kinetics/cohesion parameter is held fixed at its "
            "Kmax=18 calibrated value; Pi_K = K_rebond_max/Kmax varies naturally with Kmax"
        ),
    }
    out_path = OUT_DIR / "frozen_predictions.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    for label in ("reversible", "persistent"):
        for pred in predictions[label]:
            print(
                f"  {label} Kmax={pred['Kmax_Pa_sqrt_m']/1e6:.0f}MPa*sqrt(m): "
                f"A_on={pred['A_on']:.4f} A_off={pred['A_off']:.4f} "
                f"p_B*={pred['p_B_star']:.4f} Pi_K={pred['Pi_K_at_this_load']:.4f} "
                f"K_rebond_pred={pred['K_rebond_predicted_Pa_sqrt_m']:.1f} Pa*sqrt(m)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
