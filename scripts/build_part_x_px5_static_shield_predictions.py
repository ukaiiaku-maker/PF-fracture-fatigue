"""PX5 (mission section 8, review-scoped): compute the two prescribed
static-shield control values -- "ceiling" and "analytical periodic-orbit-
matched" -- for D2 and D5 across the five-point seed-1720 Kmax grid.

Ceiling: K_b_static = K_rebond_max itself (the same 900000.0 Pa*sqrt(m)
frozen cohesive-strength constant every PX4 finite-cohesion row already
uses, per-protocol/Kmax-INDEPENDENT by construction -- eta_K*sqrt(Eprime*
G_max) does not depend on the loading schedule). This is the same value
the pre-Part-X static_shield_attribution study used (K_B_STATIC_Pa_sqrt_m
= 900000.0), reused verbatim here rather than recomputed, since it is by
definition schedule-independent.

Analytical periodic-orbit-matched: K_b_static = K_rebond_max * mean_p_B,
where mean_p_B is the TRUE periodic-orbit cycle-mean bonded occupancy for
D2's/D5's own resolved rebonding config (row config + the same
chemistry_factor/K_target overrides every PX4 finite-cohesion job
applies) at that Kmax -- computed analytically via
analytical_periodic_orbit + predicted_static_shield_equivalent_K_b
(reused verbatim, zero fitting to any physical trajectory, per mission
section 6's own no-fitting requirement). Because mean_p_B < 1 whenever
the crack does not stay bonded for the entire cycle, this control is
<= the ceiling by construction.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine  # noqa: E402
from arrhenius_fracture.crack_rebonding_part_x_kinetic_regime_v10230 import (  # noqa: E402
    analytical_periodic_orbit, predicted_static_shield_equivalent_K_b,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402
from part_x_run_one_job import _base_rebonding_cfg, _resolve_screen_rebonding_cfg  # noqa: E402

KMAX_GRID_Pa_sqrt_m = [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]
R_REF = -0.5
FREQUENCY_HZ = 1000.0
HOLD_S = 0.0
CHEMISTRY_FACTOR = 1.0
K_TARGET_CEILING_Pa_sqrt_m = 900000.0
T_K = 300.0
N_PHASE_ANALYTICAL = 64

PROTOCOL_ROW = {"D2": "COMPETING_REVERSIBLE", "D5": "PASSIVATION_LIMITED"}


def main() -> None:
    engine, _ = build_a_native_engine()
    Eprime_Pa = reduced_modulus_Pa(engine.G, engine.nu)
    r_eff_m = float(engine.r_eff())

    registry = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"]

    rows_out = []
    for protocol, row_name in PROTOCOL_ROW.items():
        row_payload = registry[row_name]
        finite_job = {
            "cohesion": "finite", "chemistry_factor": CHEMISTRY_FACTOR,
            "K_rebond_max_target_Pa_sqrt_m": K_TARGET_CEILING_Pa_sqrt_m,
        }
        cfg = _resolve_screen_rebonding_cfg(finite_job, row_payload, Eprime_Pa)
        eta_K = cfg.rebond_K_geometry_factor
        G_max = cfg.restored_work_of_separation_J_m2
        K_rebond_max_recomputed = eta_K * (Eprime_Pa * G_max) ** 0.5
        if abs(K_rebond_max_recomputed - K_TARGET_CEILING_Pa_sqrt_m) > 1.0e-3:
            raise RuntimeError(
                f"{protocol}: recomputed K_rebond_max {K_rebond_max_recomputed} != ceiling target "
                f"{K_TARGET_CEILING_Pa_sqrt_m} -- cfg resolution is inconsistent with PX4's own finite-cohesion jobs"
            )

        for Kmax in KMAX_GRID_Pa_sqrt_m:
            orbit = analytical_periodic_orbit(
                cfg=cfg, T_K=T_K, Kmax_Pa_sqrt_m=Kmax, R=R_REF, frequency_Hz=FREQUENCY_HZ,
                minimum_load_hold_s=HOLD_S, r_contact_m=r_eff_m, n_phase=N_PHASE_ANALYTICAL,
            )
            K_b_orbit = predicted_static_shield_equivalent_K_b(orbit, cfg, Eprime_Pa)
            rows_out.append({
                "protocol": protocol, "row_name": row_name, "Kmax_Pa_sqrt_m": Kmax, "R": R_REF,
                "frequency_Hz": FREQUENCY_HZ, "minimum_load_hold_s": HOLD_S,
                "mean_p_B_periodic_orbit": orbit["mean_p_B"],
                "ceiling_K_b_static_Pa_sqrt_m": K_TARGET_CEILING_Pa_sqrt_m,
                "periodic_orbit_matched_K_b_static_Pa_sqrt_m": K_b_orbit,
                "convergence_residual": orbit["convergence_residual"],
            })

    with (ARTIFACTS_DIR / "px5_static_shield_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    (ARTIFACTS_DIR / "px5_static_shield_predictions.json").write_text(json.dumps({
        "schema": "v10230_part_x_px5_static_shield_predictions_v1",
        "Eprime_Pa": Eprime_Pa, "r_eff_m": r_eff_m, "rows": rows_out,
    }, indent=2, default=str))

    print(f"Wrote px5_static_shield_predictions.{{csv,json}}: {len(rows_out)} rows")
    for r in rows_out:
        print(f"  {r['protocol']} Kmax={r['Kmax_Pa_sqrt_m']/1e6:5.1f}MPa  mean_p_B={r['mean_p_B_periodic_orbit']:.6f}  "
              f"ceiling={r['ceiling_K_b_static_Pa_sqrt_m']:.1f}  orbit_matched={r['periodic_orbit_matched_K_b_static_Pa_sqrt_m']:.4f}")


if __name__ == "__main__":
    main()
