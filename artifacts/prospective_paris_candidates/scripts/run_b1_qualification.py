"""B1_RAW_A_NATIVE_QUALIFICATION: run the existing b1_event_conditioned_emission
reduced state-transfer model for A_NATIVE at Kmax=12,15,18,24 (R=0.1, T=300K,
f=1000Hz), with a frozen multi-seed bank and a sample-doubling convergence
check, and compare against the real qualified A_NATIVE physical radius/rate.

B1Controls overrides are set from the qualified A_NATIVE provenance per the
amended protocol:
  n_phase=80, n_bins=80, r0_m=1.0e-6,
  blunting_length_m=8.204647996828617e-7 (physics__blunting_length_m),
  cleavage_hits=3.2732414351776242 (physics__cleavage_hits),
  cleavage_tau_s=6.992153587194454e-7 (physics__cleavage_correlation_time_s).
Every other B1Controls field is a reduced-model design default (explicitly
NOT sourced from a qualified A_NATIVE field) -- recorded in the output as
"unaudited_reduced_model_defaults".

This computes:
  - raw B1 r_eff_m (mean post-event radius after burn-in) and da_dN, with
    Monte Carlo uncertainty from 3 seeds and a sample-doubling check
    (sample_events=80 vs 160) at each K;
  - a terminal-equivalent B1 diagnostic (radius of the LAST sampled event),
    compared separately from the mean, since the physical table only
    reports a single (terminal-window) tip_radius_m per job;
  - the A0_RATE_EQUIVALENT_RADIUS_DIAGNOSTIC_ONLY (radius that would make
    the A0 formula match the real physical rate) -- diagnostic only, never
    used in candidate scoring per the amended protocol.

No physical (production-engine) simulation is run here -- this is the
existing, already-implemented reduced analytical/stochastic model.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from a_native_manifest import load_a_native_manifest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    B1Controls, b1_event_conditioned_emission, RenewalControls, cycle_growth_and_slope,
)

OUT = Path(__file__).resolve().parents[1]

# Physical reference points: A_NATIVE, R=0.1, seed=1720, from the already
# hash-verified 1A_PT03_PT08_R_developed_points.csv (codex/v10.2.30-R-ratio-nominal-deltaK)
PHYSICAL = {
    12.0: dict(tip_radius_m=1.0105318630661543e-06, da_dN=2.448159884833225e-08),
    15.0: dict(tip_radius_m=1.041289045198725e-06, da_dN=3.003481684450034e-07),
    18.0: dict(tip_radius_m=1.1004741824397272e-06, da_dN=4.473410023231299e-07),
    24.0: dict(tip_radius_m=1.343158213754612e-06, da_dN=5.580641987406302e-07),
}
SEED_BANK = [1720, 2024, 3031]
K_GRID = [12.0, 15.0, 18.0, 24.0]
R = 0.1


def run_b1(manifest, extra, K, seed, sample_events):
    controls = B1Controls(
        n_phase=80, n_bins=80, r0_m=1.0e-6,
        blunting_length_m=extra["physics__blunting_length_m"],
        cleavage_hits=extra["physics__cleavage_hits"],
        cleavage_tau_s=extra["physics__cleavage_correlation_time_s"],
        hazard_seed=seed, sample_events=sample_events,
    )
    row = dict(rho_source0_m2=extra["rho_source0_m2"])
    t0 = time.time()
    result = b1_event_conditioned_emission(manifest, row, Kmax_MPa_sqrt_m=K, R=R,
                                            frequency_Hz=1000.0, temperature_K=300.0, controls=controls)
    result["wall_seconds"] = time.time() - t0
    result["seed"] = seed
    result["sample_events_requested"] = sample_events
    result["Kmax_MPa_sqrt_m"] = K
    return result


def a0_rate_equivalent_radius(manifest, extra, K, target_da_dN):
    """Diagnostic only: what fixed radius would make the A0 formula match
    the real physical rate. NEVER used in candidate scoring/selection."""
    controls = RenewalControls(hits=extra["physics__cleavage_hits"],
                                tau_s=extra["physics__cleavage_correlation_time_s"],
                                event_length_m=5.0e-6, frequency_Hz=1000.0,
                                temperature_K=300.0, n_phase=4096)

    def f(r_m):
        c = RenewalControls(**{**controls.__dict__, "radius_m": r_m})
        return cycle_growth_and_slope(manifest.cleavage, K, R, c)["da_dN"] - target_da_dN

    lo, hi = 1.0e-8, 1.0e-4
    try:
        return brentq(f, lo, hi, xtol=1e-14, rtol=1e-10, maxiter=200)
    except ValueError:
        return float("nan")


def main() -> None:
    manifest, extra = load_a_native_manifest()
    records = []
    for K in K_GRID:
        for seed in SEED_BANK:
            for n_sample in (80, 160):
                rec = run_b1(manifest, extra, K, seed, n_sample)
                rec["physical_tip_radius_m"] = PHYSICAL[K]["tip_radius_m"]
                rec["physical_da_dN"] = PHYSICAL[K]["da_dN"]
                rec["a0_rate_equivalent_radius_m_DIAGNOSTIC_ONLY"] = a0_rate_equivalent_radius(
                    manifest, extra, K, PHYSICAL[K]["da_dN"]
                )
                records.append(rec)
                print(json.dumps({k: rec[k] for k in
                                  ("Kmax_MPa_sqrt_m", "seed", "sample_events_requested", "r_eff_m",
                                   "da_dN", "physical_tip_radius_m", "physical_da_dN",
                                   "a0_rate_equivalent_radius_m_DIAGNOSTIC_ONLY", "wall_seconds")}))

    (OUT / "b1_raw_a_native_qualification_records.json").write_text(
        json.dumps(dict(
            schema="v10.2.30_prospective_paris_b1_qualification_v1",
            seed_bank=SEED_BANK, K_grid=K_GRID, R=R,
            controls_overrides_from_qualified_A_NATIVE=dict(
                n_phase=80, n_bins=80, r0_m=1.0e-6,
                blunting_length_m=extra["physics__blunting_length_m"],
                cleavage_hits=extra["physics__cleavage_hits"],
                cleavage_tau_s=extra["physics__cleavage_correlation_time_s"],
            ),
            unaudited_reduced_model_defaults=[
                "maximum_cycles", "burn_events", "sample_events(base=80, doubled to 160 for convergence)",
                "burgers_m", "shear_modulus_Pa", "taylor_stress_fraction", "forest_floor_m2",
                "reference_density_m2", "reference_front_width_m", "reference_source_area_m2",
                "mpz_length_m", "n_systems", "source_zone_length_m", "source_line_per_activation",
                "crystal_theta_deg", "schmid_reference", "stress_cap_Pa", "event_length_m",
                "avalanche_minimum_factor", "avalanche_maximum_factor",
            ],
            records=records,
        ), indent=2, default=str) + "\n"
    )
    print(f"\nWrote {len(records)} B1 qualification records.")


if __name__ == "__main__":
    main()
