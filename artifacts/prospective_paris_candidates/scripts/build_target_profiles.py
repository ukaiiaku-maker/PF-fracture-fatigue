"""Section 4: define and freeze the P25/P40/P55 target local-slope and
target rate profiles, for both the STANDARD_WINDOW and EXTENDED_FLAT_WINDOW
slope-window variants, using the freshly source-qualified g_star anchor
(NOT the unrecoverable historical INV_OPENING_M4_V1 value).

m_target(K) = m_entry + (m_mid - m_entry) * L_on(K) - (m_mid - m_exit) * L_off(K)
  m_entry = 0.80 * m_mid ; m_exit = 0.70 * m_mid
  L_on(K)  = 1 / (1 + exp(-(ln K - ln K_on)  / w_on))
  L_off(K) = 1 / (1 + exp(-(ln K - ln K_off) / w_off))

ln g_target(K) = ln g_star + integral_{ln Kstar}^{ln K} m_target(K') d ln K'
(exact log-space integration via cumulative trapezoid on a dense grid, then
interpolated onto every requested output grid -- this keeps the integral
truly continuous rather than re-truncated per grid).

No physical simulation is run here. Pure, deterministic, hashed math.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

OUT = Path(__file__).resolve().parents[1]
GSTAR_FILE = OUT / "rate_anchor_gstar.json"

K_STAR = 18.0
TARGETS = {"P25": 2.5, "P40": 4.0, "P55": 5.5}
WINDOWS = {
    "STANDARD_WINDOW": dict(K_on=12.75, K_off=22.0, w_on=0.040, w_off=0.060),
    "EXTENDED_FLAT_WINDOW": dict(K_on=12.30, K_off=23.00, w_on=0.030, w_off=0.045),
}
PHYSICAL_GRID = [12.0, 12.75, 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 24.3]
DENSE_N = 4001  # >> the mandated minimum of 257; keeps the frozen integral essentially exact
K_LO, K_HI = 12.0, 24.3


def m_target(K: np.ndarray, m_mid: float, window: dict) -> np.ndarray:
    m_entry = 0.80 * m_mid
    m_exit = 0.70 * m_mid
    lnK = np.log(K)
    L_on = 1.0 / (1.0 + np.exp(-(lnK - np.log(window["K_on"])) / window["w_on"]))
    L_off = 1.0 / (1.0 + np.exp(-(lnK - np.log(window["K_off"])) / window["w_off"]))
    return m_entry + (m_mid - m_entry) * L_on - (m_mid - m_exit) * L_off


def dense_log_grid() -> np.ndarray:
    return np.exp(np.linspace(np.log(K_LO), np.log(K_HI), DENSE_N))


def ln_g_target_dense(K_dense: np.ndarray, m_mid: float, window: dict, ln_gstar: float) -> np.ndarray:
    lnK = np.log(K_dense)
    slopes = m_target(K_dense, m_mid, window)
    integral = cumulative_trapezoid(slopes, lnK, initial=0.0)
    i_star = int(np.argmin(np.abs(K_dense - K_STAR)))
    # exact log-space integral referenced at ln(K_star); interpolate the tiny
    # residual to K_star exactly rather than snapping to the nearest grid node
    integral_at_star = np.interp(np.log(K_STAR), lnK, integral)
    return ln_gstar + (integral - integral_at_star)


def row_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> None:
    gstar_record = json.loads(GSTAR_FILE.read_text())
    g_star = float(gstar_record["gstar_m_per_cycle"])
    ln_gstar = float(np.log(g_star))
    assert abs(gstar_record["Kstar_MPa_sqrt_m"] - K_STAR) < 1e-9

    K_dense = dense_log_grid()
    dense_records = []
    slope_rows = []
    rate_rows = []

    for name, m_mid in TARGETS.items():
        for wname, window in WINDOWS.items():
            profile_id = f"{name}_{wname}"
            ln_g_dense = ln_g_target_dense(K_dense, m_mid, window, ln_gstar)
            m_dense = m_target(K_dense, m_mid, window)
            dense_records.append(dict(profile_id=profile_id, target_name=name, window=wname,
                                       m_mid=m_mid, K=K_dense, m=m_dense, ln_g=ln_g_dense))

            for K in PHYSICAL_GRID:
                m_val = float(np.interp(np.log(K), np.log(K_dense), m_dense))
                ln_g_val = float(np.interp(np.log(K), np.log(K_dense), ln_g_dense))
                slope_rows.append(dict(profile_id=profile_id, target_name=name, window=wname,
                                        m_mid=m_mid, Kmax_MPa_sqrt_m=K, m_target=round(m_val, 6)))
                rate_rows.append(dict(profile_id=profile_id, target_name=name, window=wname,
                                       m_mid=m_mid, Kmax_MPa_sqrt_m=K,
                                       g_target_m_per_cycle=float(np.exp(ln_g_val))))

    # dense analytical-grid CSVs (one row per profile per dense K node)
    import csv
    with (OUT / "target_slope_profiles.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["profile_id", "target_name", "window", "m_mid",
                                            "Kmax_MPa_sqrt_m", "m_target"], lineterminator="\n")
        w.writeheader()
        w.writerows(slope_rows)
        for rec in dense_records:
            for K, m in zip(rec["K"], rec["m"]):
                w.writerow(dict(profile_id=f"{rec['profile_id']}__DENSE", target_name=rec["target_name"],
                                 window=rec["window"], m_mid=rec["m_mid"],
                                 Kmax_MPa_sqrt_m=round(float(K), 6), m_target=round(float(m), 6)))

    with (OUT / "target_rate_profiles.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["profile_id", "target_name", "window", "m_mid",
                                            "Kmax_MPa_sqrt_m", "g_target_m_per_cycle"], lineterminator="\n")
        w.writeheader()
        w.writerows(rate_rows)
        for rec in dense_records:
            for K, ln_g in zip(rec["K"], rec["ln_g"]):
                w.writerow(dict(profile_id=f"{rec['profile_id']}__DENSE", target_name=rec["target_name"],
                                 window=rec["window"], m_mid=rec["m_mid"],
                                 Kmax_MPa_sqrt_m=round(float(K), 6),
                                 g_target_m_per_cycle=float(np.exp(ln_g))))

    definition = dict(
        schema="v10.2.30_prospective_paris_target_profile_definition_v1",
        Kstar_MPa_sqrt_m=K_STAR,
        gstar_m_per_cycle=g_star,
        gstar_source_row_sha256=gstar_record["source_row_sha256"],
        gstar_provenance_file="rate_anchor_gstar.json",
        m_entry_fraction=0.80, m_exit_fraction=0.70,
        targets={k: v for k, v in TARGETS.items()},
        windows=WINDOWS,
        physical_grid_MPa_sqrt_m=PHYSICAL_GRID,
        dense_grid_n=DENSE_N, dense_grid_K_lo=K_LO, dense_grid_K_hi=K_HI,
        formula_m_target="m_entry + (m_mid-m_entry)*L_on(K) - (m_mid-m_exit)*L_off(K)",
        formula_L="1/(1+exp(-(lnK-lnK_edge)/w_edge))",
        formula_rate="ln g_target(K) = ln g_star + integral_{lnKstar}^{lnK} m_target(K') dlnK' (exact log-space cumulative trapezoid on a dense >=257-point grid)",
    )
    definition["definition_sha256"] = row_hash({k: v for k, v in definition.items() if k != "definition_sha256"})
    (OUT / "target_profile_definition.json").write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")

    slope_sha = hashlib.sha256((OUT / "target_slope_profiles.csv").read_bytes()).hexdigest()
    rate_sha = hashlib.sha256((OUT / "target_rate_profiles.csv").read_bytes()).hexdigest()
    print(json.dumps(dict(profiles=len(dense_records), slope_rows=len(slope_rows) , rate_rows=len(rate_rows),
                          target_slope_profiles_sha256=slope_sha, target_rate_profiles_sha256=rate_sha,
                          definition_sha256=definition["definition_sha256"]), indent=2))


if __name__ == "__main__":
    main()
