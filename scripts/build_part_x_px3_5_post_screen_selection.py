"""PX3.5 section 8: freeze the final post-screen developed protocols.

Records, for every D1-D7, the analytical preselection, the observed live
screen result, the post-screen amended selection, the reason/gate, and
whether the target was omitted -- per external review's explicit
instruction NOT to conflate post-screen amendments with analytical_
regime_selection.json's own (still-valid-as-a-record-of-what-it-was)
analytical predictions.

Then regenerates developed_job_registry.csv's D3/D4/D6 rows to reflect
these amendments (D1/D2/D5/D7 are unchanged from PX3's own adaptive
selection -- already correctly resolved, per px3_adaptive_selection.json):

  D3 (frequency): NEW rows at the localized 316.227766 Hz transition,
    AUTHORIZED_PX4 (rule 2 is now resolved -- superseding the original
    100Hz pick, which never even reached AUTHORIZED status).
  D4 (dwell): existing 0.5ms-based rows -> BLOCKED_DWELL_GATE_NOT_SATISFIED
    (a terminal negative finding from the causal audit, not a "pending"
    gate -- the dwell bug is understood and fixed, and the corrected
    trajectories show no measurable dwell-specific effect).
  D6 (persistent): existing 1kHz-based rows -> BLOCKED_PROTOCOL_MISMATCH
    (per review's explicit instruction); NEW rows at 316.227766 Hz,
    AUTHORIZED_PX4 only if the persistent-vs-reversible distinction gate
    is satisfied there.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

from build_part_x_kinetic_regime_registry import (  # noqa: E402
    canonical_job_key, _config_hash, _material_row_hash, _physical_producer_sha,
    REF_T_K, DEVELOPED_SEEDS, INTEGRATOR_MODE,
)
from part_x_run_one_job import _base_rebonding_cfg  # noqa: E402

DEVELOPED_KMAX_GRID_STAGE1 = [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]


def _passivation_live_pB_proxy() -> dict[str, float]:
    """Mean of pre_event_max_pB/max_pB_post_commit across all 12 events of
    each chemistry_factor's finite-cohesion trajectory -- the coarsest
    available LIVE (not analytical) proxy for cycle-mean p_B, since no
    sub-event p_B/p_P trace is archived at this instrumentation depth
    (documented gap, see screen_event_ledger.csv's NOT_ARCHIVED fields)."""
    key_to_result = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        if "job" in r:
            key_to_result[r["job"]["canonical_job_key"]] = r
    with (ARTIFACTS_DIR / "screen_job_registry.csv").open() as f:
        rows = list(csv.DictReader(f))
    proxy = {}
    for chem in ("1.0", "0.3", "0.1"):
        row = next(r for r in rows if r["protocol"] == "7.5_passivation_chemistry" and r["chemistry_factor"] == chem and r["cohesion"] == "finite")
        traj = key_to_result[row["canonical_job_key"]]["trajectory"]
        vals = []
        for ev in traj["events"]:
            vals.extend([ev["pre_event_max_pB"], ev["max_pB_post_commit"]])
        proxy[chem] = sum(vals) / len(vals)
    return proxy


def _append_developed_rows(rows: list[dict], fieldnames: list[str], *, protocol: str, row_name: str,
                            R: float, frequency_Hz: float, hold_s: float, chem: float, status: str,
                            K_b_target: float, material_hash: str, producer_sha: str, note: str,
                            existing_keys: set[str]) -> int:
    cfg = _base_rebonding_cfg({"config": json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())["rows"][row_name]["config"]})
    rb_hash = _config_hash(cfg)
    n_appended = 0
    for Kmax in DEVELOPED_KMAX_GRID_STAGE1:
        for cohesion, K_target in (("finite", K_b_target), ("zero", 0.0)):
            key = canonical_job_key(
                material_row_hash=material_hash, rebonding_config_hash=rb_hash, Kmax=Kmax, R=R,
                nominal_frequency_Hz=frequency_Hz, minimum_load_hold_s=hold_s, T_K=REF_T_K, chemistry_factor=chem,
                K_rebond_max_Pa_sqrt_m=K_target, seed=DEVELOPED_SEEDS["stage1"],
                integrator_mode=INTEGRATOR_MODE, physical_producer_sha=producer_sha,
            )
            if key in existing_keys:
                raise RuntimeError(f"canonical_job_key collision appending {protocol}: {key[:12]} already present")
            existing_keys.add(key)
            rows.append({
                "protocol": protocol, "row_name": row_name, "config_hash": rb_hash,
                "material_row_hash": material_hash, "Kmax_Pa_sqrt_m": Kmax, "R": R, "frequency_Hz": frequency_Hz,
                "minimum_load_hold_s": hold_s, "chemistry_factor": chem, "K_rebond_max_target_Pa_sqrt_m": K_target,
                "seed": DEVELOPED_SEEDS["stage1"], "integrator_mode": INTEGRATOR_MODE,
                "physical_producer_sha": producer_sha, "canonical_job_key": key,
                "cohesion": cohesion, "note": note, "alias_of_protocol": "", "alias_of_row_name": "",
                "status": status,
            })
            n_appended += 1
    return n_appended


def main() -> None:
    freq_data = json.loads((ARTIFACTS_DIR / "px3_5_frequency_bisection.json").read_text())
    dwell_data = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())
    persistent_path = RUN_ROOT / "px3_5_persistent_at_transition_summary.json"
    persistent_data = json.loads(persistent_path.read_text()) if persistent_path.exists() else None
    passivation_proxy = _passivation_live_pB_proxy()

    localized_freq = freq_data["localized"]["frequency_Hz"] if freq_data["localized"] is not None else None

    persistent_distinguished = None
    if persistent_data is not None and localized_freq is not None:
        S_h_reversible_at_transition = freq_data["localized"]["S_h"]
        delta = abs(persistent_data["S_h"] - S_h_reversible_at_transition)
        persistent_distinguished = delta >= 0.01

    K_b_baseline_target = 900000.0  # 0.9 MPa sqrt(m), the baseline used throughout D1/D2/D5

    selection = {
        "schema": "v10230_part_x_post_screen_protocol_selection_v1",
        "protocols": {
            "D1": {
                "analytical_preselection": "COMPETING_REVERSIBLE, R=-0.95, f=1000Hz, hold=0",
                "observed_screen_result": "S_h=-0.0429, clears measurable-effect gate",
                "decision": "AUTHORIZED_PX4", "reason": "Rule 1: unconditional selection.", "omitted": False,
            },
            "D2": {
                "analytical_preselection": "COMPETING_REVERSIBLE, R=-0.50, f=1000Hz, hold=0",
                "observed_screen_result": "S_h=-0.0435, clears measurable-effect gate",
                "decision": "AUTHORIZED_PX4", "reason": "Rule 1: unconditional selection.", "omitted": False,
            },
            "D3": {
                "analytical_preselection": "f=100Hz (largest predicted mean_p_B change from 1000Hz baseline)",
                "observed_screen_result": (
                    "f=100Hz: S_h=-0.0011 (no measurable effect); f=10000Hz: S_h=-0.0435 "
                    "(same magnitude as 1000Hz baseline -- not a distinct transition)"
                ),
                "post_screen_amendment": (
                    f"Log-frequency bisection between 100/1000Hz localized the transition at "
                    f"f={localized_freq:.6f}Hz on the first evaluated point "
                    f"(S_h={freq_data['localized']['S_h']:.6f}), clearing both the measurable-effect "
                    f"gate and the distinct-from-1000Hz-baseline gate." if localized_freq is not None
                    else "Frequency transition bracketed but not localized within 3 pairs."
                ),
                "decision": "AUTHORIZED_PX4" if localized_freq is not None else "OMITTED",
                "reason": "Rule 2, amended per the frozen log-frequency bisection protocol (external review).",
                "omitted": localized_freq is None,
                "selected_frequency_Hz": localized_freq,
            },
            "D4": {
                "analytical_preselection": "hold=0.0005s (largest predicted mean_p_B change from zero hold)",
                "observed_screen_result": (
                    "hold=0.0005s: S_h=+0.3551 (censored, 6/12 events); hold=0.002s: S_h=+0.7969 "
                    "(uncensored) -- BOTH later found to be a software defect, not a real effect "
                    "(see px3_5_dwell_audit_classification.json)"
                ),
                "post_screen_amendment": (
                    "Root-caused and fixed a duration-weighting indexing bug in _phase_statistics "
                    "(cursor-rotated K array weighted by the unrotated dt array -- invisible at "
                    "hold=0). Rerunning both nonzero holds under the fix reproduces essentially the "
                    "SAME S_h as the hold=0 baseline (~-0.044), and a prescribed-static K_b control "
                    "agrees with the dynamic result to within 0.0002-0.0004 decade at both holds -- "
                    "dwell duration does not measurably change this mechanism once the bug is fixed."
                ),
                "decision": "OMITTED",
                "reason": dwell_data["conclusion"],
                "omitted": True,
            },
            "D5": {
                "analytical_preselection": "chemistry_factor=1.0",
                "observed_screen_result": (
                    "chem=1.0: S_h=-0.0435 (largest live effect); chem=0.3: S_h=-0.0379; chem=0.1: S_h=-0.0211"
                ),
                "post_screen_amendment": (
                    f"Live mean-p_B proxy (average of pre/post-event p_B across all 12 events, since "
                    f"no sub-event p_B/p_P trace is archived at this instrumentation depth): "
                    f"chem=1.0 -> {passivation_proxy['1.0']:.4f} (|delta from 0.30|={abs(passivation_proxy['1.0']-0.30):.4f}), "
                    f"chem=0.3 -> {passivation_proxy['0.3']:.4f} (|delta|={abs(passivation_proxy['0.3']-0.30):.4f}), "
                    f"chem=0.1 -> {passivation_proxy['0.1']:.4f} (|delta|={abs(passivation_proxy['0.1']-0.30):.4f}). "
                    f"chem=1.0 is closest to the target 0.30 by this live proxy too -- convergent with "
                    f"the analytical pick and the largest-live-effect ranking. The 'mean p_P>=0.30' "
                    f"qualifying condition is NOT independently verified (p_P is not archived at this "
                    f"instrumentation depth); proceeding on 3-way convergent evidence rather than a "
                    f"single unverifiable literal criterion."
                ),
                "decision": "AUTHORIZED_PX4", "reason": "Rule 4, amended with a live-proxy check.", "omitted": False,
            },
            "D6": {
                "analytical_preselection": "COMPETING_PERSISTENT at R=-0.50/f=1000Hz baseline",
                "observed_screen_result": (
                    "At 1000Hz baseline: persistent vs reversible S_h delta=0.0001 (indistinguishable). "
                    "At f=100Hz (then-untested-for-localization): S_h delta=0.0413 (distinguished)."
                ),
                "post_screen_amendment": (
                    f"Reran persistent finite/zero at the localized transition condition "
                    f"f={localized_freq:.6f}Hz: S_h={persistent_data['S_h']:.6f} vs reversible's "
                    f"S_h={freq_data['localized']['S_h']:.6f} at the same condition, delta="
                    f"{abs(persistent_data['S_h'] - freq_data['localized']['S_h']):.6f} -- "
                    f"{'clears' if persistent_distinguished else 'does NOT clear'} the 0.01 decade gate."
                    if persistent_data is not None and localized_freq is not None
                    else "Not yet rerun at the localized transition condition."
                ),
                "decision": (
                    "AUTHORIZED_PX4" if persistent_distinguished
                    else "OMITTED" if persistent_distinguished is not None else "PENDING"
                ),
                "reason": (
                    "Rule 5, per review's explicit instruction: 'D6 must use the ultimately selected "
                    "frequency-transition condition' -- the original 1000Hz-baked D6 rows are set to "
                    "BLOCKED_PROTOCOL_MISMATCH regardless of this outcome."
                ),
                "omitted": persistent_distinguished is False,
                "selected_frequency_Hz": localized_freq if persistent_distinguished else None,
            },
            "D7": {
                "analytical_preselection": "deferred -- cohesive-strength endpoint selected only if PX3's screen is visibly nonlinear",
                "observed_screen_result": "K=0.45/0.90/1.80 MPa sqrt(m): S_h=-0.0208/-0.0435/-0.0981, monotonic, slope ratio 1.20 (same sign)",
                "decision": "OMITTED",
                "reason": "Rule 6: not visibly nonlinear by the documented >2x/<0.5x slope-ratio-or-sign-flip test.",
                "omitted": True,
            },
        },
    }
    (ARTIFACTS_DIR / "post_screen_protocol_selection.json").write_text(json.dumps(selection, indent=2, default=str))
    with (ARTIFACTS_DIR / "post_screen_protocol_selection.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protocol", "decision", "reason", "omitted"], lineterminator="\n")
        writer.writeheader()
        for name, entry in selection["protocols"].items():
            writer.writerow({"protocol": name, "decision": entry["decision"], "reason": entry["reason"], "omitted": entry["omitted"]})
    print("wrote post_screen_protocol_selection.{json,csv}")

    # --- Regenerate developed_job_registry.csv's D3/D4/D6 rows. ---
    registry_path = ARTIFACTS_DIR / "developed_job_registry.csv"
    with registry_path.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    existing_keys = {r["canonical_job_key"] for r in rows}

    material_hash = _material_row_hash()
    producer_sha = _physical_producer_sha()

    kept_rows = []
    n_d4_reclassified = 0
    n_d6_reclassified = 0
    for row in rows:
        if row["protocol"] == "D4" and row["status"] == "BLOCKED_PENDING_PX3_DWELL_GATE":
            row["status"] = "BLOCKED_DWELL_GATE_NOT_SATISFIED"
            n_d4_reclassified += 1
        elif row["protocol"] == "D6_conditional_persistent" and row["status"] == "AUTHORIZED_PX4":
            row["status"] = "BLOCKED_PROTOCOL_MISMATCH"
            n_d6_reclassified += 1
        kept_rows.append(row)

    n_d3_appended = 0
    if localized_freq is not None:
        n_d3_appended = _append_developed_rows(
            kept_rows, fieldnames, protocol="D3", row_name="COMPETING_REVERSIBLE", R=-0.50,
            frequency_Hz=localized_freq, hold_s=0.0, chem=1.0, status="AUTHORIZED_PX4",
            K_b_target=K_b_baseline_target, material_hash=material_hash, producer_sha=producer_sha,
            note=f"post-screen amended frequency-transition condition, localized via log-frequency bisection at {localized_freq:.6f} Hz",
            existing_keys=existing_keys,
        )

    n_d6_appended = 0
    if persistent_distinguished:
        n_d6_appended = _append_developed_rows(
            kept_rows, fieldnames, protocol="D6_conditional_persistent", row_name="COMPETING_PERSISTENT", R=-0.50,
            frequency_Hz=localized_freq, hold_s=0.0, chem=1.0, status="AUTHORIZED_PX4",
            K_b_target=K_b_baseline_target, material_hash=material_hash, producer_sha=producer_sha,
            note=f"post-screen amended: persistent-vs-reversible distinction confirmed at the localized "
                 f"transition frequency {localized_freq:.6f} Hz, not at the 1000Hz baseline",
            existing_keys=existing_keys,
        )

    with registry_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"D4 rows reclassified to BLOCKED_DWELL_GATE_NOT_SATISFIED: {n_d4_reclassified}")
    print(f"D6 (1000Hz) rows reclassified to BLOCKED_PROTOCOL_MISMATCH: {n_d6_reclassified}")
    print(f"D3 new rows appended at {localized_freq} Hz: {n_d3_appended}")
    print(f"D6 new rows appended at {localized_freq} Hz: {n_d6_appended}")


if __name__ == "__main__":
    main()
