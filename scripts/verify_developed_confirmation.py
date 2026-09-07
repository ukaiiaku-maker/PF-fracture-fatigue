"""Strict, independently re-deriving verifier for the developed-response
confirmation campaign. Depends ONLY on files tracked in this branch
(artifacts/crack_rebonding_developed_confirmation/*, artifacts/
crack_rebonding_causal_pilot_v2/frozen_configuration.json and
A_native_provenance.json, and this branch's own physical-producer
provenance record) -- never on any gitignored runs/ directory, so it must
still pass with every runs/ output deleted.

v2 (evidence-hardening pass, post-review): the review noted that v1's
classification re-derivation imported analyze_developed_confirmation and
called its own analyze_seed(), so it would silently reproduce a shared
analysis bug (such as v1's mislabeled tail windows) rather than catch it.
This version adds a SEPARATE recomputation path (independent_recompute
below) that reimplements the window selection, da/dN, S_h, delta_m, and
classification-input assembly directly against the shared, reused
primitives (stable_growth_gate, event_index_window_rate,
true_final_half_indices, three_point_slope_fit,
classify_developed_confirmation, apply_tail_sensitivity_gate) WITHOUT
calling analyze_developed_confirmation.analyze_seed() -- so a defect
specific to that module's glue code shows up as a verifier disagreement
rather than being silently reproduced.

Also added per review: exact event-ordinal-sequence and accepted-length-
sequence identity checks (not just aggregate assertions), exact event-
count identity, per-trajectory R/seed/rebonding-config-hash/material-
manifest-hash checks, a hard requirement that the accepted-length
identity check pass before the waiting-time simplification is used, and a
check against the physical-producer provenance record (Section 4) rather
than assuming HEAD is the producer commit.

v3 (decomposition-repair pass, post-second-review): this level of
independence is accurately described as INDEPENDENT_LEDGER_REDUCTION_
WITH_SHARED_QUALIFIED_PRIMITIVES, not "fully independent" -- it still
calls stable_growth_gate, event_index_window_rate,
true_final_half_indices/true_final_six_indices, three_point_slope_fit,
classify_developed_confirmation, and apply_tail_sensitivity_gate, which
are the campaign's frozen/reused formulas themselves (reimplementing
those would defeat the point of reusing them verbatim), not analysis
glue. Extended per this review round: true_final_six and the
last_50_um/last_25_um extension windows are now independently recomputed
and cross-checked against the saved decision (not just true_final_half);
hard checks confirm the saved decision's developed-window diagnostics are
NOT aliased to its all-event diagnostics (the second substantive defect
this round found) and that the retracted inter-event action-weighted
K_rebond mean is recorded as NOT_ARCHIVED, not silently reintroduced.
Fields the tracked ledger genuinely does not archive per trajectory
(Kmax_Pa_sqrt_m, T_K, frequency_Hz, a nested per-event RNG seed/state
identifier distinct from hazard_event_index, and an explicit no-resume
marker) are reported as NOT_ARCHIVED rather than inferred from the
trajectory's own name or a hard-coded study constant; T_K/frequency_Hz
are instead validated through the single shared frozen_configuration_
sha256 reproducibility check, which covers them structurally since every
trajectory reloads that same file.

Usage:
    <pinned interpreter> scripts/verify_developed_confirmation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture.a_native_engine_v10230 import (  # noqa: E402
    build_a_native_engine,
    load_a_native_provenance,
)
from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (  # noqa: E402
    STABILITY_DEFINITION_STRING,
    apply_tail_sensitivity_gate,
    classify_developed_confirmation,
    developed_interval_event_indices,
    effective_horizon_censored,
    event_index_window_rate,
    extension_window_rate,
    stable_growth_gate,
    true_final_half_indices,
    true_final_six_indices,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (  # noqa: E402
    ContactModel, CrackRebondingControls, FeedbackMode, InitialPrecrackWakeMode, RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    three_point_slope_fit,
)
from arrhenius_fracture.crack_rebonding_v10230 import reduced_modulus_Pa  # noqa: E402

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
DEV_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
FREQUENCY_HZ = 1000.0
FROZEN_R = -0.95
MAX_ACCEPTED_EVENTS = 30
MAX_PROJECTED_EXTENSION_m = 1.5e-4
SLOPE_GATE = 0.25

TRACKED_ARTIFACT_NAMES = [
    "developed_confirmation_frozen_protocol.json",
    "developed_confirmation_job_registry.csv",
    "developed_confirmation_predictions.json",
    "developed_confirmation_verification_contract.json",
    "developed_confirmation_physical_producer_provenance.json",
    "event_ledger.json",
    "event_ledger.csv",
    "developed_confirmation_decision.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_git_blob(commit: str, rel_path: str) -> str:
    """Hash of the file's content AT a historical commit -- used to check
    physical-producer-code identity against the commit that actually ran
    the trajectories, independent of subsequent (legitimate) edits to the
    same file later in this evidence-hardening branch."""
    content = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"], cwd=REPO_ROOT,
        check=True, capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _config_from_frozen(frozen: dict, key: str) -> CrackRebondingControls:
    payload = dict(frozen["configs"][key])
    payload["model_level"] = RebondModelLevel(payload["model_level"])
    payload["contact_model"] = ContactModel(payload["contact_model"])
    payload["feedback_mode"] = FeedbackMode(payload["feedback_mode"])
    payload["initial_precrack_wake_mode"] = InitialPrecrackWakeMode(payload["initial_precrack_wake_mode"])
    return CrackRebondingControls(**payload)


def _traj_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def _independent_S_h(zero_win: dict, finite_win: dict) -> tuple[float | None, bool]:
    """Standalone re-implementation (not shared with analyze_developed_
    confirmation._S_h_from_windows) of the S_h formula and the accepted-
    length-identity precondition for the waiting-time simplification."""
    len_ident = (
        zero_win["da_m"] > 0.0 and finite_win["da_m"] > 0.0
        and abs(zero_win["da_m"] - finite_win["da_m"]) < 1.0e-12 * max(zero_win["da_m"], finite_win["da_m"])
    )
    zr = zero_win["da_m"] / zero_win["dN"] if zero_win["dN"] > 0.0 else None
    fr = finite_win["da_m"] / finite_win["dN"] if finite_win["dN"] > 0.0 else None
    if not zr or not fr or zr <= 0.0 or fr <= 0.0:
        return None, len_ident
    return math.log10(fr / zr), len_ident


NOT_ARCHIVED_PER_TRAJECTORY_FIELDS = [
    "Kmax_Pa_sqrt_m", "T_K", "frequency_Hz",
    "nested_per_event_rng_seed_or_state_identifier", "explicit_no_resume_marker",
]


def independent_recompute(ledger: dict, protocol: dict, decision: dict) -> tuple[dict, dict, dict]:
    """INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES:
    reimplements per-pair hard checks, window selection, S_h, and delta_m
    fitting directly against the shared, reused primitives -- never calls
    analyze_developed_confirmation.analyze_seed(). Not a from-scratch
    reimplementation of the primitives themselves (stable_growth_gate,
    the window-index functions, three_point_slope_fit, and the
    classifiers ARE the campaign's frozen/reused formulas, intentionally
    reused rather than reimplemented)."""
    checks: dict[str, bool] = {}
    details: dict[str, object] = {
        "not_archived_per_trajectory_fields": NOT_ARCHIVED_PER_TRAJECTORY_FIELDS,
        "not_archived_note": (
            "the tracked ledger does not record Kmax/T_K/frequency_Hz per "
            "trajectory (T_K/frequency_Hz are instead validated structurally "
            "via the single shared frozen_configuration_sha256 check, since "
            "every trajectory reloads that same file), nor a nested per-event "
            "RNG seed/state snapshot distinct from hazard_event_index, nor an "
            "explicit per-trajectory no-resume marker (no-resume is a "
            "structural guarantee of run_developed_confirmation_stage.py's "
            "refusal to overwrite an existing result path, not a stored "
            "field) -- these are reported here as not archived rather than "
            "inferred from the trajectory name or a hard-coded constant"
        ),
    }

    seeds_present = set()
    for seed in (1720, 1001723):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            Kmax_MPa = int(round(Kmax / 1.0e6))
            for cohesion in ("zero", "finite"):
                if _traj_name(seed, Kmax_MPa, cohesion) in ledger["trajectories"]:
                    seeds_present.add(seed)

    expected_hash_zero = protocol["material_and_provenance"]["config_hashes_reused_verbatim"]["RB2_reversible_zero"]
    expected_hash_finite = protocol["material_and_provenance"]["config_hashes_reused_verbatim"]["RB2_reversible_finite"]
    expected_material_row_hash = protocol["material_and_provenance"]["complete_active_material_row_sha256"]

    manifest_hashes_seen = set()
    per_seed: dict[int, dict] = {}
    for seed in sorted(seeds_present):
        S_h_developed_by_K: dict[float, float | None] = {}
        S_h_true_final_half_by_K: dict[float, float | None] = {}
        S_h_true_final_six_by_K: dict[float, float | None] = {}
        S_h_last_50_um_by_K: dict[float, float | None] = {}
        S_h_last_25_um_by_K: dict[float, float | None] = {}
        gate_pass_all = True
        uncensored_all = True

        for Kmax in KMAX_GRID_Pa_sqrt_m:
            Kmax_MPa = int(round(Kmax / 1.0e6))
            name_zero = _traj_name(seed, Kmax_MPa, "zero")
            name_finite = _traj_name(seed, Kmax_MPa, "finite")
            if name_zero not in ledger["trajectories"] or name_finite not in ledger["trajectories"]:
                continue
            zero_traj = ledger["trajectories"][name_zero]
            finite_traj = ledger["trajectories"][name_finite]
            key = f"K{Kmax_MPa}MPa_seed{seed}"
            zero_events = zero_traj["events"]
            finite_events = finite_traj["events"]
            n_zero, n_finite = len(zero_events), len(finite_events)

            checks[f"{key}_event_count_identical"] = n_zero == n_finite
            checks[f"{key}_event_ordinal_sequence_exact_zero"] = (
                [e["event_index"] for e in zero_events] == list(range(n_zero))
            )
            checks[f"{key}_event_ordinal_sequence_exact_finite"] = (
                [e["event_index"] for e in finite_events] == list(range(n_finite))
            )
            checks[f"{key}_accepted_length_sequence_identical"] = n_zero == n_finite and all(
                abs(zero_events[i]["accepted_length_m"] - finite_events[i]["accepted_length_m"]) < 1.0e-15
                for i in range(min(n_zero, n_finite))
            )
            checks[f"{key}_R_matches_frozen_value"] = zero_traj["R"] == FROZEN_R and finite_traj["R"] == FROZEN_R
            checks[f"{key}_top_level_seed_matches_trajectory_name"] = (
                zero_traj["seed"] == seed and finite_traj["seed"] == seed
            )
            # hazard_event_index is a per-event ordinal identifying which
            # draw in the RNG threshold stream this event corresponds to --
            # NOT the same thing as a nested per-event RNG seed/state
            # snapshot (which the ledger does not archive at all; reported
            # explicitly below as not_archived_fields rather than assumed).
            checks[f"{key}_hazard_event_index_present_every_event"] = (
                all(e.get("hazard_event_index") is not None for e in zero_events)
                and all(e.get("hazard_event_index") is not None for e in finite_events)
            )
            checks[f"{key}_rebonding_config_hash_zero_matches_frozen"] = (
                zero_traj["rebonding_cfg_hash"] == expected_hash_zero
            )
            checks[f"{key}_rebonding_config_hash_finite_matches_frozen"] = (
                finite_traj["rebonding_cfg_hash"] == expected_hash_finite
            )
            checks[f"{key}_material_manifest_source_row_hash_zero"] = (
                zero_traj["manifest_audit"]["source_row_sha256"] == expected_material_row_hash
            )
            checks[f"{key}_material_manifest_source_row_hash_finite"] = (
                finite_traj["manifest_audit"]["source_row_sha256"] == expected_material_row_hash
            )
            manifest_hashes_seen.add(zero_traj["manifest_audit"]["compatibility_manifest_sha256"])
            manifest_hashes_seen.add(finite_traj["manifest_audit"]["compatibility_manifest_sha256"])

            gate_zero = stable_growth_gate(zero_events, frequency_Hz=FREQUENCY_HZ)
            gate_finite = stable_growth_gate(finite_events, frequency_Hz=FREQUENCY_HZ)
            gate_pass_all = gate_pass_all and gate_zero["stable_growth_provisional"] and gate_finite["stable_growth_provisional"]

            horizon_zero = effective_horizon_censored(
                zero_traj, max_accepted_events=MAX_ACCEPTED_EVENTS, max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
            )
            horizon_finite = effective_horizon_censored(
                finite_traj, max_accepted_events=MAX_ACCEPTED_EVENTS, max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
            )
            eff_censored_zero = (
                (horizon_zero["raw_censored"] or horizon_zero["stopped_by_budget_exhaustion"])
                and not gate_zero["stable_growth_provisional"]
            )
            eff_censored_finite = (
                (horizon_finite["raw_censored"] or horizon_finite["stopped_by_budget_exhaustion"])
                and not gate_finite["stable_growth_provisional"]
            )
            uncensored_all = uncensored_all and not eff_censored_zero and not eff_censored_finite

            hz_zero = [e["hazard_threshold_action"] for e in zero_events]
            hz_finite = [e["hazard_threshold_action"] for e in finite_events]
            checks[f"{key}_hazard_threshold_sequence_identical"] = (
                len(hz_zero) == len(hz_finite) and all(a == b for a, b in zip(hz_zero, hz_finite))
            )
            checks[f"{key}_zero_all_bulk_action_qualified"] = all(e["all_bulk_action_qualified"] for e in zero_events)
            checks[f"{key}_finite_all_bulk_action_qualified"] = all(e["all_bulk_action_qualified"] for e in finite_events)

            dev_z = gate_zero["developed_interval"]
            dev_f = gate_finite["developed_interval"]
            S_h_dev, dev_len_ident = _independent_S_h(dev_z, dev_f)
            checks[f"{key}_developed_length_identity_holds_before_simplification"] = dev_len_ident
            S_h_developed_by_K[Kmax] = S_h_dev

            idx_zero = true_final_half_indices(n_zero)
            idx_finite = true_final_half_indices(n_finite)
            w_zero = event_index_window_rate(zero_events, idx_zero, FREQUENCY_HZ)
            w_finite = event_index_window_rate(finite_events, idx_finite, FREQUENCY_HZ)
            S_h_tfh, tfh_len_ident = _independent_S_h(w_zero, w_finite)
            checks[f"{key}_true_final_half_length_identity_holds_before_simplification"] = tfh_len_ident
            checks[f"{key}_true_final_half_window_is_actual_tail"] = (
                idx_zero == list(range(n_zero - math.ceil(n_zero / 2), n_zero))
                and idx_finite == list(range(n_finite - math.ceil(n_finite / 2), n_finite))
            )
            S_h_true_final_half_by_K[Kmax] = S_h_tfh

            idx6_zero = true_final_six_indices(n_zero)
            idx6_finite = true_final_six_indices(n_finite)
            w6_zero = event_index_window_rate(zero_events, idx6_zero, FREQUENCY_HZ)
            w6_finite = event_index_window_rate(finite_events, idx6_finite, FREQUENCY_HZ)
            S_h_tf6, tf6_len_ident = _independent_S_h(w6_zero, w6_finite)
            checks[f"{key}_true_final_six_length_identity_holds_before_simplification"] = tf6_len_ident
            checks[f"{key}_true_final_six_window_is_actual_tail"] = (
                idx6_zero == list(range(max(n_zero - 6, 0), n_zero))
                and idx6_finite == list(range(max(n_finite - 6, 0), n_finite))
            )
            S_h_true_final_six_by_K[Kmax] = S_h_tf6

            w50_zero = extension_window_rate(zero_events, 50.0, FREQUENCY_HZ)
            w50_finite = extension_window_rate(finite_events, 50.0, FREQUENCY_HZ)
            S_h_50, len_ident_50 = _independent_S_h(w50_zero, w50_finite)
            checks[f"{key}_last_50_um_length_identity_holds_before_simplification"] = len_ident_50
            S_h_last_50_um_by_K[Kmax] = S_h_50

            w25_zero = extension_window_rate(zero_events, 25.0, FREQUENCY_HZ)
            w25_finite = extension_window_rate(finite_events, 25.0, FREQUENCY_HZ)
            S_h_25, len_ident_25 = _independent_S_h(w25_zero, w25_finite)
            checks[f"{key}_last_25_um_length_identity_holds_before_simplification"] = len_ident_25
            S_h_last_25_um_by_K[Kmax] = S_h_25

            # Cross-check the developed-window event-index selection and
            # the developed/all-event diagnostic non-aliasing this review
            # round specifically flagged, against the saved decision.
            dev_idx_zero_independent = developed_interval_event_indices(zero_events)
            dev_idx_finite_independent = developed_interval_event_indices(finite_events)
            checks[f"{key}_developed_indices_exclude_transient_events_zero"] = (
                0 not in dev_idx_zero_independent if n_zero > 4 else True
            )
            checks[f"{key}_developed_indices_exclude_transient_events_finite"] = (
                0 not in dev_idx_finite_independent if n_finite > 4 else True
            )
            saved_pair = None
            for seed_key in (f"seed_{seed}_analysis",):
                sa = decision.get(seed_key)
                if sa is not None:
                    saved_pair = sa["per_pair"].get(key)
            if saved_pair is not None:
                checks[f"{key}_developed_event_ids_zero_match_saved"] = (
                    dev_idx_zero_independent == saved_pair.get("developed_window_event_ids_zero")
                )
                checks[f"{key}_developed_event_ids_finite_match_saved"] = (
                    dev_idx_finite_independent == saved_pair.get("developed_window_event_ids_finite")
                )
                checks[f"{key}_developed_diagnostics_not_aliased_to_all_event"] = bool(
                    saved_pair.get("developed_window_diagnostics_is_not_alias_of_all_event")
                )
                for cohesion_label in ("zero", "finite"):
                    diag_key = f"{cohesion_label}_exposure_and_action" if cohesion_label == "zero" else "finite_exposure_and_action"
                    aw = saved_pair.get(diag_key, {}).get("action_weighted_K_rebond_true_inter_event_weighted", {})
                    checks[f"{key}_{cohesion_label}_inter_event_weighted_status_not_archived_all_event"] = (
                        aw.get("status") == "NOT_ARCHIVED"
                    )
                    dev_aw = saved_pair.get("diagnostics_developed_window", {}).get(cohesion_label, {}).get(
                        "action_weighted_K_rebond_true_inter_event_weighted", {}
                    )
                    checks[f"{key}_{cohesion_label}_inter_event_weighted_status_not_archived_developed"] = (
                        dev_aw.get("status") == "NOT_ARCHIVED"
                    )

        x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]

        def _fit_delta_m(y_by_K: dict[float, float | None]) -> float | None:
            if not all(y_by_K.get(K) is not None for K in KMAX_GRID_Pa_sqrt_m):
                return None
            y = [y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m]
            return three_point_slope_fit(x, y)["slope"]

        dev_delta_m = _fit_delta_m(S_h_developed_by_K)
        tfh_delta_m = _fit_delta_m(S_h_true_final_half_by_K)
        tf6_delta_m = _fit_delta_m(S_h_true_final_six_by_K)
        last50_delta_m = _fit_delta_m(S_h_last_50_um_by_K)
        last25_delta_m = _fit_delta_m(S_h_last_25_um_by_K)

        # Cross-check these extended windows' delta_m against the saved
        # decision's fits_by_window for the same seed (tight tolerance).
        saved_seed_analysis = decision.get(f"seed_{seed}_analysis")
        if saved_seed_analysis is not None:
            for wname, recomputed in (
                ("true_final_six", tf6_delta_m), ("last_50_um", last50_delta_m), ("last_25_um", last25_delta_m),
            ):
                saved_fit = saved_seed_analysis["fits_by_window"].get(wname)
                if saved_fit is None or recomputed is None:
                    checks[f"seed{seed}_{wname}_delta_m_reproducible"] = saved_fit is None and recomputed is None
                else:
                    checks[f"seed{seed}_{wname}_delta_m_reproducible"] = (
                        abs(saved_fit["delta_m_least_squares"] - recomputed) < 1.0e-9
                    )

        per_seed[seed] = {
            "uncensored": uncensored_all, "gate_pass": gate_pass_all,
            "developed_delta_m": dev_delta_m, "true_final_half_delta_m": tfh_delta_m,
            "true_final_six_delta_m": tf6_delta_m,
            "last_50_um_delta_m": last50_delta_m, "last_25_um_delta_m": last25_delta_m,
            "S_h_developed_by_K": S_h_developed_by_K,
        }

    checks["single_compatibility_manifest_used_across_all_trajectories"] = len(manifest_hashes_seen) <= 1

    both_seeds_pass = (
        1720 in per_seed and 1001723 in per_seed
        and per_seed[1720]["uncensored"] and per_seed[1720]["gate_pass"]
        and per_seed[1001723]["uncensored"] and per_seed[1001723]["gate_pass"]
    )
    delta_m_developed_by_seed = None
    S_h_developed_by_Kmax_by_seed = None
    true_terminal_delta_m_by_seed = None
    if both_seeds_pass:
        delta_m_developed_by_seed = {
            1720: per_seed[1720]["developed_delta_m"], 1001723: per_seed[1001723]["developed_delta_m"],
        }
        S_h_developed_by_Kmax_by_seed = {
            K: {1720: per_seed[1720]["S_h_developed_by_K"][K], 1001723: per_seed[1001723]["S_h_developed_by_K"][K]}
            for K in KMAX_GRID_Pa_sqrt_m
        }
        true_terminal_delta_m_by_seed = {
            1720: per_seed[1720]["true_final_half_delta_m"], 1001723: per_seed[1001723]["true_final_half_delta_m"],
        }

    primary = classify_developed_confirmation(
        both_seeds_pass_gate_uncensored=both_seeds_pass,
        delta_m_developed_by_seed=delta_m_developed_by_seed,
        S_h_developed_by_Kmax_by_seed=S_h_developed_by_Kmax_by_seed,
        slope_gate=SLOPE_GATE,
    )
    recomputed_classification = apply_tail_sensitivity_gate(
        primary, true_terminal_delta_m_by_seed=true_terminal_delta_m_by_seed, slope_gate=SLOPE_GATE,
    )
    details["independent_recomputed_primary_classification"] = primary["classification"]
    details["independent_recomputed_terminal_classification"] = recomputed_classification["classification"]
    details["independent_delta_m_developed_by_seed"] = delta_m_developed_by_seed
    details["independent_true_final_half_delta_m_by_seed"] = true_terminal_delta_m_by_seed

    return recomputed_classification, checks, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if sys.executable != pilot.REQUIRED_PYTHON:
        raise SystemExit(f"wrong interpreter: expected {pilot.REQUIRED_PYTHON!r}, got {sys.executable!r}")

    for name in TRACKED_ARTIFACT_NAMES:
        checks[f"artifact_present_{name}"] = (DEV_ARTIFACTS / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    protocol = json.loads((DEV_ARTIFACTS / "developed_confirmation_frozen_protocol.json").read_text())
    parent_frozen = json.loads((PARENT_PILOT_ARTIFACTS / "frozen_configuration.json").read_text())
    ledger = json.loads((DEV_ARTIFACTS / "event_ledger.json").read_text())
    decision = json.loads((DEV_ARTIFACTS / "developed_confirmation_decision.json").read_text())
    provenance_record = json.loads(
        (DEV_ARTIFACTS / "developed_confirmation_physical_producer_provenance.json").read_text()
    )

    checks["ledger_config_hash_matches_parent_frozen"] = (
        ledger["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )

    provenance = load_a_native_provenance()
    checks["complete_row_sha256_matches_frozen_protocol"] = (
        provenance["complete_row_sha256"]
        == protocol["material_and_provenance"]["complete_active_material_row_sha256"]
    )

    bare_engine, _ = build_a_native_engine(None)
    Eprime_Pa = reduced_modulus_Pa(bare_engine.G, bare_engine.nu)
    r_eff = max(bare_engine.r_eff(), 1.0e-9)
    recomputed_frozen = pilot.freeze_pilot_configuration(
        Eprime_Pa=Eprime_Pa, reference_contact_radius_m=r_eff,
        engine_G_Pa=bare_engine.G, engine_nu=bare_engine.nu,
        a_native_provenance_sha256=provenance["complete_row_sha256"],
    )
    checks["parent_frozen_configuration_reproducible"] = (
        recomputed_frozen["frozen_configuration_sha256"] == parent_frozen["frozen_configuration_sha256"]
    )

    for key in ("RB2_reversible_zero", "RB2_reversible_finite"):
        cfg = _config_from_frozen(parent_frozen, key)
        checks[f"{key}_config_hash_reproducible"] = cfg.config_hash() == parent_frozen["config_hashes"][key]
    checks["dmd_poincare_acceleration_disabled"] = (
        parent_frozen["common_settings"]["dmd_poincare_acceleration_enabled"] is False
    )
    checks["stability_definition_string_unaltered"] = (
        protocol["developed_growth_and_stationarity_gate"]["stability_definition_string_verbatim"]
        == STABILITY_DEFINITION_STRING
    )

    # Physical-producer provenance: the code that actually generated the
    # twelve trajectories must be byte-identical to what the provenance
    # record claims, verified from git history (not assumed to be HEAD).
    # Verify against the HISTORICAL commit's blob content (not the live
    # working file) -- this proves the producer_code_commit itself has not
    # been rewritten/corrupted, independent of legitimate later edits to
    # the same files in this evidence-hardening branch (e.g. new analysis-
    # only functions added to crack_rebonding_developed_confirmation_
    # v10230.py alongside the untouched stable_growth_gate).
    producer_commit = provenance_record["commit_timeline"]["producer_code_commit"]["sha"]
    for rel_path, expected_sha in provenance_record["producer_file_sha256_at_producer_commit"].items():
        if rel_path == "note" or not rel_path.endswith(".py"):
            continue
        actual_sha = sha256_git_blob(producer_commit, rel_path)
        checks[f"producer_file_at_producer_commit_matches_recorded_hash_{rel_path}"] = actual_sha == expected_sha

    # The two files that actually drive trajectory generation and gate
    # evaluation (run_developed_confirmation_stage.py and stable_growth_
    # gate's own module) must additionally be unchanged from
    # producer_code_commit all the way through v1_final_evidence_commit --
    # i.e. Stage 1 and Stage 2 genuinely ran identical physics code.
    v1_commit = provenance_record["commit_timeline"]["v1_final_evidence_commit"]["sha"]
    for rel_path in (
        "scripts/run_developed_confirmation_stage.py",
        "arrhenius_fracture/crack_rebonding_developed_confirmation_v10230.py",
    ):
        diff = subprocess.run(
            ["git", "diff", producer_commit, v1_commit, "--", rel_path], cwd=REPO_ROOT,
            check=True, capture_output=True,
        ).stdout
        checks[f"stage1_stage2_identical_producer_code_{rel_path}"] = diff == b""

    stage2_authorized = decision.get("stage_2_conditional_gate", {}).get("authorized", False)
    seeds_present = set()
    for seed in (1720, 1001723):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            Kmax_MPa = int(round(Kmax / 1.0e6))
            for cohesion in ("zero", "finite"):
                if _traj_name(seed, Kmax_MPa, cohesion) in ledger["trajectories"]:
                    seeds_present.add(seed)
    checks["stage_2_seed_absent_unless_authorized"] = (
        True if stage2_authorized else 1001723 not in seeds_present
    )

    # Independent recomputation path -- does NOT call
    # analyze_developed_confirmation.analyze_seed().
    recomputed_classification, independent_checks, independent_details = independent_recompute(
        ledger, protocol, decision,
    )
    checks.update(independent_checks)
    details.update(independent_details)

    recomputed_normalized = json.loads(json.dumps(recomputed_classification, sort_keys=True))
    saved_normalized = json.loads(json.dumps(decision["terminal_classification"], sort_keys=True))
    checks["terminal_classification_exact_match_independent_path"] = recomputed_normalized == saved_normalized
    details["saved_terminal_classification"] = decision["terminal_classification"]["classification"]
    details["independently_recomputed_terminal_classification"] = recomputed_classification["classification"]

    checks["multi_K_paris_slope_campaign_not_authorized"] = (
        decision.get("multi_K_paris_slope_campaign_authorized", False) is False
    )
    checks["part_x_not_authorized"] = decision.get("part_x_authorized", False) is False
    checks["production_merge_not_authorized"] = (
        decision.get("production_line_merge_authorized", False) is False
    )

    file_hashes = {name: sha256_file(DEV_ARTIFACTS / name) for name in TRACKED_ARTIFACT_NAMES}
    (DEV_ARTIFACTS / "file_hashes.json").write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    overall_pass = all(checks.values())
    verification = {
        "schema": "v10.2.30_crack_rebonding_developed_confirmation_verification_v3",
        "depends_on_gitignored_run_files": False,
        "independence_level": "INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES",
        "independent_recomputation_path": (
            "reimplements window selection (developed/true_final_half/true_final_six/"
            "last_50_um/last_25_um)/S_h/delta_m/classification-input assembly directly "
            "against the shared, reused primitives (stable_growth_gate, "
            "event_index_window_rate, extension_window_rate, true_final_half_indices, "
            "true_final_six_indices, three_point_slope_fit, "
            "classify_developed_confirmation, apply_tail_sensitivity_gate); does not "
            "call analyze_developed_confirmation.analyze_seed(). This is NOT a "
            "from-scratch reimplementation of those shared primitives themselves -- "
            "reimplementing the campaign's own frozen/reused gate and classifier "
            "formulas would defeat the point of reusing them verbatim."
        ),
        "checks": checks,
        "details": details,
        "file_hashes": file_hashes,
        "overall_pass": overall_pass,
    }
    verification_path = DEV_ARTIFACTS / "verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(f"wrote {verification_path}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
