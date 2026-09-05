"""PX0 -- source and evidence freeze for the v10.2.30 crack-rebonding Part X
compression-conditioned physical campaign (signed-K contact surrogate).

Inventories and hashes the eight inherited crack-rebonding studies, records
the authoritative production-engine construction (real MRO, material row,
n_phase/mpz_n_bins/wake_n_bins, existing rebonding config hashes), and writes
the five PX0 provenance/scope files under artifacts/crack_rebonding_part_x_v1/
before any Part X source or physical result exists.

Run once, from a clean worktree checked out at the authoritative base
(a72d46557f5eba45c8c2e0a428574e5b9b624c81), then commit its output as PX0.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BASE_SHA = "a72d46557f5eba45c8c2e0a428574e5b9b624c81"
OUT_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

INHERITED_STUDIES = {
    "causal_pilot": {
        "branch": "codex/v10.2.30-crack-rebonding-causal-pilot",
        "artifact_dir": None,  # superseded in-place by causal_pilot_v2's directory
    },
    "causal_pilot_v2": {
        "branch": "codex/v10.2.30-crack-rebonding-causal-pilot-v2",
        "artifact_dir": "crack_rebonding_causal_pilot_v2",
    },
    "minimal_slope_screen": {
        "branch": "codex/v10.2.30-crack-rebonding-minimal-slope-screen",
        "artifact_dir": "crack_rebonding_minimal_slope_screen_v1",
    },
    "slope_exposure_continuation": {
        "branch": "codex/v10.2.30-crack-rebonding-slope-exposure-continuation",
        "artifact_dir": "crack_rebonding_slope_exposure_continuation",
    },
    "developed_confirmation": {
        "branch": "codex/v10.2.30-crack-rebonding-developed-confirmation",
        "artifact_dir": "crack_rebonding_developed_confirmation",
    },
    "developed_decomposition_repair": {
        "branch": "codex/v10.2.30-crack-rebonding-developed-decomposition-repair",
        "artifact_dir": "crack_rebonding_developed_confirmation",  # repaired in place
    },
    "developed_evidence_hardening": {
        "branch": "codex/v10.2.30-crack-rebonding-developed-evidence-hardening",
        "artifact_dir": "crack_rebonding_developed_confirmation",  # hardened in place
    },
    "static_shield_attribution": {
        "branch": "codex/v10.2.30-crack-rebonding-static-shield-attribution",
        "artifact_dir": "crack_rebonding_static_shield_attribution",
    },
    "static_shield_evidence_hardening": {
        "branch": "codex/v10.2.30-crack-rebonding-static-shield-evidence-hardening",
        "artifact_dir": "crack_rebonding_static_shield_attribution",  # hardened in place
    },
}

PRODUCTION_SOURCE_FILES = [
    "arrhenius_fracture/persistent_site_cyclic_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_coupled_v10229.py",
    "arrhenius_fracture/persistent_site_coupled_hazard_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_coupled_audited_v10229.py",
    "arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py",
    "arrhenius_fracture/persistent_site_cyclic_energy_gated_corrected_v10230.py",
    "arrhenius_fracture/crack_rebonding_v10230.py",
    "arrhenius_fracture/crack_rebonding_kinetics_v10230.py",
    "arrhenius_fracture/a_native_engine_v10230.py",
    "arrhenius_fracture/material_manifest.py",
    "arrhenius_fracture/reduced_shared_state_v1023.py",
]


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_show(rev: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=REPO_ROOT, capture_output=True
    )
    if result.returncode != 0:
        return None
    return result.stdout


def build_source_provenance() -> dict:
    file_hashes = {}
    for rel in PRODUCTION_SOURCE_FILES:
        p = REPO_ROOT / rel
        file_hashes[rel] = sha256_path(p) if p.exists() else None

    from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine, load_a_native_provenance
    from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CONTACT_SEMANTICS_LABEL

    engine, manifest_audit = build_a_native_engine()
    mro = [f"{c.__module__}.{c.__name__}" for c in type(engine).__mro__]
    provenance = load_a_native_provenance()
    row_sha256 = provenance["complete_row_sha256"]

    return {
        "schema": "v10230_crack_rebonding_part_x_source_provenance_v1",
        "authoritative_source_branch": (
            "codex/v10.2.30-crack-rebonding-static-shield-evidence-hardening"
        ),
        "authoritative_full_head_sha": BASE_SHA,
        "part_x_branch": "codex/v10.2.30-crack-rebonding-part-x",
        "required_python": (
            "/opt/homebrew/Caskroom/miniconda/base/envs/"
            "arrhenius-sharp-front-v10-codex/bin/python"
        ),
        "material_row": {
            "option_key": "A_NATIVE",
            "candidate_id": manifest_audit["candidate_id"],
            "mpz_length_um": manifest_audit["mpz_length_um"],
            "mpz_n_bins": manifest_audit["mpz_n_bins"],
            "complete_active_material_row_sha256": row_sha256,
            "compatibility_manifest_sha256": manifest_audit["compatibility_manifest_sha256"],
            "reconstruction_classification": manifest_audit["reconstruction_classification"],
        },
        "production_engine": {
            "top_class": mro[0],
            "mro": mro,
            "n_phase": 80,
            "mpz_n_bins": int(manifest_audit["mpz_n_bins"]),
            "wake_n_bins": int(getattr(engine.mpz, "wake_n_bins", -1)),
            "contact_model_label": CONTACT_SEMANTICS_LABEL,
        },
        "production_source_file_sha256": file_hashes,
    }


def build_inherited_inventory() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    artifact_hashes: dict[str, dict] = {}
    seen_dirs: set[str] = set()

    for study, meta in INHERITED_STUDIES.items():
        branch = meta["branch"]
        branch_sha = run(["git", "rev-parse", branch])
        is_ancestor = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", branch_sha, BASE_SHA],
                cwd=REPO_ROOT,
            ).returncode
            == 0
        )
        art_dir = meta["artifact_dir"]
        n_files = 0
        dir_sha256_of_listing = None
        shared_with = ""
        if art_dir and art_dir not in seen_dirs:
            seen_dirs.add(art_dir)
            full = REPO_ROOT / "artifacts" / art_dir
            if full.exists():
                file_map = {}
                for f in sorted(full.rglob("*")):
                    if f.is_file():
                        rel = str(f.relative_to(REPO_ROOT))
                        file_map[rel] = sha256_path(f)
                artifact_hashes[art_dir] = file_map
                n_files = len(file_map)
                dir_sha256_of_listing = sha256_bytes(
                    json.dumps(file_map, sort_keys=True).encode("utf-8")
                )
        elif art_dir in seen_dirs:
            # Directory already hashed under an earlier study in this same
            # loop (in-place repair/hardening passes share one artifact
            # directory with the study they amended) -- not zero files,
            # just not re-listed here to avoid a duplicate hash entry.
            n_files = len(artifact_hashes.get(art_dir, {}))
            dir_sha256_of_listing = sha256_bytes(
                json.dumps(artifact_hashes.get(art_dir, {}), sort_keys=True).encode("utf-8")
            )
            shared_with = art_dir
        rows.append(
            {
                "study": study,
                "branch": branch,
                "branch_full_sha": branch_sha,
                "is_ancestor_of_base": is_ancestor,
                "artifact_dir": art_dir or "",
                "tracked_file_count": n_files,
                "artifact_listing_sha256": dir_sha256_of_listing or "",
                "shares_artifact_dir_hashed_under": shared_with,
            }
        )

    # causal_pilot_v2's branch tip carries exactly one file beyond BASE
    # (regime_equivalence_analysis.json); record it read-only for reference
    # without merging that branch.
    extra_path = "artifacts/crack_rebonding_causal_pilot_v2/regime_equivalence_analysis.json"
    extra_blob = git_show("codex/v10.2.30-crack-rebonding-causal-pilot-v2", extra_path)
    extra_record = None
    if extra_blob is not None:
        extra_record = {
            "path": extra_path,
            "source_branch": "codex/v10.2.30-crack-rebonding-causal-pilot-v2",
            "sha256": sha256_bytes(extra_blob),
            "present_in_base": (REPO_ROOT / extra_path).exists(),
            "key_finding": (
                "classification=REVERSIBLE_PERSISTENT_EQUIVALENT_FOR_SLOPE_SCREEN "
                "at the OLD causal-pilot-v2 RB2 presets (reversible A_on=A_off=1; "
                "persistent A_on=10,A_off=0.1): both seeds passed the equivalence "
                "gate (|S_h difference| < 0.0002 decade, threshold 0.005). PX2's "
                "COMPETING_PERSISTENT row must therefore NOT reuse these exact "
                "presets unmodified -- it needs materially reduced rupture "
                "relative to COMPETING_REVERSIBLE to actually separate from it "
                "under PX2's own quantitative gates (section 6.3)."
            ),
        }

    return rows, {"artifact_hashes": artifact_hashes, "causal_pilot_v2_tip_only_file": extra_record}


def build_mission_scope() -> dict:
    return {
        "schema": "v10230_crack_rebonding_part_x_mission_scope_v1",
        "mission_name": "SIGNED_K_CRACK_REBONDING_PART_X",
        "in_scope": [
            "A: at least one genuinely nonsaturated clean-surface reversible rebonding regime",
            "B: reversible vs persistent clean-surface kinetics distinguished or shown indistinguishable",
            "C: RB3 PASSIVATION_GATED_REBOND exercised end to end (active P<->C<->B)",
            "D: developed sensitivity to Kmax, R, frequency, minimum-load dwell, "
            "kinetic chemistry factor, restored cohesive strength",
            "E: mechanism classification of each response axis",
            "F: manuscript-grade censor-aware tables, figures, provenance, strict verifier",
        ],
        "out_of_scope": [
            "RESOLVED_GAP_TRACTION / resolved opposing-face contact",
            "plastic-wake closure at positive nominal R",
            "crack-length reduction / crack-surface removal / topological healing",
            "HAZARD_AND_ENERGY_GATE_COUPLED feedback mode",
            "stochastic healing",
            "closure-corrected DeltaK_eff reporting",
            "production-line merge",
        ],
        "contact_model_label": "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT",
        "retained_prior_classifications": {
            "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT": (
                "scoped ONLY to the existing SAT_EXISTING row (A_NATIVE, T=300K, "
                "R=-0.95, f=1000Hz, Kmax=15/18/21 MPa sqrt(m), RB2 reversible, "
                "K_rebond_max=0.9 MPa sqrt(m), seeds 1720/1001723). Not generalized "
                "to new nonsaturated or passivation-limited regimes before testing."
            ),
            "STATIC_DYNAMIC_LOCALIZER_PARITY_UNRESOLVED": (
                "retained until the static-control event localization is placed on, "
                "or demonstrated equivalent to, the exact phase-resolved event-time "
                "path (PX1.4)."
            ),
        },
        "existing_developed_reference": {
            "material": "A_NATIVE", "T_K": 300.0, "R": -0.95, "frequency_Hz": 1000.0,
            "Kmax_MPa_sqrt_m": [15.0, 18.0, 21.0],
            "K_rebond_max_MPa_sqrt_m": 0.9,
            "seeds": [1720, 1001723],
            "accepted_events": 30, "total_extension_um": 150.0,
            "delta_m_seed_1720": 0.687925, "delta_m_seed_1001723": 0.687358,
            "approximate_rate_reduction_pct": {"15": 25, "18": 9, "21": 6},
        },
        "persistent_worktree": (
            "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/"
            "v10230-crack-rebonding-part-x"
        ),
        "persistent_run_root": "runs/crack_rebonding_part_x_v1",
        "tracked_evidence_root": "artifacts/crack_rebonding_part_x_v1",
        "max_physical_workers": 3,
    }


def build_completion_contract() -> dict:
    return {
        "schema": "v10230_crack_rebonding_part_x_completion_contract_v1",
        "terminal_label": "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE",
        "required_qualifiers": [
            "CLEAN_REVERSIBLE_KINETICS_QUALIFIED",
            "PERSISTENT_KINETICS_DISTINGUISHED|PERSISTENT_KINETICS_NOT_DISTINGUISHED",
            "PASSIVATION_GATED_REBONDING_QUALIFIED|PASSIVATION_EFFECT_BELOW_RESOLUTION",
            "FREQUENCY_SENSITIVITY_QUALIFIED|FREQUENCY_SENSITIVITY_SMALL",
            "DWELL_SENSITIVITY_QUALIFIED|DWELL_SENSITIVITY_SMALL",
            "KINETIC_CHEMISTRY_SENSITIVITY_QUALIFIED|KINETIC_CHEMISTRY_SENSITIVITY_SMALL",
            "COHESIVE_STRENGTH_SENSITIVITY_QUALIFIED|COHESIVE_STRENGTH_SENSITIVITY_SMALL",
            "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT|"
            "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY|DYNAMIC_REBONDING_HISTORY_REQUIRED",
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT",
            "HAZARD_ONLY_COHESIVE_FEEDBACK",
            "TOPOLOGICAL_HEALING_NOT_MODELED",
            "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
        ],
        "commit_sequence": [
            "PX0 provenance and completion contract",
            "PX1 dwell/RB3/instrumentation/localizer software qualification",
            "PX2 kinetic rows and analytical prediction freeze",
            "PX3 single-K physical screen and adaptive selection",
            "PX4 developed multi-K Stage 1",
            "PX5 required second-seed developed confirmation",
            "PX6 static/dynamic attribution",
            "PX7 final analysis, figures, portable verifier, and decision",
        ],
        "gates": {
            "screen_measurable_effect": "|S_h| >= 0.01 decade AND |S_h| > max(0.005, 5x numerical/action bound)",
            "developed_stationarity": {
                "exclude_first_um": 20, "minimum_total_extension_um": 100,
                "assess_last_um": 50, "minimum_developed_events": 10,
                "late_early_rate_ratio_bounds": [0.5, 2.0],
            },
            "response_classes": [
                "REBONDING_NO_MEASURABLE_RATE_EFFECT", "REBONDING_RATE_OFFSET_LIKE",
                "REBONDING_STEEPENS_RESPONSE", "REBONDING_FLATTENS_RESPONSE",
                "REBONDING_CURVED_ONSET_OR_CROSSOVER", "REBONDING_SEED_SENSITIVE",
            ],
            "static_dynamic_attribution": {
                "dominant_max_S_h_diff_decade": 0.01, "dominant_max_delta_m_diff": 0.10,
                "mixed_max_S_h_diff_decade": 0.03, "mixed_max_delta_m_diff": 0.25,
            },
        },
        "trajectory_budget": {
            "max_accepted_events": 30, "max_extension_um": 150, "max_cycles": 1.0e12,
            "max_concurrent_workers": 3, "resume_authorized": False,
        },
        "not_authorized": [
            "resolved-face-contact mission", "topological healing",
            "closure-corrected DeltaK_eff", "automatic production-line merge",
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_provenance = build_source_provenance()
    (OUT_DIR / "source_provenance.json").write_text(
        json.dumps(source_provenance, indent=2, sort_keys=False) + "\n"
    )

    rows, hashes_payload = build_inherited_inventory()
    with (OUT_DIR / "inherited_result_inventory.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "inherited_artifact_hashes.json").write_text(
        json.dumps(hashes_payload, indent=2, sort_keys=True) + "\n"
    )

    (OUT_DIR / "mission_scope.json").write_text(
        json.dumps(build_mission_scope(), indent=2, sort_keys=False) + "\n"
    )
    (OUT_DIR / "completion_contract.json").write_text(
        json.dumps(build_completion_contract(), indent=2, sort_keys=False) + "\n"
    )

    file_hashes = {}
    for name in (
        "source_provenance.json", "inherited_result_inventory.csv",
        "inherited_artifact_hashes.json", "mission_scope.json", "completion_contract.json",
    ):
        file_hashes[name] = sha256_path(OUT_DIR / name)
    (OUT_DIR / "file_hashes.json").write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")

    print("PX0 provenance written to", OUT_DIR)
    for k, v in file_hashes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
