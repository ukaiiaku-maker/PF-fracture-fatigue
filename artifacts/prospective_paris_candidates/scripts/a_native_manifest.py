"""Construct the current, fully-provenanced A_NATIVE MaterialManifest from
artifacts/prospective_paris_candidates/A_native_provenance.json (copied from
codex/v10.2.30-crack-rebonding-part-x's artifacts/crack_rebonding_causal_pilot_v2/,
where it was deterministically reconstructed and hash-verified back to the
immutable, git-tracked v9.14 knee-search source row -- see that JSON's
provenance_chain for the full audit trail). Not a guess: complete_row_sha256
is independently re-verified below.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from arrhenius_fracture.material_manifest import ExpFloorBarrier, TransportBarrier, MaterialManifest

PROV_PATH = Path(__file__).resolve().parents[1] / "A_native_provenance.json"


def _f(row: dict, key: str) -> float:
    return float(row[key])


def load_a_native_manifest() -> tuple[MaterialManifest, dict]:
    doc = json.loads(PROV_PATH.read_text())
    row = doc["complete_active_material_row"]
    canon = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recomputed = hashlib.sha256(canon).hexdigest()
    if recomputed != doc["complete_row_sha256"]:
        raise ValueError(f"A_NATIVE row hash mismatch: {recomputed} != {doc['complete_row_sha256']}")

    emission = ExpFloorBarrier(
        G00_eV=_f(row, "emit_G00_eV"), gT_eV_per_K=_f(row, "emit_gT_eV_per_K"),
        sigc0_Pa=_f(row, "emit_sigc0_GPa") * 1.0e9, sT_Pa_per_K=_f(row, "emit_sT_GPa_per_K") * 1.0e9,
        alpha=_f(row, "emit_exp_a"), exponent=_f(row, "emit_exp_n"),
        floor_fraction=_f(row, "emit_floor_frac"), attempt_frequency_s=1.0e11,
    )
    cleavage = ExpFloorBarrier(
        G00_eV=_f(row, "cleave_G00_eV"), gT_eV_per_K=_f(row, "cleave_gT_eV_per_K"),
        sigc0_Pa=_f(row, "cleave_sigc0_GPa") * 1.0e9, sT_Pa_per_K=_f(row, "cleave_sT_GPa_per_K") * 1.0e9,
        alpha=_f(row, "cleave_exp_a"), exponent=_f(row, "cleave_exp_n"),
        floor_fraction=_f(row, "cleave_floor_frac"), attempt_frequency_s=1.0e12,
    )
    manifest = MaterialManifest(
        name="A_NATIVE", candidate_id="A_NATIVE", cleavage=cleavage, emission=emission,
        peierls=TransportBarrier(
            H0_eV=_f(row, "peierls_H0_eV"), activation_entropy_kB=_f(row, "peierls_activation_entropy_kB"),
            alpha=_f(row, "peierls_exp_a"), exponent=_f(row, "peierls_exp_n"),
            attempt_frequency_s=_f(row, "peierls_nu0_s"),
        ),
        taylor=TransportBarrier(
            H0_eV=_f(row, "taylor_H0_eV"), activation_entropy_kB=_f(row, "taylor_activation_entropy_kB"),
            alpha=_f(row, "taylor_exp_a"), exponent=_f(row, "taylor_exp_n"),
            attempt_frequency_s=_f(row, "taylor_nu0_s"),
        ),
        taylor_corr_rho_c_m2=_f(row, "taylor_corr_rho_c_m2"), taylor_corr_scale=_f(row, "taylor_corr_scale"),
        source_sites_per_system=_f(row, "source_sites_per_system"),
        encounter_efficiency=_f(row, "encounter_efficiency"),
        retained_recovery_rate_s=_f(row, "retained_recovery_rate_s"),
        source_refresh_length_m=_f(row, "source_refresh_length_um") * 1.0e-6,
        c_blunt=_f(row, "c_blunt"), max_K_shield_MPa_sqrt_m=_f(row, "max_K_shield_MPa_sqrt_m"),
    )
    extra = dict(
        rho_source0_m2=_f(row, "rho_source0_m2"),
        physics__cleavage_hits=_f(row, "physics__cleavage_hits"),
        physics__cleavage_correlation_time_s=_f(row, "physics__cleavage_correlation_time_s"),
        physics__blunting_length_m=_f(row, "physics__blunting_length_m"),
        qualified_production_head=doc["qualified_production_head"],
        complete_row_sha256=doc["complete_row_sha256"],
        source_file=str(PROV_PATH),
    )
    return manifest, extra


if __name__ == "__main__":
    manifest, extra = load_a_native_manifest()
    print(json.dumps(dict(cleavage=manifest.cleavage.__dict__, extra=extra), indent=2, default=str))
