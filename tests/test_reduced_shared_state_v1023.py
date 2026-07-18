from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.physical_shielding_v1022 import (
    install_uncapped_physical_shielding,
)
from arrhenius_fracture.reduced_shared_state_v1023 import (
    FALLBACK_ROLES,
    SharedReducedConfig,
    build_shared_engine,
    fallback_manifest_path,
    fallback_registry,
    load_manifest,
    replay_shared_state,
    write_shared_result,
)


def _small_config() -> SharedReducedConfig:
    return SharedReducedConfig(
        mpz_length_um=4.0,
        mpz_n_bins=8,
        wake_length_um=4.0,
        wake_n_bins=8,
        blunting_length_um=0.5,
        max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596),
    ).validate()


def test_fallback_manifests_are_preserved_and_loadable():
    registry = fallback_registry()
    assert registry["schema"] == "v10.2.3_fallback_DBTT_parameterizations"
    assert {row["candidate_id"] for row in registry["parameterizations"]} == set(
        FALLBACK_ROLES
    )
    for candidate, role in FALLBACK_ROLES.items():
        path = fallback_manifest_path(candidate)
        assert path.is_file()
        manifest = load_manifest(candidate_id=candidate)
        assert manifest.candidate_id == candidate
        assert manifest.name == "DBTT"
        assert manifest.c_blunt > 0.0
        assert manifest.max_K_shield_MPa_sqrt_m == 1.0
        record = next(
            row for row in registry["parameterizations"]
            if row["candidate_id"] == candidate
        )
        assert record["role"] == role


def test_shared_engine_uses_spatial_production_state_and_anisotropic_emitter():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    assert engine.mpz.mobile.shape == (2, 8)
    assert engine.mpz.retained.shape == (2, 8)
    assert engine.mpz.accumulated_slip.shape == (2, 8)
    assert engine.mpz._emit.__func__.__name__ == "_anisotropic_campaign_emit"
    # validated_scalar keeps the inherited common transport operator
    assert engine.mpz.evolve.__func__.__name__ == "evolve"
    assert np.allclose(
        engine.mpz._anisotropic_drive_factors,
        np.asarray([0.132886, 0.008596]),
    )


def test_uncapped_shared_engine_uses_raw_field_above_legacy_reference():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="full")
    engine.mpz.retained[0, 0] = 100.0
    with install_uncapped_physical_shielding():
        raw = float(engine._active_shielding_raw_uncapped())
        effective = float(engine._active_shielding_signed())
    legacy = manifest.max_K_shield_MPa_sqrt_m * 1.0e6
    assert raw > legacy
    assert effective == raw


def test_shielding_off_is_a_real_instance_ablation():
    manifest = load_manifest(candidate_id="DBTT_A0003837")
    engine = build_shared_engine(manifest, _small_config(), mode="shielding_off")
    engine.mpz.retained[0, 0] = 100.0
    with install_uncapped_physical_shielding():
        assert float(engine._active_shielding_signed()) == 0.0
        assert float(engine.K_shield()) == 0.0


def test_backstress_off_sets_zero_campaign_backstress():
    manifest = load_manifest(candidate_id="DBTT_A0002333")
    engine = build_shared_engine(manifest, _small_config(), mode="backstress_off")
    engine.mpz.mobile[0, 0] = 10.0
    engine.mpz.retained[0, 0] = 10.0
    engine.mpz.evolve(1.0e-12, 500.0, 2.0e9, _small_config().b_m)
    assert np.allclose(engine.mpz.anisotropic_last_sigma_back_by_system_Pa, 0.0)


def test_replay_is_deterministic_and_writes_spatial_final_state(tmp_path: Path):
    manifest = load_manifest(candidate_id="DBTT_A0002277")
    schedule = [
        {
            "dt_s": 1.0e-8,
            "temperature_K": 650.0,
            "K_Pa_sqrt_m": 10.0e6,
            "drive_factor_0": 0.132886,
            "drive_factor_1": 0.008596,
        },
        {
            "dt_s": 2.0e-8,
            "temperature_K": 650.0,
            "K_Pa_sqrt_m": 12.0e6,
            "drive_factor_0": 0.132886,
            "drive_factor_1": 0.008596,
        },
    ]
    first = replay_shared_state(manifest, schedule, _small_config(), mode="full")
    second = replay_shared_state(manifest, schedule, _small_config(), mode="full")
    assert first["raw_equals_effective_when_shielding_active"] is True
    assert second["raw_equals_effective_when_shielding_active"] is True
    assert len(first["history"]) == 2
    for name in first["_final_arrays"]:
        assert np.array_equal(first["_final_arrays"][name], second["_final_arrays"][name])

    out = tmp_path / "replay"
    write_shared_result(first, out)
    assert (out / "shared_reduced_summary.json").is_file()
    assert (out / "shared_reduced_audit.json").is_file()
    assert (out / "shared_reduced_history.csv").is_file()
    state_path = out / "shared_reduced_final_state.npz"
    assert state_path.is_file()
    state = np.load(state_path)
    assert state["mobile"].shape == (2, 8)
    audit = json.loads((out / "shared_reduced_audit.json").read_text())
    assert audit["constitutive_K_shield_clip_applied"] is False
    assert audit["legacy_manifest_cap_used_in_kinetics"] is False
    assert audit["spatial_final_state_saved"] is True
