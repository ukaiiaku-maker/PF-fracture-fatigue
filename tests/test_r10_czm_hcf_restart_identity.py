from scripts.run_r10_czm_hcf import restart_identity_matches


def identity(phase_count=16):
    return {
        "candidate_id": "candidate",
        "complete_row_sha256": "hash",
        "factor": 1.01,
        "temperature_K": 300.0,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "plastic_feedback": True,
        "phase_count": phase_count,
        "hazard_seed": 17,
        "patch_m": 1e-6,
        "observation_path_m": 120e-6,
    }


def test_restart_identity_binds_phase_count():
    saved = identity()
    assert restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        16, 17, 1e-6, 120e-6)
    assert not restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        64, 17, 1e-6, 120e-6)


def test_restart_identity_binds_hazard_seed():
    saved = identity()
    assert not restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        16, 18, 1e-6, 120e-6)


def test_restart_identity_binds_material_and_observation_geometry():
    saved = identity()
    assert not restart_identity_matches(
        saved, "candidate", "other", 1.01, 300.0, 0.1, 1000.0, True,
        16, 17, 1e-6, 120e-6)
    assert not restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        16, 17, 2e-6, 120e-6)
    assert not restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        16, 17, 1e-6, 60e-6)


def test_legacy_restart_is_adopted_once():
    saved = identity()
    del saved["phase_count"]
    del saved["hazard_seed"]
    del saved["complete_row_sha256"]
    del saved["patch_m"]
    del saved["observation_path_m"]
    assert restart_identity_matches(
        saved, "candidate", "hash", 1.01, 300.0, 0.1, 1000.0, True,
        64, 19, 2e-6, 60e-6)


def test_restart_identity_seals_explicit_environment_alternative():
    saved=identity();saved["environment_stiffness_Pa_per_m"]=1.25e17
    assert restart_identity_matches(
        saved,"candidate","hash",1.01,300.,.1,1000.,True,
        16,17,1e-6,120e-6,1.25e17)
    assert not restart_identity_matches(
        saved,"candidate","hash",1.01,300.,.1,1000.,True,
        16,17,1e-6,120e-6,2.5e17)
    assert not restart_identity_matches(
        saved,"candidate","hash",1.01,300.,.1,1000.,True,
        16,17,1e-6,120e-6,None)
    control=identity()
    assert not restart_identity_matches(
        control,"candidate","hash",1.01,300.,.1,1000.,True,
        16,17,1e-6,120e-6,1.25e17)
