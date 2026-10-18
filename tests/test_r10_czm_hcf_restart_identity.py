from scripts.run_r10_czm_hcf import restart_identity_matches


def identity(phase_count=16):
    return {
        "candidate_id": "candidate",
        "factor": 1.01,
        "temperature_K": 300.0,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "plastic_feedback": True,
        "phase_count": phase_count,
        "hazard_seed": 17,
    }


def test_restart_identity_binds_phase_count():
    saved = identity()
    assert restart_identity_matches(
        saved, "candidate", 1.01, 300.0, 0.1, 1000.0, True, 16, 17)
    assert not restart_identity_matches(
        saved, "candidate", 1.01, 300.0, 0.1, 1000.0, True, 64, 17)


def test_restart_identity_binds_hazard_seed():
    saved = identity()
    assert not restart_identity_matches(
        saved, "candidate", 1.01, 300.0, 0.1, 1000.0, True, 16, 18)


def test_legacy_restart_is_adopted_once():
    saved = identity()
    del saved["phase_count"]
    del saved["hazard_seed"]
    assert restart_identity_matches(
        saved, "candidate", 1.01, 300.0, 0.1, 1000.0, True, 64, 19)
