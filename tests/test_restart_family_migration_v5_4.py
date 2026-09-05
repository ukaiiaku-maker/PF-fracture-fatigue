from pathlib import Path

import pytest

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.restart_family_migration_v11 import (
    canonical_hash, migrate_shared_engine_family,
)
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)
from scripts.qualify_pf_branching_integrated_envelope_v5_4 import (
    SOURCE_FAMILY_SHA, TARGET_FAMILY_SHA, TARGET_PHYSICS,
    initialized_engine, migrate_checkpoint, physical_identity,
)


OLD = Path(
    "/private/tmp/current-source-branching-replay-preflight-v5/analysis_outputs/"
    "pf_current_source_branching_execution_seal_v5_1/pinned_inputs/"
    "theta40_signed_kernel_family.json"
)
NEW = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_theta40_signed_kernel_append_only_v5_3_"
    "20260901T120000Z_final/family.json"
)
BASE = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_current_source_branching_capability_20260829/"
    "theta40_matched_pair"
)
CONTROL = BASE / (
    "theta40_control_max1_seed3621/checkpoint/transitions/"
    "step0000001_mesh_adaptation_g0002.json"
)
ENABLED = BASE / (
    "theta40_enabled_max2_seed3621/checkpoint/transitions/"
    "step0000001_mesh_adaptation_g0002.json"
)


pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in (OLD, NEW, CONTROL, ENABLED)),
    reason="immutable V5.4 direct-source fixtures are not mounted",
)


def _families():
    return (
        ActiveOnlySigned2DShieldingKernelFamily.from_json(OLD),
        ActiveOnlySigned2DShieldingKernelFamily.from_json(NEW),
    )


@pytest.mark.parametrize("checkpoint_path", [CONTROL, ENABLED])
def test_actual_step1_restart_family_migration_is_state_exact(checkpoint_path: Path):
    old, new = _families()
    checkpoint = restore_branch_checkpoint(checkpoint_path)
    migrated, audit = migrate_checkpoint(checkpoint, old, new)
    assert audit["qualification"] == "PASS"
    assert audit["migration_count"] == 1
    assert audit["process_state_sha256_before"] == audit["process_state_sha256_after"]
    assert audit["rng_threshold_sha256_before"] == audit["rng_threshold_sha256_after"]
    assert audit["engine_mpz_same_bound_family_object"] is True
    assert audit["old_family_reachable_after_migration"] is False
    before = physical_identity(checkpoint)
    after = physical_identity(migrated)
    assert before == after


def test_actual_migrated_pair_remains_policy_only():
    old, new = _families()
    control, _ = migrate_checkpoint(restore_branch_checkpoint(CONTROL), old, new)
    enabled, _ = migrate_checkpoint(restore_branch_checkpoint(ENABLED), old, new)
    left = physical_identity(control); right = physical_identity(enabled)
    assert left.pop("branching_policy") is False
    assert right.pop("branching_policy") is True
    assert left == right


def test_actual_migration_rejects_second_application():
    old, new = _families()
    checkpoint = restore_branch_checkpoint(CONTROL)
    engine = initialized_engine(checkpoint, new)
    engine, _ = migrate_shared_engine_family(
        engine, checkpoint.shared_process_state,
        source_family=old, target_family=new,
        source_family_sha256=SOURCE_FAMILY_SHA,
        target_family_sha256=TARGET_FAMILY_SHA,
        target_family_physics_fingerprint=TARGET_PHYSICS,
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        migrate_shared_engine_family(
            engine, checkpoint.shared_process_state,
            source_family=old, target_family=new,
            source_family_sha256=SOURCE_FAMILY_SHA,
            target_family_sha256=TARGET_FAMILY_SHA,
            target_family_physics_fingerprint=TARGET_PHYSICS,
        )
