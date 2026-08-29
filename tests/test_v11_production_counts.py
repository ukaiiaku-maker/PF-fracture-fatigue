from pathlib import Path

import pytest

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.production_counts_v11 import production_front_counts


def test_preserved_step481_distinguishes_branch_objects_from_births():
    path = Path("runs/v11_canonical_45deg_700K_seed3621_1000um_long_growth_v1/checkpoint/latest.json")
    if not path.is_file():
        pytest.skip("preserved step-481 production fixture is unavailable")
    checkpoint = restore_branch_checkpoint(path)
    counts = production_front_counts(checkpoint.state)
    assert counts["network_branch_object_count"] == 15
    assert counts["committed_branch_birth_count"] == 7
    assert counts["maximum_branch_births"] == 8
    assert counts["active_front_count"] == 6
