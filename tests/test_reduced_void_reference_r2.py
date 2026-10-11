from arrhenius_fracture.reduced_void_reference_r2 import (
    DONOR_COMMIT, DONOR_TREE, donor_receipt,
)


def test_reduced_void_donor_is_exact_commit_and_keeps_birth_paths_distinct():
    receipt = donor_receipt()
    assert receipt["commit"] == DONOR_COMMIT
    assert receipt["tree"] == DONOR_TREE
    assert receipt["source_worktree_dirty_ignored"] is True
    assert set(receipt["birth_pathways"]) == {
        "vacancy_diffusion", "direct_dislocation_interaction",
        "cavity_surface_emission",
    }
    assert receipt["birth_pathways"]["direct_dislocation_interaction"] != receipt[
        "birth_pathways"
    ]["cavity_surface_emission"]
