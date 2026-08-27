from arrhenius_fracture import sharp_front_v10_2_30_candidate_monotonic as entry


def test_candidate_monotonic_entry_has_no_physics_implementation():
    assert callable(entry.main)
    assert not hasattr(entry, "PersistentSiteStateResolvedTipEngine")
