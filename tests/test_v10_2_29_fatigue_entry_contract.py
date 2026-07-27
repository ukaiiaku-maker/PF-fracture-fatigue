from arrhenius_fracture import sharp_front_v10_2_29_fatigue as entry


def test_four_canonical_parameterizations_are_unchanged():
    assert entry._paper.VALID_OPTIONS == {
        "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
        "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
        "v913_paper_weakT01_0129902_persistent_sites": "v913_zeroD_sobol_0129902",
        "v913_paper_ceramic01_0077080_persistent_sites": "v913_zeroD_sobol_0077080",
    }


def test_fatigue_preparation_disables_duplicate_state_paths():
    args = [
        "--fatigue-cycles",
        "--cyclic-mechanics",
        "--pz-spatial-state",
        "--pz-recovery-per-s",
        "0",
    ]
    entry._prepare_fatigue_args(args)
    assert "--no-cyclic-mechanics" in args
    assert "--no-pz-spatial-state" in args
    assert "--cyclic-mechanics" not in args
    assert "--pz-spatial-state" not in args


def test_stage3_overlay_hides_only_fatigue_flag():
    observed = {}

    def original(args):
        observed["args"] = list(args)
        args.append("--original-mutated")
        return 3621

    args = ["--mode", "2d", "--fatigue-cycles", "--max-fronts", "1"]
    seed = entry._fatigue_capable_stage3_validity(original, args)
    assert seed == 3621
    assert "--fatigue-cycles" not in observed["args"]
    assert "--fatigue-cycles" in args
    assert "--original-mutated" in args
