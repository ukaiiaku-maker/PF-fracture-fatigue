from scripts.report_v13_clock_and_pair_mechanism import ensemble_gate


def test_one_intermediate_case_can_pass_without_all_eight_intermediate():
    rows=[{'probability':p} for p in (0.,0.,0.,0.,.2,1.,1.,1.)]
    gate=ensemble_gate(rows,common_model_physically_defensible=True,exact_parent_parity=True,
        exact_pair_and_fallback=True,stable_ordering=True)
    assert gate['status']=='PASS_SHORT_ENSEMBLE_WARRANTED'


def test_intermediate_probability_alone_does_not_qualify_physics():
    gate=ensemble_gate([{'probability':.2},{'probability':0}])
    assert gate['status']=='NOT_PASSED_NO_ENSEMBLE'
    assert 'common_pair_model_kinetic_scales_not_physically_established' in gate['reasons']
