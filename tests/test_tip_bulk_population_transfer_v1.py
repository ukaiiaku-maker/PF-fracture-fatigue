import numpy as np
import pytest

from arrhenius_fracture.tip_bulk_population_transfer_v1 import (
    conservative_tip_bulk_transfer,
)


def test_parameterized_tip_to_bulk_transfer_is_positive_and_conservative():
    tip=np.array([4.0,3.0,2.0,1.0]); bulk=np.array([2.0e12,1.0e12])
    area=np.array([1.0e-12,2.0e-12])
    new_tip,new_bulk,row=conservative_tip_bulk_transfer(
        tip,bulk,area,direction="tip_to_bulk",requested_line_content=5.0,
        tip_support_weights=np.ones(4),bulk_support_weights=np.array([1.0,3.0]),
    )
    assert row["transferred_line_content"]==5.0
    assert np.sum(new_tip)+np.sum(new_bulk*area)==pytest.approx(
        np.sum(tip)+np.sum(bulk*area),abs=1e-13,
    )
    assert np.all(new_tip>=0.0) and np.all(new_bulk>=0.0)
    assert row["physical_transfer_coefficient"] is None


def test_parameterized_bulk_to_tip_transfer_caps_at_available_support():
    tip=np.zeros(3); bulk=np.array([2.0,4.0]); area=np.array([1.0,1.0])
    new_tip,new_bulk,row=conservative_tip_bulk_transfer(
        tip,bulk,area,direction="bulk_to_tip",requested_line_content=20.0,
        tip_support_weights=np.array([1.0,2.0,1.0]),bulk_support_weights=np.ones(2),
    )
    assert row["limited_by_available_population"]
    assert row["transferred_line_content"]==6.0
    assert np.sum(new_tip)==pytest.approx(6.0)
    assert np.sum(new_bulk*area)==pytest.approx(0.0,abs=1e-15)


@pytest.mark.parametrize("direction",["tip_to_bulk","bulk_to_tip"])
def test_zero_transfer_is_exact_noop(direction):
    tip=np.array([1.0,2.0]); bulk=np.array([3.0]); area=np.array([2.0])
    a,b,row=conservative_tip_bulk_transfer(
        tip,bulk,area,direction=direction,requested_line_content=0.0,
        tip_support_weights=np.ones(2),bulk_support_weights=np.ones(1),
    )
    np.testing.assert_array_equal(a,tip); np.testing.assert_array_equal(b,bulk)
    assert row["conservation_residual_line_content"]==0.0
