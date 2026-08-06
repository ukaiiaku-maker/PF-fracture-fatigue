import math

from arrhenius_fracture.crack_network_v11 import CrackBranchState, CrackNetworkState
from arrhenius_fracture.network_metrics_v11 import crack_growth_metrics


def branch(name, parent, generation, points):
    return CrackBranchState(
        branch_id=name, parent_branch_id=parent, generation=generation,
        initiation_event=generation, path=tuple(points),
        orientation_history_rad=tuple(0.0 for _ in range(max(1, len(points) - 1))),
    )


def metric(*branches):
    network = CrackNetworkState(tuple(branches), primary_branch_id="root", branching_enabled=len(branches) > 1)
    return crack_growth_metrics(network, initial_crack_length_m=0.0)


def test_straight_100um_crack_is_100um():
    result = metric(branch("root", None, 0, ((0.0, 0.0), (100e-6, 0.0))))
    assert math.isclose(result.max_root_to_tip_path_extension_m, 100e-6)


def test_parallel_100um_daughters_do_not_sum_for_target():
    root = branch("root", None, 0, ((0.0, 0.0),))
    upper = branch("upper", "root", 1, ((0.0, 0.0), (100e-6, 0.0)))
    lower = branch("lower", "root", 1, ((0.0, 0.0), (0.0, 100e-6)))
    result = metric(root, upper, lower)
    assert math.isclose(result.network_total_new_crack_length_m, 200e-6)
    assert math.isclose(result.max_root_to_tip_path_extension_m, 100e-6)


def test_parent_100um_plus_child_50um_is_150um():
    root = branch("root", None, 0, ((0.0, 0.0), (100e-6, 0.0)))
    child = branch("child", "root", 1, ((100e-6, 0.0), (150e-6, 0.0)))
    assert math.isclose(metric(root, child).max_root_to_tip_path_extension_m, 150e-6)


def test_branch_enumeration_does_not_change_metric():
    root = branch("root", None, 0, ((0.0, 0.0), (100e-6, 0.0)))
    a = branch("a", "root", 1, ((100e-6, 0.0), (130e-6, 40e-6)))
    z = branch("z", "root", 1, ((100e-6, 0.0), (150e-6, 0.0)))
    assert metric(root, a, z) == metric(z, root, a)
