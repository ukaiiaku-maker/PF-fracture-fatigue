import ast
from pathlib import Path
import subprocess


def test_canonical_parent_body_unchanged_except_default_off_capture():
    path = "arrhenius_fracture/sharp_front_v11_branching.py"
    old = ast.parse(subprocess.check_output(["git", "show", "55955722db40ef65c43d2f1c08357ad8f77b82fa:"+path], text=True))
    new = ast.parse(Path(path).read_text())
    before = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "run_2d")
    after = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "run_2d")
    assert [a.arg for a in after.args.kwonlyargs] == ["parent_capture", "inherited_primary_race"]
    assert isinstance(after.args.kw_defaults[0], ast.Constant) and after.args.kw_defaults[0].value is None
    assert isinstance(after.args.kw_defaults[1], ast.Constant) and after.args.kw_defaults[1].value is False
    after.args = before.args
    class RemoveCapture(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("probe_tip_id", "probe_tip"):
                # Demonstrated pre-existing NameError: diagnostic aliases for
                # the exact controlling observation bound by the process hook.
                assert isinstance(node.value, ast.Attribute) and node.value.value.id == "controlling"
                return None
            return self.generic_visit(node)
        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id in ("parent_capture", "inherited_primary_race") for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)
    after = RemoveCapture().visit(after)
    assert ast.dump(before) == ast.dump(after)


def test_first_parent_capture_has_no_historical_restore_path():
    from scripts.run_v13_clean_parents import FirstParentCapture
    import inspect
    source = inspect.getsource(FirstParentCapture)
    assert "restore_branch_checkpoint" not in source
    assert "state.competition.consumed_event_ids" in source
    assert "_persistent_emit" in source


def test_physical_sensitivity_preregistered_before_launch():
    import json
    plan = json.loads(Path("analysis_outputs/v13_physical_companion_qualification/preregistered_plan.json").read_text())
    assert len(plan["cases"]) == 8
    assert plan["theta_deg"] == 40 and plan["maximum_active_fronts_parent"] == 1
    assert plan["quenched_disorder_J"] == 0
    assert not plan["sampler_only_Monte_Carlo"] and not plan["historical_atlas_continuation"]
    assert len(plan["beta_B_log10_grid"]) == 13


def test_actual_entry_uses_the_same_pinned_four_class_registry():
    import inspect
    from scripts.run_v13_clean_parents import case_run, ROWS
    text = inspect.getsource(case_run)
    assert "pf_v2_four_class_pf_transfer_registry.csv" in text
    assert "pf_v2_four_class_pf_transfer_selection.json" in text
    assert ROWS["weakT"][0] == "oneD_v2_focused_weak_T_0016"
    assert ROWS["ceramic"][0] == "oneD_v2_focused_ceramic_like_0018"


def test_extracted_process_hook_keeps_diagnostic_tip_aliases_defined():
    tree = ast.parse(Path("arrhenius_fracture/sharp_front_v11_branching.py").read_text())
    update = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "update_shared")
    aliases = {node.targets[0].id: ast.unparse(node.value) for node in update.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert aliases["probe_tip_id"] == "controlling.tip_id"
    assert aliases["probe_tip"] == "controlling.tip_xy_m"


def test_first_parent_terminal_label_is_supported_without_rerun(tmp_path):
    import json
    from arrhenius_fracture.branch_output_v11 import BranchOutputWriter
    writer = BranchOutputWriter(tmp_path)
    writer.complete(status="v13_first_baseline_cleavage_captured", final_checkpoint="parent/accepted_single.json", validation={"first_cleavage": True})
    assert json.loads((tmp_path/"run_complete.json").read_text())["status"] == "v13_first_baseline_cleavage_captured"


def test_every_driver_topology_snapshot_call_supplies_required_arguments():
    import inspect
    from arrhenius_fracture.branch_snapshot_v11 import write_topology_snapshot
    signature=inspect.signature(write_topology_snapshot)
    tree=ast.parse(Path('arrhenius_fracture/sharp_front_v11_branching.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='write_topology_snapshot']
    assert calls
    for call in calls:
        signature.bind(*([None]*len(call.args)),**{kw.arg:None for kw in call.keywords})


def test_existing_daughter_stop_status_can_be_published_without_mechanics(tmp_path):
    import json
    from arrhenius_fracture.branch_output_v11 import BranchOutputWriter
    BranchOutputWriter(tmp_path).complete(status='qualified_daughter_early_stop',
        final_checkpoint='checkpoint/latest.json',validation={'existing_checkpoint':True})
    assert json.loads((tmp_path/'run_complete.json').read_text())['status']=='qualified_daughter_early_stop'
