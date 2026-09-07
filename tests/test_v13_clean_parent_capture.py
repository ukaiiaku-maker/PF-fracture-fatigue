import ast
from pathlib import Path
import subprocess


def test_canonical_parent_body_unchanged_except_default_off_capture():
    path = "arrhenius_fracture/sharp_front_v11_branching.py"
    old = ast.parse(subprocess.check_output(["git", "show", "55955722db40ef65c43d2f1c08357ad8f77b82fa:"+path], text=True))
    new = ast.parse(Path(path).read_text())
    before = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "run_2d")
    after = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "run_2d")
    assert len(after.args.kwonlyargs) == 1 and after.args.kwonlyargs[0].arg == "parent_capture"
    assert isinstance(after.args.kw_defaults[0], ast.Constant) and after.args.kw_defaults[0].value is None
    after.args = before.args
    class RemoveCapture(ast.NodeTransformer):
        def visit_If(self, node):
            if any(isinstance(n, ast.Name) and n.id == "parent_capture" for n in ast.walk(node.test)):
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
