import ast
import json
from pathlib import Path
import subprocess


def test_full_diagnostics_default_off_preserves_existing_production_source():
    path='arrhenius_fracture/primary_race_production_v13.py'
    before=ast.parse(subprocess.check_output(['git','show','f09ec84:'+path],text=True))
    after=ast.parse(Path(path).read_text())
    old=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_production_mark')
    new=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_production_mark')
    assert new.args.kwonlyargs[-1].arg=='evaluate_expired_diagnostics'
    assert new.args.kw_defaults[-1].value is False
    new.args=old.args
    class StripDiagnostic(ast.NodeTransformer):
        def visit_If(self,node):
            if isinstance(node.test,ast.Name) and node.test.id=='evaluate_expired_diagnostics':return None
            if isinstance(node.test,ast.BoolOp) and any(isinstance(n,ast.Name) and n.id=='evaluate_expired_diagnostics' for n in ast.walk(node.test)):
                node.test=node.test.values[0]
            return self.generic_visit(node)
    assert ast.dump(old)==ast.dump(StripDiagnostic().visit(new))


def test_source_parent_and_owner_guard_unchanged():
    for path in ('arrhenius_fracture/sharp_front_v11_branching.py','arrhenius_fracture/tip_directional_observation_v11.py',
                 'arrhenius_fracture/inherited_primary_race_v13.py'):
        assert Path(path).read_bytes()==subprocess.check_output(['git','show','f09ec84:'+path])


def test_preregistered_orientation_screen_and_followup_gate():
    from scripts.run_v13_transition_screen import PLAN
    p=json.loads(PLAN.read_text())
    assert p['orientation_order_deg']==[15,30] and len(p['screening_cases'])==8
    assert p['screen_seed']==3621 and p['maximum_workers']==2
    assert p['target_unbranched_um']==75
    assert not p['new_committed_branches_in_screen']
    assert p['no_new_theta40_seeds'] and p['no_family_build'] and p['no_branch_parameter_changes']
    assert all('theta40' not in c for c in p['screening_cases'])


def test_freeze_and_provenance_separate_commits():
    from scripts.run_v13_transition_screen import OUT,verify_frozen
    p=json.loads((OUT/'accepted_ensemble_freeze.json').read_text())
    assert len(p['per_case_execution_commits'])==16
    assert len({p['record_commit'],p['corrected_execution_source_commit'],p['final_report_producer_code_commit']})==3
    assert verify_frozen()==len(p['files'])
