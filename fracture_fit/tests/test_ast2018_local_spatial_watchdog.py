"""Direct spatial refinement must reject an incomplete required condition."""

from experimental_fits.ast2018 import local_spatial
from pathlib import Path
import pytest
import json


def test_profile_seeded_local_refinement_has_isolated_checkpoint_paths():
    root=Path('runs/example')
    default=local_spatial._output_paths(root,'FIGURE5_ALTERNATIVE')
    seeded=local_spatial._output_paths(root,'FIGURE5_ALTERNATIVE','profile_seeded')
    assert not set(default).intersection(seeded)
    assert seeded[2].name=='best_1d_refined_figure5_alternative_profile_seeded.json'
    with pytest.raises(ValueError):
        local_spatial._output_paths(root,'FIGURE5_ALTERNATIVE','../unsafe')


def test_completed_local_rounds_write_best_vector_after_candidate_selection(
        tmp_path,monkeypatch):
    scenario='FIGURE5_ALTERNATIVE'
    tag='profile_seeded'
    output,state_path,best_path,_=local_spatial._output_paths(tmp_path,scenario,tag)
    start=tmp_path/'profile_vector.json'
    x=[.5]*7
    start.write_text(json.dumps({'x':x}))
    (tmp_path/'config_figure5_alternative.json').write_text(json.dumps({
        'ast_assumptions':{}}))
    state_path.write_text(json.dumps({'round':3,'x':x,'cost':72.0,
        'step':.04,'watchdog_limit_s':3600.0,'indices':[0,3,6],
        'source_analytical_vector':str(start),'condition_ids':list(local_spatial.PRIORITY11)}))
    output.write_text(json.dumps({'key':local_spatial._key(x),'x':x,
                                  'cost':72.0,'error':None})+'\n')
    class EmptyPool:
        def __init__(self,**_kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*_args):return False
        def submit(self,*_args):raise AssertionError('completed search must not rerun')
    monkeypatch.setattr(local_spatial,'ProcessPoolExecutor',EmptyPool)
    actual=local_spatial.local_refine(tmp_path,tmp_path,scenario,start,
                                      workers=1,rounds=3,tag=tag)
    assert actual==best_path
    assert json.loads(best_path.read_text())['cost_priority11']==72.0


def test_spatial_local_score_stops_after_required_numerical_failure(monkeypatch):
    class FailedObjective:
        case_calls=0
        progress_callback=None

        def evaluate(self,_x,*,detail):
            self.case_calls+=1
            self.progress_callback("required_condition","numerical_maximum_steps",
                                   self.case_calls)
            raise AssertionError("a failed 1-D case must end the candidate")

    monkeypatch.setattr(local_spatial,"_CTX",FailedObjective())
    row=local_spatial._score([.5]*7,100)
    assert row["cost"]==1e300
    assert row["error"]=="NumericalReject: required_condition: numerical_maximum_steps"
    assert not row["watchdog_timeout"]
    assert row["case_calls"]==1
    assert local_spatial._CTX.progress_callback is None
