"""Fixed-vector sensitivity survives a launcher restart without repeating cases."""
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from experimental_fits.ast2018 import sensitivity


def test_variant_checkpoint_skips_completed_forward_calls(tmp_path,monkeypatch):
    scenario="TABLE_PRIMARY"
    (tmp_path/"config_table_primary.json").write_text(json.dumps(
        {"ast_assumptions":{}}))
    vector=tmp_path/"best_table_primary.json"
    vector.write_text(json.dumps({"x":[.5]*7}))
    mods=[None,None,None,SimpleNamespace(CommonPhysics=object)]
    monkeypatch.setattr(sensitivity,"load_native",lambda _: (mods,{}))
    candidate=SimpleNamespace(rho_source0_m2=2.0)
    physics=SimpleNamespace(cleavage_nu0_s=1.0,emission_nu0_s=1.0)
    monkeypatch.setattr(sensitivity,"historical_start",
                        lambda *args:(candidate,physics,{}))
    calls=[]
    def fake_variant(spec):
        name=spec[5]
        calls.append(name)
        return {"variant":name,"kind":spec[6],"override":spec[7],
                "cost":1.0,"runtime_s":0.0,"forward_case_calls":26,
                "detail":[],"interpretation":"test"}
    monkeypatch.setattr(sensitivity,"_variant_task",fake_variant)

    class Pool:
        def __init__(self,*,max_workers):self.max_workers=max_workers
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,fn,spec):
            future=Future();future.set_result(fn(spec));return future
    monkeypatch.setattr(sensitivity,"ProcessPoolExecutor",Pool)
    first=sensitivity.analyze(tmp_path,tmp_path,scenario,vector,workers=2)
    assert len(calls)==19
    assert len(first)==27
    second=sensitivity.analyze(tmp_path,tmp_path,scenario,vector,workers=2)
    assert len(calls)==19
    assert second==first
