import csv
import json
from pathlib import Path
import pytest
from arrhenius_fracture import prospective_paris_transfer_engine_v10230 as engine
from arrhenius_fracture import sharp_front_v10_2_30_prospective_transfer_fixed_deltaK as entry
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import CLEAVAGE_FIELDS,IDENTITY_FIELDS,digest,select_frozen_option

IDS=['P40_TRANSFER_CALIBRATED_GEN2','P25_TRANSFER_V1_RANK1','P55_TRANSFER_V1_RANK1']


@pytest.mark.parametrize('cid',IDS)
def test_frozen_transfer_row_manifest_identity(cid):
    selected=engine.select_transfer_option(cid)
    native=select_frozen_option('A_NATIVE')
    assert {k for k in selected.row if selected.row[k]!=native.row[k]} == CLEAVAGE_FIELDS|IDENTITY_FIELDS
    m,audit=engine.build_transfer_manifest(cid)
    assert audit['candidate_row_sha256']==digest(selected.row)
    assert audit['material_manifest_sha256']==digest(m.as_dict())
    assert not audit['rebonding'] and not audit['PT_substitution']


def _copy_freeze(monkeypatch,tmp_path):
    original=engine.REGISTRY
    registry=tmp_path/original.name;registry.write_bytes(original.read_bytes())
    freeze=tmp_path/engine.FREEZE.name;freeze.write_bytes(engine.FREEZE.read_bytes())
    selection=tmp_path/engine.SELECTION.name;selection.write_bytes(engine.SELECTION.read_bytes())
    monkeypatch.setattr(engine,'REGISTRY',registry);monkeypatch.setattr(engine,'FREEZE',freeze);monkeypatch.setattr(engine,'SELECTION',selection)
    return registry,freeze


@pytest.mark.parametrize('field,value',[('c_blunt','99'),('peierls_H0_eV','0.1'),('taylor_exp_n','10'),('Tref_K','481.33'),('physics__persistent_backstress_scale','2')])
def test_noncleavage_change_refused_even_with_rehashed_prospective_row(monkeypatch,tmp_path,field,value):
    registry,freeze=_copy_freeze(monkeypatch,tmp_path)
    rows=list(csv.DictReader(registry.open()));rows[0][field]=value
    with registry.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    d=json.loads(freeze.read_text());d['registry_sha256']=engine.sha256_file(registry);d['candidates'][0]['complete_row_sha256']=digest(rows[0]);freeze.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='non-cleavage'):
        engine.select_transfer_option(rows[0]['candidate_id'],registry)


def test_second_transfer_update_refused(monkeypatch,tmp_path):
    registry,freeze=_copy_freeze(monkeypatch,tmp_path);d=json.loads(freeze.read_text());d['transfer_update_number']=2;freeze.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='only one'):
        engine.select_transfer_option(IDS[0],registry)


def test_ineligible_row_refused(monkeypatch,tmp_path):
    registry,freeze=_copy_freeze(monkeypatch,tmp_path);d=json.loads(freeze.read_text());d['candidates'][0]['eligible']=False;freeze.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='eligible'):
        engine.select_transfer_option(IDS[0],registry)


def test_production_selection_reaches_frozen_row_and_restores(monkeypatch,tmp_path):
    p=tmp_path/'high_cycle_run_manifest.json';p.write_text('{}')
    original=entry.paper._SOURCE_SELECT_OPTION
    monkeypatch.setattr(entry.fixed,'main',lambda args: entry.paper._select_option_four_class(IDS[0],engine.REGISTRY))
    result=entry.main(['--out',str(tmp_path),'--parameter-option',IDS[0]])
    assert result.candidate_id==IDS[0]
    assert entry.paper._SOURCE_SELECT_OPTION is original
    assert json.loads(p.read_text())['prospective_candidate']['candidate_id']==IDS[0]


def test_no_virgin_directory_precreation(tmp_path):
    out=tmp_path/'virgin'
    with pytest.raises(ValueError,match='claim virgin'):
        entry.main(['--out',str(out),'--parameter-option',IDS[0]])
    assert not out.exists()
