import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from experimental_fits.ast2018.config import AstAssumptions
from experimental_fits.ast2018.confirm import confirm_all


ARCHIVE=Path(__file__).resolve().parents[1]/"Ast2018_tungsten_joint_fit_data.zip"


def test_paired_confirmation_reuses_only_complete_matching_cases(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith("Ast2018_joint_fit_data/data/") and name.endswith(".csv"):
                archive.extract(name,tmp_path)
    data_dir=tmp_path/"Ast2018_joint_fit_data"
    run_dir=tmp_path/"runs"
    output=run_dir/"confirmation_table_primary_resume_regression"
    condition="Tp21C_rate0p2"
    for level in ("reduced","spatial"):
        folder=output/level/condition
        folder.mkdir(parents=True)
        (folder/"summary.json").write_text(json.dumps({
            "level":level,"condition_id":condition,"status":"complete",
            "numerical_failure":None})+"\n")
        (folder/"mechanical_history.csv").write_text("time_s\n0\n")
        (folder/"ast_observer.csv").write_text("time_s\n0\n")
        (folder/"events.json").write_text("[]\n")
    assumptions=AstAssumptions()
    x=[.5]*7
    rows=confirm_all(Path("/missing/native/source"),data_dir,run_dir,"TABLE_PRIMARY",
                     x,assumptions,levels=("reduced","spatial"),workers=1,
                     condition_ids=[condition],stage="resume_regression")
    assert len(rows)==2
    assert {r["level"] for r in rows}=={"reduced","spatial"}
    with pytest.raises(RuntimeError,match="vector or convention changed"):
        confirm_all(Path("/missing/native/source"),data_dir,run_dir,"TABLE_PRIMARY",
                    [.4]*7,assumptions,levels=("reduced","spatial"),workers=1,
                    condition_ids=[condition],stage="resume_regression")
