from pathlib import Path
from zipfile import ZipFile

from experimental_fits.ast2018.data import AstData


ARCHIVE = Path(__file__).resolve().parents[1]/"Ast2018_tungsten_joint_fit_data.zip"


def test_conflict_scenario_changes_only_two_onsets(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith("Ast2018_joint_fit_data/data/") and name.endswith(".csv"):
                archive.extract(name,tmp_path)
    data = AstData.load(tmp_path/"Ast2018_joint_fit_data")
    primary = data.toughness_scenario("TABLE_PRIMARY")
    alternate = data.toughness_scenario("FIGURE5_ALTERNATIVE")
    changed = [(a["condition_id"],a["observable"],a["value"],b["value"])
               for a,b in zip(primary,alternate) if a["value"] != b["value"]]
    assert changed == [
        ("Tp75C_rate1p6","Kq_Jcontinuous_onset","7.6","8.858"),
        ("Tp100C_rate1p6","Kq_Jcontinuous_onset","5.8","7.5652"),
    ]
