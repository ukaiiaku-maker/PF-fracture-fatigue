from pathlib import Path
from zipfile import ZipFile

from experimental_fits.ast2018.config import AstAssumptions
from experimental_fits.ast2018.data import AstData


ARCHIVE=Path(__file__).resolve().parents[1]/"Ast2018_tungsten_joint_fit_data.zip"


def test_inferred_displacement_speeds_respect_reported_protocol(tmp_path):
    with ZipFile(ARCHIVE) as archive:
        archive.extract("Ast2018_joint_fit_data/data/table1_conditions.csv",tmp_path)
        for name in ("table1_toughness_long.csv","J_R_curves.csv",
                     "K_displacement_curves.csv","relative_stiffness.csv",
                     "no_cleavage_endpoints.csv","source_discrepancies.csv"):
            archive.extract(f"Ast2018_joint_fit_data/data/{name}",tmp_path)
    data=AstData.load(tmp_path/"Ast2018_joint_fit_data")
    assumptions=AstAssumptions()
    for condition in data.conditions:
        specimen=assumptions.specimen(condition)
        v=specimen.initial_displacement_rate(
            float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"]))
        assert 5e-9*(1-1e-12)<=v<=150e-9*(1+1e-12)
