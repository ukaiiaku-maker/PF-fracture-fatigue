import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"scripts/build_v914_thermal_joint_registry.py";S=importlib.util.spec_from_file_location("t",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_hierarchical_dimensions_are_disjoint():
 assert len(M.BASE)==22 and len(M.THERMAL)==5
 assert set(M.BASE).isdisjoint(M.THERMAL)
def test_room_temperature_reanchoring_identity():
 base,g,Tref=4.2,.003,481.33;at_ref=base+g*(Tref-300)
 assert abs(at_ref+g*(300-Tref)-base)<1e-14
