import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"scripts/audit_v914_fracture_atlas_join.py";S=importlib.util.spec_from_file_location("audit_join",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_mapping_partitions_unified_vector():
 assert len(M.ATLAS_COORDS)==22
 assert set(M.THERMAL)=={"Tref_K","cleave_gT_eV_per_K","cleave_sT_GPa_per_K","emit_gT_eV_per_K","emit_sT_GPa_per_K"}
 assert set(M.ATLAS_COORDS).isdisjoint(M.THERMAL)
def test_canonical_anchor_ids_are_exact():
 assert set(M.CANONICAL.values())=={"Peak-T","DBTT","weak-T","ceramic-like"}
