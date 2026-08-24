from pathlib import Path
def test_selector_declares_five_balanced_roles():
 s=(Path(__file__).parents[1]/"scripts/select_v914_joint_fracture_batch.py").read_text()
 assert s.count('"REALISTIC_FATIGUE_')==4
 assert '"INTERMEDIATE_MAXIMIN"' in s
 assert '"SELECTED_NOT_LAUNCHED"' in s
