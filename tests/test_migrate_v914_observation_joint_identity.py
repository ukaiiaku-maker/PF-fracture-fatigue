from pathlib import Path
def test_migration_keeps_three_lineage_axes_distinct():
 s=(Path(__file__).parents[1]/"scripts/migrate_v914_observation_joint_identity.py").read_text()
 assert 'obs["base_candidate_id"]' in s and 'obs["thermal_profile_id"]' in s and 'obs["joint_candidate_id"]' in s
