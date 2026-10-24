"""A later retained vector must preserve and replace its old final evidence."""
import json

from experimental_fits.ast2018.selection import (
    archive_superseded_final,retained_vector_path)


def test_profile_seeded_vector_precedence_and_archival(tmp_path):
    stem='figure5_alternative'
    initial=tmp_path/f'best_1d_refined_{stem}.json'
    selected=tmp_path/f'best_1d_refined_{stem}_profile_seeded.json'
    initial.write_text(json.dumps({'x':[.1]}))
    selected.write_text(json.dumps({'x':[.2]}))
    assert retained_vector_path(tmp_path,'FIGURE5_ALTERNATIVE')==selected
    final=tmp_path/f'confirmation_{stem}'
    final.mkdir()
    (final/'vector.json').write_text(json.dumps({'x':[.1]}))
    ensemble=tmp_path/f'ensemble_{stem}_spatial'
    ensemble.mkdir()
    (ensemble/'runs.json').write_text('old evidence')
    archive_superseded_final(tmp_path,'FIGURE5_ALTERNATIVE',[.2])
    archive=tmp_path/f'superseded_{stem}_before_profile_seeded'
    assert (archive/f'confirmation_{stem}/vector.json').exists()
    assert (archive/f'ensemble_{stem}_spatial/runs.json').read_text()=='old evidence'
    final.mkdir()
    (final/'vector.json').write_text(json.dumps({'x':[.2]}))
    archive_superseded_final(tmp_path,'FIGURE5_ALTERNATIVE',[.2])
    assert json.loads((final/'vector.json').read_text())['x']==[.2]
