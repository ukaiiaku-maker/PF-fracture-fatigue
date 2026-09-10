"""Independent prelaunch integrity gate; not a campaign-completion verifier."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/prospective_paris_candidates'
FREEZE = '636d107524106989cb40feed7b8d42464c2fac9e'


def verify():
    import sys
    sys.path.insert(0, str(ROOT))
    from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import build_frozen_manifest
    amendment = json.loads((OUT / 'p40_launch_pathway_amendment.json').read_text())
    subprocess.run(['git', 'merge-base', '--is-ancestor', FREEZE, 'HEAD'], cwd=ROOT, check=True)
    for path, expected in amendment['unchanged_frozen_hashes'].items():
        original = subprocess.check_output(['git', 'show', f'{FREEZE}:{path}'], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == expected, f'freeze hash: {path}'
        assert (ROOT / path).read_bytes() == original, f'frozen file changed: {path}'
    for path, expected in amendment['source_hashes'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, f'source hash: {path}'
    path = 'arrhenius_fracture/parameter_registry_v9111.py'
    assert (ROOT / path).read_bytes() == subprocess.check_output(['git', 'show', f'{FREEZE}:{path}'], cwd=ROOT)
    _, audit = build_frozen_manifest()
    assert audit['material_manifest_sha256'] == amendment['material_manifest_sha256']
    assert audit['candidate_row_sha256'] == amendment['scientific_candidate_row_sha256']
    return {'passed': True, 'scope': 'prelaunch_only', 'frozen_files': len(amendment['unchanged_frozen_hashes'])}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
