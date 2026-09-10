"""Frozen prospective rows: direct production manifest selection, default off.

Uses the SelectedResponseOption construction of accepted A_NATIVE commit
30db009ff7172728a6bdc885886f21b94cbec225. No solver or registry gate changes.
The fixed-DeltaK adapter retains the production kernel and engine stack.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

from .material_manifest import MaterialManifest
from .parameter_registry_v9111 import SelectedResponseOption, sha256_file, write_compatibility_manifest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/prospective_paris_candidates'
REGISTRY = ARTIFACTS / 'p40_candidate_registry.csv'
SELECTION = ARTIFACTS / 'p40_candidate_selection.json'
P40 = 'P40_RADIUS_SCALED_ENVELOPE_V1'
P40_SHA = 'eb1373bd5a124b296e8f12dcc2a2c07fd0bb9af42d1dd0f28f36179d2eda8281'
BACKGROUND_SHA = '04518e244b7a446a120ea6e2d15a68758cb54a225d099609deeb23da32679971'
CLEAVAGE_FIELDS = frozenset(('cleave_G00_eV', 'cleave_sigc0_GPa', 'cleave_exp_a', 'cleave_exp_n', 'cleave_floor_frac'))
IDENTITY_FIELDS = frozenset(('candidate_id', 'option_key', 'role', 'mechanism_summary', 'validation_status'))


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def select_frozen_option(option_key, registry_path=REGISTRY, **kwargs):
    """Select only the exact frozen P40 and recovered A_NATIVE control rows."""
    if Path(registry_path).resolve() != REGISTRY.resolve():
        raise ValueError('prospective construction requires the frozen registry')
    if option_key not in ('A_NATIVE', P40):
        raise ValueError('option is not eligible for frozen prospective construction')
    provenance = json.loads((ARTIFACTS / 'A_native_provenance.json').read_text())
    background = provenance['complete_active_material_row']
    if digest(background) != BACKGROUND_SHA or provenance['complete_row_sha256'] != BACKGROUND_SHA:
        raise ValueError('A_NATIVE background hash mismatch')
    freeze = json.loads((ARTIFACTS / 'prediction_freeze_manifest.json').read_text())
    if freeze['complete_candidate_row_sha256'] != P40_SHA:
        raise ValueError('P40 freeze identity mismatch')
    for filename, key in ((REGISTRY, 'registry_csv_sha256'), (SELECTION, 'selection_json_sha256')):
        if sha256_file(filename) != freeze['launcher_artifacts'][key]:
            raise ValueError('frozen launcher artifact hash mismatch')
    with REGISTRY.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or {r['option_key'] for r in rows} != {'A_NATIVE', P40}:
        raise ValueError('frozen registry option identity mismatch')
    candidate = next(r for r in rows if r['option_key'] == P40)
    if digest(candidate) != P40_SHA or candidate != freeze['complete_candidate_row']:
        raise ValueError('P40 complete row hash mismatch')
    changed = {k for k in set(background) | set(candidate) if background.get(k) != candidate.get(k)}
    if changed != CLEAVAGE_FIELDS | IDENTITY_FIELDS:
        raise ValueError('candidate differs outside the five cleavage and identity fields')
    native = next(r for r in rows if r['option_key'] == 'A_NATIVE')
    if native != background:
        raise ValueError('A_NATIVE registry row differs from recovered background')
    row = native if option_key == 'A_NATIVE' else candidate
    return SelectedResponseOption(
        option_key=option_key, candidate_id=row['candidate_id'], material_class=row['material_class'],
        role=row.get('role', ''), mechanism_summary=row.get('mechanism_summary', ''),
        validation_status=row.get('validation_status', ''),
        mpz_length_um=float(row['L_pz_um_recommended']),
        mpz_n_bins=int(float(row['n_bins_recommended'])), row=dict(row),
        registry_path=str(REGISTRY), registry_sha256=sha256_file(REGISTRY),
    )


def build_frozen_manifest(option_key=P40):
    selected = select_frozen_option(option_key)
    with tempfile.TemporaryDirectory(prefix='prospective_manifest_') as directory:
        path = write_compatibility_manifest(selected, Path(directory) / 'manifest.csv')
        manifest = MaterialManifest.from_csv(path)
        audit = {
            'option_key': selected.option_key, 'candidate_id': selected.candidate_id,
            'candidate_row_sha256': digest(selected.row), 'a_native_background_row_sha256': BACKGROUND_SHA,
            'registry_sha256': selected.registry_sha256,
            'compatibility_manifest_sha256': sha256_file(path),
            'material_manifest_sha256': digest(manifest.as_dict()),
            'construction_source_sha256': sha256_file(__file__),
            'construction_source': str(Path(__file__).relative_to(ROOT)),
            'original_freeze_commit': '636d107524106989cb40feed7b8d42464c2fac9e',
            'rebonding': False, 'PT_substitution': False,
        }
    return manifest, audit
