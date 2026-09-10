"""Direct construction for prospectively frozen transfer candidates only."""
from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile

from .material_manifest import MaterialManifest
from .parameter_registry_v9111 import sha256_file, write_compatibility_manifest
from .prospective_paris_candidate_engine_v10230 import (
    ARTIFACTS, BACKGROUND_SHA, CLEAVAGE_FIELDS, IDENTITY_FIELDS, digest, select_frozen_option,
)

FREEZE = ARTIFACTS / 'transfer_candidate_freeze_v1.json'
REGISTRY = ARTIFACTS / 'transfer_candidate_registry_v1.csv'
SELECTION = ARTIFACTS / 'transfer_candidate_selection_v1.json'
ALLOWED = {'P40_TRANSFER_CALIBRATED_GEN2', 'P25_TRANSFER_V1_RANK1', 'P25_TRANSFER_V1_RANK2',
           'P55_TRANSFER_V1_RANK1', 'P55_TRANSFER_V1_RANK2'}


def select_transfer_option(option_key, registry_path=REGISTRY, **kwargs):
    if Path(registry_path).resolve() != REGISTRY or option_key not in ALLOWED:
        raise ValueError('not a scoped prospective transfer option')
    freeze = json.loads(FREEZE.read_text())
    for p,key in ((REGISTRY,'registry_sha256'),(SELECTION,'selection_sha256'),
                  (ARTIFACTS/'p40_transfer_update.json','transfer_sha256')):
        if sha256_file(p) != freeze[key]:
            raise ValueError('prospective transfer freeze hash mismatch')
    if freeze['transfer_update_number'] != 1:
        raise ValueError('only one transfer update is allowed')
    eligible = {r['candidate_id']:r for r in freeze['candidates']}
    if option_key not in eligible or not eligible[option_key]['eligible']:
        raise ValueError('candidate was not prospectively eligible')
    with REGISTRY.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(eligible) or len({r['option_key'] for r in rows}) != len(rows):
        raise ValueError('duplicate or missing frozen candidate')
    native = select_frozen_option('A_NATIVE')
    if digest(native.row) != BACKGROUND_SHA:
        raise ValueError('background changed')
    for row in rows:
        key = row['option_key']
        if key not in eligible or row['candidate_id'] != key or digest(row) != eligible[key]['complete_row_sha256']:
            raise ValueError('candidate row hash or identity mismatch')
        changed = {k for k in set(row)|set(native.row) if row.get(k) != native.row.get(k)}
        if not CLEAVAGE_FIELDS <= changed or not changed <= CLEAVAGE_FIELDS|IDENTITY_FIELDS:
            raise ValueError('non-cleavage background changed')
    row = next(r for r in rows if r['option_key'] == option_key)
    return replace(native,option_key=option_key,candidate_id=option_key,row=row,
                   role=row['role'],mechanism_summary=row['mechanism_summary'],validation_status=row['validation_status'],
                   registry_path=str(REGISTRY),registry_sha256=sha256_file(REGISTRY))


def build_transfer_manifest(option_key):
    selected=select_transfer_option(option_key)
    with tempfile.TemporaryDirectory(prefix='prospective_transfer_manifest_') as directory:
        path=write_compatibility_manifest(selected,Path(directory)/'manifest.csv')
        manifest=MaterialManifest.from_csv(path)
        audit=dict(option_key=option_key,candidate_id=option_key,candidate_row_sha256=digest(selected.row),
            material_manifest_sha256=digest(manifest.as_dict()),compatibility_manifest_sha256=sha256_file(path),
            a_native_background_row_sha256=BACKGROUND_SHA,registry_sha256=sha256_file(REGISTRY),
            freeze_sha256=sha256_file(FREEZE),transfer_sha256=sha256_file(ARTIFACTS/'p40_transfer_update.json'),
            construction_source_sha256=sha256_file(__file__),rebonding=False,PT_substitution=False)
    return manifest,audit
