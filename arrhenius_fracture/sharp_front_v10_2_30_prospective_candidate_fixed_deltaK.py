"""Opt-in frozen prospective row selection for the unchanged physical driver."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from . import sharp_front_v10_2_27 as paper
from . import sharp_front_v10_2_30_fixed_deltaK as fixed
from .prospective_paris_candidate_engine_v10230 import (
    P40, REGISTRY, SELECTION, build_frozen_manifest, select_frozen_option,
)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('V10230_RESTART_CHECKPOINT_DIR') or any('resume' in a or 'restart' in a for a in args):
        raise ValueError('prospective trajectories must never resume')
    option = paper._option_value(args, '--parameter-option', os.environ.get('PARAMETER_OPTION'))
    manifest, audit = build_frozen_manifest(option)
    out_value = paper._option_value(args, '--out')
    if not out_value:
        raise ValueError('prospective entry requires --out claimed by the physical launcher')
    out = Path(out_value)
    run_manifest_path = out / 'high_cycle_run_manifest.json'
    if not run_manifest_path.is_file():
        raise ValueError('physical launcher must claim virgin output and write launch manifest first')
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest['prospective_candidate'] = audit
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + '\n')
    originals = (paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS, paper._SOURCE_SELECT_OPTION)
    paper.DEFAULT_REGISTRY = REGISTRY
    paper.SELECTION_RECORD = SELECTION
    paper.VALID_OPTIONS = {'A_NATIVE': 'A_NATIVE', P40: P40}
    paper._SOURCE_SELECT_OPTION = select_frozen_option
    try:
        return fixed.main(args)
    finally:
        (paper.DEFAULT_REGISTRY, paper.SELECTION_RECORD, paper.VALID_OPTIONS, paper._SOURCE_SELECT_OPTION) = originals


if __name__ == '__main__':
    main()
