"""Opt-in production entry for the once-frozen transfer candidate cohort."""
import json
import os
from pathlib import Path
import sys
from . import sharp_front_v10_2_27 as paper
from . import sharp_front_v10_2_30_fixed_deltaK as fixed
from .prospective_paris_transfer_engine_v10230 import (
    REGISTRY,SELECTION,build_transfer_manifest,select_transfer_option,
)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if os.environ.get('V10230_RESTART_CHECKPOINT_DIR') or any('resume' in a or 'restart' in a for a in args):
        raise ValueError('prospective trajectories must never resume')
    option=paper._option_value(args,'--parameter-option',os.environ.get('PARAMETER_OPTION'))
    _,audit=build_transfer_manifest(option)
    out=paper._option_value(args,'--out')
    path=Path(out or '')/'high_cycle_run_manifest.json'
    if not out or not path.is_file():
        raise ValueError('physical launcher must claim virgin output first')
    payload=json.loads(path.read_text());payload['prospective_candidate']=audit
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    originals=(paper.DEFAULT_REGISTRY,paper.SELECTION_RECORD,paper.VALID_OPTIONS,paper._SOURCE_SELECT_OPTION)
    paper.DEFAULT_REGISTRY=REGISTRY;paper.SELECTION_RECORD=SELECTION
    order=json.loads(SELECTION.read_text())['canonical_option_order']
    paper.VALID_OPTIONS={key:key for key in order};paper._SOURCE_SELECT_OPTION=select_transfer_option
    try:
        return fixed.main(args)
    finally:
        (paper.DEFAULT_REGISTRY,paper.SELECTION_RECORD,paper.VALID_OPTIONS,paper._SOURCE_SELECT_OPTION)=originals


if __name__=='__main__':main()
