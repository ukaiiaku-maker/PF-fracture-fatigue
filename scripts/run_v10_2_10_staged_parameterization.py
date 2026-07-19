#!/usr/bin/env python3
"""Run v10.2.9 staged physics with corrected v10.2.10 promotion gates."""
from __future__ import annotations

from arrhenius_fracture.quality_diversity_v10210 import (
    MODEL_ID,
    QualityDiversityConfig,
    select_quality_diverse,
)
from scripts import run_v10_2_9_staged_parameterization as base


def main() -> None:
    # The v10.2.9 wrapper owns analytical-trajectory flattening and audit writing.
    # Replace only its selector symbols before it installs the stage hooks.
    base.SELECTION_MODEL_ID = MODEL_ID
    base.QualityDiversityConfig = QualityDiversityConfig
    base.select_quality_diverse = select_quality_diverse
    base.main()


if __name__ == "__main__":
    main()
