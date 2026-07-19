#!/usr/bin/env python3
"""Run v10.2.8 staged scoring with the v10.2.9 shared constitutive core."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from arrhenius_fracture.signed_kernel_family_v1029 import (
    StateResolvedSignedShieldingKernelFamily,
)
from arrhenius_fracture.state_resolved_reduced_campaign_v1029 import (
    StateResolvedProductionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "run_v10_2_8_staged_parameterization.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("_v1028_staged_runner", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load staged runner {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    base = _load_base()
    base.MODEL_ID = "v10.2.9_staged_parameterization_effective_opening_core"
    base.StateResolvedSignedShieldingKernelFamily = (
        StateResolvedSignedShieldingKernelFamily
    )
    base.StateResolvedProductionConfig = StateResolvedProductionConfig
    return base.main()


if __name__ == "__main__":
    main()
