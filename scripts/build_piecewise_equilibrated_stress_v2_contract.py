#!/usr/bin/env python3
"""Write the prospective piecewise-equilibrated verifier contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.piecewise_equilibrated_boundary_stress_v2_contract import (
    contract_record,
)


OUTPUT = ROOT / "artifacts/v7_piecewise_equilibrated_stress_v2/v7_contract.json"


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(contract_record(), indent=2, sort_keys=True) + "\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
