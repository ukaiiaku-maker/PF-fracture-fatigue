#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    MODEL_ID as LOAD_ID,
)
from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    MODEL_ID as RESPONSE_ID,
)
from arrhenius_fracture.state_resolved_signed_engine_v10214 import (
    MODEL_ID as ENGINE_ID,
)


def main() -> None:
    print(
        json.dumps(
            {
                "response_model": RESPONSE_ID,
                "load_invariance_model": LOAD_ID,
                "engine_model": ENGINE_ID,
                "active_kernel_mechanically_measured": True,
                "wake_kernel_mechanically_measured": False,
                "wake_shielding_supported": False,
                "production_parameterization_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
