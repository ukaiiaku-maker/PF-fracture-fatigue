#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import arrhenius_fracture

PACKAGE_ROOT = Path(arrhenius_fracture.__file__).resolve().parent
EXPECTED_ROOT = REPOSITORY_ROOT / "arrhenius_fracture"
if getattr(arrhenius_fracture, "PROJECT_ID", None) != "PF-fracture-fatigue":
    raise RuntimeError(
        "wrong arrhenius_fracture project imported: "
        f"{getattr(arrhenius_fracture, 'PROJECT_ID', None)!r} from {PACKAGE_ROOT}"
    )
if PACKAGE_ROOT != EXPECTED_ROOT:
    raise RuntimeError(
        f"stale arrhenius_fracture import: expected {EXPECTED_ROOT}, got {PACKAGE_ROOT}"
    )

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    MODEL_ID as LOAD_ID,
)
from arrhenius_fracture.interaction_integral_v10214 import (
    MODEL_ID as INTERACTION_ID,
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
                "project_id": arrhenius_fracture.PROJECT_ID,
                "project_repository": arrhenius_fracture.PROJECT_REPOSITORY,
                "project_release": arrhenius_fracture.PROJECT_RELEASE,
                "protected_public_api_version": arrhenius_fracture.__version__,
                "package_root": str(PACKAGE_ROOT),
                "repository_local_import_verified": True,
                "response_model": RESPONSE_ID,
                "interaction_integral_model": INTERACTION_ID,
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
