"""Install the v10.2.30 active-state VHCF block patch for campaign subprocesses."""
from __future__ import annotations

import os


if os.environ.get("V10230_ACTIVE_STATE_BLOCK_CONTROL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from arrhenius_fracture.active_state_block_control_v10230 import (
        install_active_state_block_control,
    )

    install_active_state_block_control()
