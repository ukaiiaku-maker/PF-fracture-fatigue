"""Install v10.2.30 VHCF runtime block-control patches in subprocesses."""
from __future__ import annotations

import os


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if _enabled("V10230_ACTIVE_STATE_BLOCK_CONTROL"):
    from arrhenius_fracture.active_state_block_control_v10230 import (
        install_active_state_block_control,
    )

    install_active_state_block_control()

if _enabled("V10230_FEEDBACK_STATE_BLOCK_CONTROL"):
    from arrhenius_fracture.feedback_state_block_control_v10230 import (
        install_feedback_state_block_control,
    )

    install_feedback_state_block_control()
