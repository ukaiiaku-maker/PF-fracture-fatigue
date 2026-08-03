#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
ERROR: this launcher is intentionally disabled.

The independent v10.2.30 audit found that removing all state targets can pass the
full VHCF horizon to an unbounded depth-first recursive committer. That can cause
multi-hour no-output hangs and is not an acceptable production algorithm.

Do not restart the four-class fatigue campaign with the feedback-state runtime.
The next production launcher must use the P0 bounded forward marcher, a
kernel-weighted shielding/state error measure, explicit event-function bracketing,
and the validated stationary-tail propagator.
EOF

exit 2
