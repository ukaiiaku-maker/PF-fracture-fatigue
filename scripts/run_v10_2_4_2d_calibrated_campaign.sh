#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
STOPPED: the v10.2.4 reduced parameter campaign is physically invalid and must
not be resumed.

The cap-free A0002333 calculation exposed two missing closures:
  1. source-site activations were transferred directly into coherent unsigned
     dislocation-line content;
  2. shielding used the legacy analytic +1/+1 channel projection rather than a
     signed 2-D unit-response operator.

Use v10.2.5 only after generating and validating:
  - a 2-D signed unit-dislocation/slip shielding kernel;
  - a mechanically derived activation-to-line-content normalization;
  - a physical source-capacity range;
  - an exact signed 2-D/replay equivalence trace.

No K-shield cap or fitted attenuation factor is an acceptable substitute.
EOF
exit 64
