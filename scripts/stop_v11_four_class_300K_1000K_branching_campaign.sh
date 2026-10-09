#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-$ROOT/runs/v11_four_class_300K_1000K_theta30_seed3621_1000um}
python3 - "$CAMPAIGN_ROOT" <<'PY'
import json,os,pathlib,signal,sys
root=pathlib.Path(sys.argv[1]); data=json.loads((root/'campaign_status.json').read_text())
for row in data['cases']:
 if row.get('pid'):
  try: os.kill(int(row['pid']), signal.SIGTERM); print(f"stopped {row['case_id']} pid={row['pid']}")
  except ProcessLookupError: pass
PY
