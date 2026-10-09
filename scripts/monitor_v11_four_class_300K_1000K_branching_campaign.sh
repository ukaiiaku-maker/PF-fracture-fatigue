#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-$ROOT/runs/v11_four_class_300K_1000K_theta30_seed3621_1000um}
python3 - "$CAMPAIGN_ROOT" <<'PY'
import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); data=json.loads((root/'campaign_status.json').read_text())
print(f"CASE             STATUS        EXT_UM BRANCHES TIPS STEP  RESTARTS")
for r in data['cases']:
 print(f"{r['case_id']:<16} {r['status']:<12} {r['latest_extension_um']:>6.1f} {r['branch_birth_count']:>8} {r['active_tip_count']:>4} {r['accepted_steps']:>5} {r['restart_count']:>9}")
print(f"campaign_root: {root}")
print(f"git_head: {json.loads((root/'campaign_manifest.json').read_text())['git_head']}")
print(f"running_pids: {[r['pid'] for r in data['cases'] if r.get('pid')]}")
print(f"disk_usage: {shutil.disk_usage(root).used} bytes used on volume")
PY
