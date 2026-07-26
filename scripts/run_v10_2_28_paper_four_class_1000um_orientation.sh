#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh" "$@"
