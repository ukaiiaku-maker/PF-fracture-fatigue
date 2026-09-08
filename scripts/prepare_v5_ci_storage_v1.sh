#!/usr/bin/env bash
# Disposable hosted-runner storage only. Never run cleanup in a user workspace.
set -euo pipefail
if [[ "${GITHUB_ACTIONS:-}" != true || "${RUNNER_ENVIRONMENT:-}" != github-hosted || "${RUNNER_OS:-}" != Linux ]]; then
  printf '%s\n' 'Refusing SDK cleanup outside a disposable GitHub-hosted Linux runner.' >&2
  exit 2
fi
case "${GITHUB_WORKSPACE:-}" in /home/runner/work/*) ;; *) exit 2 ;; esac
minimum_available_kib=$((24 * 1024 * 1024))
available_kib=$(df -Pk / | awk 'NR==2 {print $4}')
df -h /
if (( available_kib < minimum_available_kib )); then
  # These are fixed preinstalled SDK directories, not repository or Python data.
  # Validate every existing target before performing any removal.
  for task_sdk_path in /usr/local/lib/android /usr/share/dotnet /opt/ghc; do
    if [[ -e "$task_sdk_path" || -L "$task_sdk_path" ]]; then
      [[ -d "$task_sdk_path" && ! -L "$task_sdk_path" && "$(realpath -- "$task_sdk_path")" == "$task_sdk_path" ]] || exit 3
      du -sh -- "$task_sdk_path"
    fi
  done
  sudo rm -rf -- /usr/local/lib/android /usr/share/dotnet /opt/ghc
  printf '%s\n' 'Removed only unused Android/.NET/GHC SDKs on this disposable runner; a new runner restores them.'
fi
available_kib=$(df -Pk / | awk 'NR==2 {print $4}')
df -h /
printf 'Available KiB: %s; required reserve KiB: %s\n' "$available_kib" "$minimum_available_kib"
(( available_kib >= minimum_available_kib ))
