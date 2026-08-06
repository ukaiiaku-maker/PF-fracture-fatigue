#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

BRANCH=${BRANCH:-v10.4.3-plastic-dominance-censor}
REMOTE=${REMOTE:-origin}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}

command -v git >/dev/null 2>&1 || {
  echo "ERROR: git is unavailable" >&2
  exit 2
}
command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is unavailable" >&2
  exit 2
}

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty; installation will not overwrite local work" >&2
  git status --short >&2
  exit 2
fi

echo "Fetching $REMOTE/$BRANCH"
git fetch --prune "$REMOTE" "$BRANCH"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
else
  git switch --track -c "$BRANCH" "$REMOTE/$BRANCH"
fi

git merge --ff-only "$REMOTE/$BRANCH"

HEAD=$(git rev-parse HEAD)
printf 'Installing branch: %s\n' "$BRANCH"
printf 'Pinned HEAD:       %s\n' "$HEAD"

conda run --no-capture-output -n "$CONDA_ENV" \
  python -m pip install -e '.[test]'

CONDA_ENV="$CONDA_ENV" EXPECTED_BRANCH="$BRANCH" \
  bash scripts/validate_v10_4_3_plastic_dominance.sh

cat > .v10_4_3_install.json <<EOF
{
  "branch": "$BRANCH",
  "commit": "$HEAD",
  "conda_environment": "$CONDA_ENV",
  "validation": "passed"
}
EOF

echo "INSTALLATION PASSED at $HEAD"
echo "Next: run the one-case inherited reuse smoke before any live matrix launch."
