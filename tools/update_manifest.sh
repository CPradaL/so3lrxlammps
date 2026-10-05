#!/bin/bash
# Regenerate MANIFEST.sha256 over the distributed files: everything in the
# release tree except what .gitignore lists (development material, builds).
# validate_release.sh checks the manifest. Needs git (only to apply the
# .gitignore rules; the tree does not have to be a repository).
set -Eeuo pipefail
RELEASE_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/so3lr-manifest.XXXXXX")
trap 'rm -rf "${TMP}"' EXIT
git init -q --bare "${TMP}/git"
cd "${RELEASE_ROOT}"
git --git-dir="${TMP}/git" --work-tree=. ls-files --others --exclude-standard \
  | grep -vx 'MANIFEST.sha256' | LC_ALL=C sort | sed 's|^|./|' \
  | xargs -d '\n' sha256sum > MANIFEST.sha256
echo "MANIFEST.sha256: $(wc -l < MANIFEST.sha256) files"
