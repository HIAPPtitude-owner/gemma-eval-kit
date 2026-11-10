#!/bin/bash
# harness/diagnostic/repair.sh -- repairs applied after harness/shared/provision.sh for the
# `diagnostic` harness only. Every change to this file gets one line in CHANGELOG.md naming
# this file's new sha256 (local_eval refuses a sweep whose repair.sh sha is not logged).
# Its sha256 is folded into the diagnostic env_hash. Never edits the harness venv
# (/root/venv-official) or any installed swegemma / adk_submission / adk_eval_core file.
#
# v1: build the scorer's image layer once; docker_semantics.py (next to this file, sha256
# pinned below, so this file's sha covers it) gives every sandbox venv a hard-linked copy.
#  - Test wheels: the highest cp313-compatible version of each distribution among the 124
#    competition wheels. The scorer does the same: _deduplicate_wheels keeps the highest
#    version per distribution (swegemma/harness/container_setup.py:107-135, :170) and
#    sandbox/setup.py's slow path does too (setup.py:91-111). tasks.jsonl names no per-task
#    dependency (its keys: base_commit created_at hints_text instance_id patch
#    problem_statement repo test_patch), so there is no per-task pin to honour.
#  - imp.py and telnetlib.py, copied into the image's site-packages (Dockerfile.sandbox:10-11).
#  - The v0 switch (factory_diagnostic.pth in the base interpreter's site-packages) is
#    removed: v1 sandbox venvs are built with system_site_packages=False and never read it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMP=/mnt/<data-root>/competition
WHEELS="$COMP/wheels"
EXPECTED_WHEELS=124
PY=/root/venv-official/bin/python
UV=/root/.local/bin/uv
TARGET=/root/factory-diagnostic/site-packages
PTH=factory_diagnostic.pth
WRAPPER_SHA256=ec2437b4b2c4888ed1d490e7e46a5cb8aaa6a987fa9252b006988b5750ec9c0e

echo "$WRAPPER_SHA256  $HERE/docker_semantics.py" | sha256sum --check --status || {
  echo "REPAIR-FAIL: $HERE/docker_semantics.py sha256 != $WRAPPER_SHA256" >&2
  exit 1
}

n=$(find "$WHEELS" -maxdepth 1 -name '*.whl' | wc -l)
if [ "$n" -ne "$EXPECTED_WHEELS" ]; then
  echo "REPAIR-FAIL: $n wheels in $WHEELS, expected $EXPECTED_WHEELS" >&2
  exit 1
fi

# One requirement per distribution; uv picks the highest cp313-compatible version of each
# (several distributions ship more than one version, e.g. 56 starlette wheels).
mapfile -t NAMES < <(find "$WHEELS" -maxdepth 1 -name '*.whl' -printf '%f\n' \
  | cut -d- -f1 | tr '[:upper:]_' '[:lower:]-' | sort -u)
echo "repair v1: ${#NAMES[@]} distributions from $n wheels"

rm -rf "$TARGET"
"$UV" pip install --python "$PY" --target "$TARGET" --no-index --no-deps \
  --find-links "$WHEELS" "${NAMES[@]}"
cp "$COMP/docker/imp.py" "$COMP/docker/telnetlib.py" "$TARGET/"

BASE_SITE=$("$PY" -c 'import os, sys; v = "python%d.%d" % sys.version_info[:2]; print(os.path.join(sys.base_prefix, "lib", v, "site-packages"))')
rm -f "$BASE_SITE/$PTH"
echo "repair v1: image layer at $TARGET; removed $BASE_SITE/$PTH (v0 switch)"
