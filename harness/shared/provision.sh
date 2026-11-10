#!/bin/bash
# harness/shared/provision.sh -- provision the WSL1 distro `gemma-wsl1` as a mirror of
# competition/docker/Dockerfile.sandbox. Used unchanged by BOTH harnesses (as_shipped and
# diagnostic). Provisioning only: no eval and no run-mode logic. Its sha256 is pinned in
# pins.json (harness.provision_sh_sha256) and asserted before every sweep.
#
# Assembled from the provisioning parts of the T1 probe scripts; each step cites its source:
#   P4 = <data-root>/wsl/probe/p4_provision313_empty.sh
#   P5 = <data-root>/wsl/probe/p5_mirror_empty_gold.sh
#   DS = <data-root>/competition/docker/Dockerfile.sandbox
#
# Run as root inside the distro:  bash /mnt/c/.../harness/shared/provision.sh
# Afterwards `uv pip freeze --python /root/venv-official/bin/python | sha256sum` must equal
# pins.json wsl1.freeze_sha256 (the sweep asserts this before every run).
set -u
export PATH="/root/.local/bin:$PATH"                                   # P4:5, P5:5
C=/mnt/<data-root>/competition                                    # P4:6, P5:6
PY=/root/venv-official/bin/python                                     # P5:8

echo "== apt (DS:4-6)"
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
  git patch protobuf-compiler pigz                                    # P4:10
echo "apt_exit=$?"                                                    # P4:11

echo "== python 3.13 (DS:1 python:3.13-slim)"
uv python install 3.13                                                # P4:14
echo "pyinstall_exit=$?"                                              # P4:15
rm -rf /root/venv-official                                            # P4:16
uv venv /root/venv-official --python 3.13                             # P4:17
"$PY" --version                                                       # P4:18
uv pip install --python "$PY" \
  --find-links /mnt/<data-root>/wheelhouse swegemma adk-submission  # P4:19-20
echo "install_exit=$?"                                                # P4:21

echo "== DS:7 pip line"
uv pip install --python "$PY" pytest pytest-timeout==2.1.0 typer pdm-backend setuptools \
  wheel poetry-core hatchling flit-core editables                     # P5:11-12
echo "mirror_pip_exit=$?"                                             # P5:13

echo "== DS:10-11 shims"
SP=$("$PY" -c 'import site; print(site.getsitepackages()[0])')        # P5:14
cp "$C/docker/imp.py" "$C/docker/telnetlib.py" "$SP/"                 # P5:15
echo "shims_exit=$?"                                                  # P5:16

echo "== DS:14-16 git config"
git config --global user.email "agent@eval"                           # P5:17
git config --global user.name "Agent"                                 # P5:18
git config --global init.defaultBranch main                           # P5:19

# DS:18 ENV TEST_TMPDIR=/tmp (P5:20). An export here does not persist into later shells,
# so src/factory/local_eval.py also sets TEST_TMPDIR=/tmp on every eval invocation.
export TEST_TMPDIR=/tmp                                               # P5:20

echo "== freeze (P5:21 wrote it to the probe dir; here only its sha256 is printed)"
uv pip freeze --python "$PY" | sha256sum
