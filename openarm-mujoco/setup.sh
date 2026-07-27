#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASSET_DIR="${ROOT_DIR}/.assets/openarm_mujoco"
OPENARM_REPOSITORY="https://github.com/enactic/openarm_mujoco.git"
OPENARM_COMMIT="8955afb54e4adfb59a236e2b4d15192b7a02865c"

mkdir -p "${ROOT_DIR}/.assets"

if [[ ! -d "${ASSET_DIR}/.git" ]]; then
  git clone "${OPENARM_REPOSITORY}" "${ASSET_DIR}"
fi

git -C "${ASSET_DIR}" fetch origin "${OPENARM_COMMIT}"
git -C "${ASSET_DIR}" checkout --detach "${OPENARM_COMMIT}"

cd "${ROOT_DIR}"
uv sync --frozen
uv run python verify_setup.py

