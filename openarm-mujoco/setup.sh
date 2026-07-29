#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
uv sync --frozen
uv run python verify_setup.py
