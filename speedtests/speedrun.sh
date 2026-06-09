#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
local_config="${script_dir}/local.sh"

if [[ -f "$local_config" ]]; then
    # shellcheck source=/dev/null
    source "$local_config"
fi

resolve_path() {
    local path_value="$1"

    if [[ "$path_value" = /* ]]; then
        printf "%s\n" "$path_value"
    else
        printf "%s/%s\n" "$repo_root" "$path_value"
    fi
}

cd "$repo_root"

# Use the project Python environment only. Thread handling is intentionally
# left to fastl2lir and the runtime environment under test.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BENCH_SERVER="${BENCH_SERVER:-$(hostname)}"

VENV_PATH="${VENV_PATH:-.venv}"
python_bin="$(resolve_path "$VENV_PATH")/bin/python"

if [[ ! -x "$python_bin" ]]; then
    if [[ "$VENV_PATH" != ".venv" ]]; then
        echo "ERROR: Python is not executable: $python_bin" >&2
        echo "Create that environment or update VENV_PATH in speedtests/local.sh." >&2
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: .venv is missing and uv is not available." >&2
        echo "Install uv or set VENV_PATH in speedtests/local.sh." >&2
        exit 1
    fi
    uv sync --extra numba
    python_bin="$(resolve_path "$VENV_PATH")/bin/python"
fi

if ! "$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")

import fastl2lir  # noqa: F401
import numba  # noqa: F401
import numpy  # noqa: F401
import scipy  # noqa: F401
PY
then
    if [[ "$VENV_PATH" != ".venv" ]]; then
        echo "ERROR: Python environment is incomplete: $python_bin" >&2
        echo "Install fastl2lir, numpy, scipy, and numba there, or update VENV_PATH." >&2
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: Python environment is incomplete and uv is not available." >&2
        echo "Set VENV_PATH in speedtests/local.sh to a complete environment." >&2
        exit 1
    fi
    uv sync --extra numba
fi

exec "$python_bin" -u speedtests/speedrun.py "$@"
