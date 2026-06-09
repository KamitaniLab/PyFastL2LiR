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

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BENCH_SERVER="${BENCH_SERVER:-$(hostname)}"

LEGACY_CONDA="${LEGACY_CONDA:-.venv-legacy}"
conda_root="$(resolve_path "$LEGACY_CONDA")"
python_bin="${conda_root}/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: legacy Python is not executable: $python_bin" >&2
    echo "Set LEGACY_CONDA in speedtests/local.sh." >&2
    exit 1
fi

if [[ -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate base
else
    export PATH="${conda_root}/bin:${PATH}"
fi

"$python_bin" - <<'PY'
import sys

import fastl2lir  # noqa: F401
import numpy  # noqa: F401
import scipy  # noqa: F401

if sys.version_info[:2] != (3, 8):
    print(f"WARNING: expected Python 3.8, got {sys.version.split()[0]}")
PY

exec "$python_bin" -u speedtests/speedrun_legacy.py "$@"
