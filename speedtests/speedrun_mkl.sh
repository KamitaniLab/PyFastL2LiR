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

# Use the requested MKL-oriented environment while importing this working
# tree's src package. Override BENCH_RUNNERS or MKL_BENCH_RUNNERS to change the
# solver subset for a specific local environment.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BENCH_SERVER="${BENCH_SERVER:-$(hostname)_mkl}"
MKL_BENCH_RUNNERS="${MKL_BENCH_RUNNERS:-numpy,numba}"
export BENCH_RUNNERS="${BENCH_RUNNERS:-$MKL_BENCH_RUNNERS}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

MKL_CONDA="${MKL_CONDA:-.venv-mkl}"
mkl_root="$(resolve_path "$MKL_CONDA")"
python_bin="${mkl_root}/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: MKL Python is not executable: $python_bin" >&2
    echo "Set MKL_CONDA in speedtests/local.sh." >&2
    exit 1
fi

"$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")

missing = []
for module_name in ("numpy", "scipy", "threadpoolctl"):
    try:
        __import__(module_name)
    except ImportError:
        missing.append(module_name)

if missing:
    raise SystemExit(
        "Missing MKL environment dependencies: "
        + ", ".join(missing)
        + "\nInstall them in the MKL environment, then rerun speedtests/speedrun_mkl.sh."
    )

import numpy
import scipy
import threadpoolctl

import fastl2lir

print("MKL speed-test environment")
print(f"  python: {sys.version.split()[0]}")
print(f"  numpy: {numpy.__version__}")
print(f"  scipy: {scipy.__version__}")
print(f"  fastl2lir file: {fastl2lir.__file__}")
print(f"  threadpoolctl: {threadpoolctl.__version__}")
print(f"  threadpool info: {threadpoolctl.threadpool_info()}")
PY

exec "$python_bin" -u speedtests/speedrun.py "$@"
