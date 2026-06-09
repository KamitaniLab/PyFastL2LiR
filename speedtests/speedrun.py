"""Current-environment speed test for PyFastL2LiR implementations.

This script compares L2-regularized linear regression fits on synthetic
fMRI-like matrices in the active project Python environment. The mathematical
objects are a Gaussian input matrix X, Gaussian target arrays Y with dense and
convolutional feature shapes, ridge weights W, bias b, and predictions on a
small validation slice. The execution stages are environment reporting,
synthetic data generation, repeated fitting for the package numpy solver, the
package numba solver, prediction-difference checks, median runtime summaries,
and CSV writing. The saved output is speedtests/speedtest_<server>.csv.

Run directly with:
    speedtests/speedrun.sh
"""

import os
import socket
from pathlib import Path

import fastl2lir

from speedtest_common import run_speedtest

server_name = os.environ.get("BENCH_SERVER") or socket.gethostname()

run_speedtest(
    fastl2lir_module=fastl2lir,
    runners=[
        {"label": "numpy", "fit_kwargs": {"solver": "numpy"}},
        {"label": "numba", "fit_kwargs": {"solver": "numba"}},
    ],
    output_csv=Path("speedtests") / f"speedtest_{server_name}.csv",
    comparison="current FastL2LiR.fit numpy and numba solver comparison",
    module_names=("numba",),
)
