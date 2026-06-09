"""Legacy-environment speed test for PyFastL2LiR.

This script measures L2-regularized linear regression runtime for the legacy
FastL2LiR implementation in the configured legacy Python environment. The
mathematical objects are synthetic Gaussian input matrices X, synthetic target
arrays Y, ridge weights W, bias b, and validation predictions used to check
repeat consistency. The execution stages are environment reporting, benchmark
case construction, repeated legacy fitting, median runtime summarization, and
CSV writing. The saved output is speedtests/speedtest_legacy_<server>.csv.

Run directly with:
    speedtests/speedrun_legacy.sh
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
        {"label": "legacy", "fit_kwargs": {}},
    ],
    output_csv=Path("speedtests") / f"speedtest_legacy_{server_name}.csv",
    comparison="legacy FastL2LiR.fit runtime",
    module_names=("scipy",),
)
