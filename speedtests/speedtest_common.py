"""Shared speed-test routines for PyFastL2LiR solver experiments.

This module benchmarks L2-regularized linear regression fits on synthetic
fMRI-like design matrices. The mathematical objects are an input matrix X
(samples by voxels), a target tensor Y (samples by model units), selected
feature counts for ridge regression, fitted weights W, fitted bias b, and
short-slice predictions used to check numerical agreement between runners.

Execution proceeds by reading benchmark dimensions from environment variables,
building named target-shape cases, generating fresh Gaussian X and Y arrays for
each repeat, fitting each configured runner in alternating order, comparing
predictions against the first runner, summarizing median runtimes, and writing a
CSV table. Saved outputs are speed-test CSV files under speedtests/ whose columns
include the server name, case dimensions, per-runner median fit times,
speedups relative to the baseline runner, and maximum prediction differences.
"""

import importlib
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np


RNG = np.random.default_rng()


def read_positive_int(name, default):
    """Read a positive integer setting from the environment.

    Inputs:
        name: Environment variable name.
        default: Integer value used when the variable is unset.
    Output:
        The parsed positive integer.
    This function validates that benchmark dimensions and repeat counts are
    positive so the experiment fails before allocating arrays.
    """
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer") from exc

    if parsed < 1:
        raise SystemExit(f"{name} must be a positive integer")

    return parsed


def get_benchmark_settings():
    """Collect benchmark dimensions and repeat counts.

    Inputs:
        Environment variables BENCH_N, BENCH_P, BENCH_K, and BENCH_REPEATS.
    Output:
        Tuple of sample count n, voxel count p, selected feature count k, and
        repeat count.
    This function centralizes the size controls used by all speed-test cases.
    """
    n = read_positive_int("BENCH_N", 1000)
    p = read_positive_int("BENCH_P", 15000)
    k = read_positive_int("BENCH_K", 500)
    repeats = read_positive_int("BENCH_REPEATS", 3)
    return n, p, k, repeats


def make_cases(k, repeats):
    """Build the target-shape cases for the speed test.

    Inputs:
        k: Number of selected input features used by ridge fitting.
        repeats: Number of repeats for each case.
        Optional BENCH_CASES environment variable with comma-separated case
        names.
    Output:
        List of tuples containing case name, Y shape after the sample axis,
        fit parameters, and repeat count.
    This function defines dense and chunked target layouts that mimic neural
    network feature layers.
    """
    cases = [
        ("fc8", (1000,), dict(alpha=100.0, n_feat=k), repeats),
        ("fc6", (4096,), dict(alpha=100.0, n_feat=k), repeats),
        (
            "conv5_chunk5",
            (5, 14, 14),
            dict(alpha=100.0, n_feat=k, chunk_size=196),
            repeats,
        ),
        (
            "conv4_chunk5",
            (5, 28, 28),
            dict(alpha=100.0, n_feat=k, chunk_size=784),
            repeats,
        ),
    ]

    case_filter = os.environ.get("BENCH_CASES")
    if not case_filter:
        return cases

    selected_cases = {name.strip() for name in case_filter.split(",") if name.strip()}
    known_cases = {name for name, *_ in cases}
    unknown_cases = selected_cases - known_cases
    if unknown_cases:
        raise SystemExit(
            "Unknown BENCH_CASES entries: " + ", ".join(sorted(unknown_cases))
        )

    return [case for case in cases if case[0] in selected_cases]


def filter_runners(runners):
    """Select benchmark runners from BENCH_RUNNERS when requested.

    Inputs:
        runners: Ordered list of available runner dictionaries.
        Optional BENCH_RUNNERS environment variable with comma-separated runner
        labels.
    Output:
        Ordered list of selected runner dictionaries.
    This function lets a researcher run the same speed-test script in
    environments that lack optional solvers such as numba while keeping the
    experiment definition explicit in the terminal log and CSV columns.
    """
    runner_filter = os.environ.get("BENCH_RUNNERS")
    if not runner_filter:
        return runners

    selected_labels = {
        label.strip() for label in runner_filter.split(",") if label.strip()
    }
    known_labels = {runner["label"] for runner in runners}
    unknown_labels = selected_labels - known_labels
    if unknown_labels:
        raise SystemExit(
            "Unknown BENCH_RUNNERS entries: " + ", ".join(sorted(unknown_labels))
        )

    selected_runners = [
        runner for runner in runners if runner["label"] in selected_labels
    ]
    if not selected_runners:
        raise SystemExit("BENCH_RUNNERS did not select any runners")

    return selected_runners


def module_version(module_name):
    """Return a module version string for the environment report.

    Inputs:
        module_name: Importable Python module name.
    Output:
        Version string, '(unknown)', or '(not installed)'.
    This function records the numerical stack used for a speed-test run.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "(not installed)"
    return getattr(module, "__version__", "(unknown)")


def package_file(module):
    """Return the resolved file path for an imported module.

    Inputs:
        module: Imported Python module object.
    Output:
        Absolute path to the module file.
    This function helps distinguish installed package code from local source
    modules during benchmark reporting.
    """
    return str(Path(module.__file__).resolve())


def make_data(n, p, y_shape):
    """Generate synthetic regression data for one benchmark repeat.

    Inputs:
        n: Number of samples.
        p: Number of input features or voxels.
        y_shape: Target-unit shape after the sample axis.
    Output:
        Tuple (X, Y) of float64 Gaussian arrays.
    This function creates fresh comparable input and target arrays for all
    runners in a single repeat.
    """
    X = RNG.normal(size=(n, p)).astype(np.float64)
    Y = RNG.normal(size=(n,) + y_shape).astype(np.float64)
    return X, Y


def runner_module(fastl2lir_module, runner):
    """Resolve the FastL2LiR module used by one runner.

    Inputs:
        fastl2lir_module: Default module used when a runner has no override.
        runner: Runner dictionary.
    Output:
        Module object that provides FastL2LiR.
    This function lets a speed test compare runners that may use different
    modules while sharing the same benchmark data.
    """
    return runner.get("module", fastl2lir_module)


def fit_once(fastl2lir_module, X, Y, runner, params):
    """Fit one runner on one generated dataset.

    Inputs:
        fastl2lir_module: Default module that provides FastL2LiR.
        X: Input matrix for ridge fitting.
        Y: Target matrix or tensor for ridge fitting.
        runner: Runner dictionary with label, optional module, optional
        model_kwargs, and fit_kwargs.
        params: Case-level fit parameters such as alpha, n_feat, and chunk_size.
    Output:
        Tuple of elapsed seconds and fitted model.
    This function constructs the requested FastL2LiR implementation, merges
    case and runner fit arguments, runs fit, and prints the elapsed time.
    """
    label = runner["label"]
    module = runner_module(fastl2lir_module, runner)
    model_kwargs = dict(runner.get("model_kwargs", {}))
    fit_kwargs = dict(params)
    fit_kwargs.update(runner.get("fit_kwargs", {}))

    print(f"  start {label} fit", flush=True)
    start = time.perf_counter()
    model = module.FastL2LiR(**model_kwargs).fit(X, Y, **fit_kwargs)
    elapsed = time.perf_counter() - start
    print(f"  done {label} fit: {elapsed:.4f} s", flush=True)
    return elapsed, model


def ordered_runners(runners, repeat_index):
    """Return runner order for one repeat.

    Inputs:
        runners: Ordered list of runner dictionaries.
        repeat_index: Zero-based repeat index.
    Output:
        Runners in forward order for even repeats and reverse order for odd
        repeats.
    This function reduces systematic timing bias from always running one
    implementation first.
    """
    if repeat_index % 2 == 0:
        return runners
    return list(reversed(runners))


def measure_case(fastl2lir_module, n, p, y_shape, params, repeats, runners):
    """Measure all runners for one benchmark case.

    Inputs:
        fastl2lir_module: Default module that provides FastL2LiR.
        n: Number of samples.
        p: Number of input features or voxels.
        y_shape: Target-unit shape after the sample axis.
        params: Case-level fit parameters.
        repeats: Number of repeated fits.
        runners: Runner dictionaries to compare.
    Output:
        Tuple of per-label elapsed-time lists and maximum prediction difference
        versus the first runner.
    This function generates data, fits all runners, and checks prediction
    agreement on the first samples for every repeat.
    """
    times_by_label = {runner["label"]: [] for runner in runners}
    max_pred_diffs = []

    for repeat_index in range(repeats):
        print(f"  repeat {repeat_index + 1}/{repeats}")

        # Generate fresh data for every repeat. Within a repeat, all runners use
        # the same data so correctness and timing remain comparable.
        X, Y = make_data(n, p, y_shape)

        models = {}
        for runner in ordered_runners(runners, repeat_index):
            elapsed, model = fit_once(fastl2lir_module, X, Y, runner, params)
            times_by_label[runner["label"]].append(elapsed)
            models[runner["label"]] = model

        pred_slice = X[: min(32, X.shape[0])]
        baseline_label = runners[0]["label"]
        baseline_pred = models[baseline_label].predict(pred_slice)
        for runner in runners[1:]:
            pred = models[runner["label"]].predict(pred_slice)
            max_pred_diffs.append(float(np.max(np.abs(baseline_pred - pred))))

    max_pred_diff = max(max_pred_diffs) if max_pred_diffs else 0.0
    return times_by_label, max_pred_diff


def median_by_label(times_by_label):
    """Compute median runtime for each runner label.

    Inputs:
        times_by_label: Mapping from runner label to elapsed-time list.
    Output:
        Mapping from runner label to median elapsed seconds.
    This function provides the primary speed-test statistic for noisy repeated
    runs.
    """
    return {label: statistics.median(times) for label, times in times_by_label.items()}


def speedup_values(runners, medians):
    """Compute speedups relative to the first runner.

    Inputs:
        runners: Ordered list of runner dictionaries.
        medians: Mapping from runner label to median elapsed seconds.
    Output:
        Mapping from non-baseline runner label to baseline_time / runner_time.
    This function gives each comparison an explicit name when more than two
    implementations are benchmarked.
    """
    if len(runners) < 2:
        return {}
    baseline = runners[0]["label"]
    return {
        runner["label"]: medians[baseline] / medians[runner["label"]]
        for runner in runners[1:]
    }


def summary_header(runners):
    """Build the CSV header for the benchmark summary.

    Inputs:
        runners: Ordered list of runner dictionaries.
    Output:
        Comma-separated CSV header string.
    This function includes per-runner median time columns and named speedup
    columns relative to the first runner.
    """
    time_columns = ",".join(f"{runner['label']}_median_s" for runner in runners)
    columns = f"server,case,n,p,k,q,repeats,{time_columns}"
    if len(runners) >= 2:
        baseline = runners[0]["label"]
        speedup_columns = ",".join(
            f"{runner['label']}_speedup_vs_{baseline}" for runner in runners[1:]
        )
        columns += f",{speedup_columns}"
    columns += ",max_pred_diff"
    return columns


def summary_line(server_name, row, runners):
    """Build one CSV data line for a benchmark case.

    Inputs:
        server_name: Host or user-specified benchmark server label.
        row: Case summary dictionary.
        runners: Ordered list of runner dictionaries.
    Output:
        Comma-separated CSV row string.
    This function serializes dimensions, medians, speedups, and prediction
    agreement into the saved summary table.
    """
    base = [
        server_name,
        row["case"],
        str(row["n"]),
        str(row["p"]),
        str(row["k"]),
        str(row["q"]),
        str(row["repeats"]),
    ]
    times = [f"{row['medians'][runner['label']]:.6f}" for runner in runners]
    values = base + times
    for runner in runners[1:]:
        values.append(f"{row['speedups'][runner['label']]:.2f}")
    values.append(f"{row['max_pred_diff']:.3e}")
    return ",".join(values)


def print_environment(
    fastl2lir_module, module_names, n, p, k, repeats, cases, server_name
):
    """Print Python, package, and benchmark-size metadata.

    Inputs:
        fastl2lir_module: Default FastL2LiR package module.
        module_names: Extra import names whose versions should be printed.
        n: Number of samples.
        p: Number of input features or voxels.
        k: Number of selected features.
        repeats: Repeat count.
        cases: Benchmark case definitions.
        server_name: Host or user-specified benchmark server label.
    Output:
        None.
    This function reports enough runtime context to interpret saved speed-test
    numbers later.
    """
    print("Environment")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  numpy: {np.__version__}")
    for module_name in module_names:
        print(f"  {module_name}: {module_version(module_name)}")
    print(f"  fastl2lir: {getattr(fastl2lir_module, '__version__', '(unknown)')}")
    print(f"  fastl2lir file: {package_file(fastl2lir_module)}")
    print(f"  server: {server_name}")
    for env_name in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        print(f"  {env_name} env: {os.environ.get(env_name, '(unset)')}")
    try:
        numba = importlib.import_module("numba")
    except ImportError:
        pass
    else:
        print(f"  numba.get_num_threads(): {numba.get_num_threads()}")
    print(f"  benchmark size: n={n}, p={p}, n_feat={k}, repeats={repeats}")
    print(f"  benchmark cases: {', '.join(name for name, *_ in cases)}")


def print_intro(comparison, runners):
    """Print the experiment description before benchmark cases run.

    Inputs:
        comparison: Human-readable comparison label.
        runners: Ordered list of runner dictionaries.
    Output:
        None.
    This function describes the primary metric, runner labels, data freshness,
    and ordering policy.
    """
    print("\nSpeed test results")
    print(f"  comparison: {comparison}")
    print("  primary metric: median runtime")
    print(f"  runners: {', '.join(runner['label'] for runner in runners)}")
    print("  execution: runners are fit sequentially, one implementation at a time")
    print("  tqdm: not patched")
    print("  data: regenerated for every repeat")
    if len(runners) >= 2:
        print("  order: alternates between forward and reverse runner order")
    print("")


def print_runner_sources(fastl2lir_module, runners):
    """Print module and argument details for each runner.

    Inputs:
        fastl2lir_module: Default module that provides FastL2LiR.
        runners: Ordered list of runner dictionaries.
    Output:
        None.
    This function records the module, model arguments, and fit arguments that
    define each computation.
    """
    print("Runner sources")
    for runner in runners:
        module = runner_module(fastl2lir_module, runner)
        model_kwargs = runner.get("model_kwargs", {})
        fit_kwargs = runner.get("fit_kwargs", {})
        print(f"  {runner['label']} module: {module.__name__}")
        print(f"  {runner['label']} file: {package_file(module)}")
        print(f"  {runner['label']} model_kwargs: {model_kwargs}")
        print(f"  {runner['label']} fit_kwargs: {fit_kwargs}")
    print("")


def run_speedtest(
    fastl2lir_module,
    runners,
    output_csv,
    comparison,
    module_names=(),
):
    """Run the full speed-test workflow and save the CSV summary.

    Inputs:
        fastl2lir_module: Default module that provides FastL2LiR.
        runners: Ordered runner dictionaries.
        output_csv: Path where the summary CSV will be saved.
        comparison: Human-readable comparison label.
        module_names: Extra import names whose versions should be printed.
    Output:
        None.
    This function orchestrates environment reporting, case measurement,
    human-readable summaries, CSV writing, and saved-path reporting.
    """
    runners = filter_runners(runners)
    n, p, k, repeats = get_benchmark_settings()
    cases = make_cases(k, repeats)
    server_name = os.environ.get("BENCH_SERVER") or socket.gethostname()

    print_environment(
        fastl2lir_module, module_names, n, p, k, repeats, cases, server_name
    )
    print_intro(comparison, runners)
    print_runner_sources(fastl2lir_module, runners)

    rows = []
    for name, y_shape, params, case_repeats in cases:
        print(name)
        q = int(np.prod(y_shape))
        print(
            f"  hyperparameters: n={n}, p={p}, k={params['n_feat']}, "
            f"q={q}, repeats={case_repeats}"
        )
        times_by_label, max_pred_diff = measure_case(
            fastl2lir_module, n, p, y_shape, params, case_repeats, runners
        )
        medians = median_by_label(times_by_label)
        rows.append(
            {
                "case": name,
                "xshape": (n, p),
                "yshape": (n,) + y_shape,
                "n": n,
                "p": p,
                "k": params["n_feat"],
                "q": q,
                "repeats": case_repeats,
                "times_by_label": times_by_label,
                "medians": medians,
                "speedups": speedup_values(runners, medians),
                "max_pred_diff": max_pred_diff,
            }
        )
        print("")

    print_case_summaries(rows, runners)
    write_summary(output_csv, rows, runners, server_name)
    print(f"\nSaved CSV: {output_csv}")


def print_case_summaries(rows, runners):
    """Print per-case timing and correctness summaries.

    Inputs:
        rows: Case summary dictionaries produced by run_speedtest.
        runners: Ordered list of runner dictionaries.
    Output:
        None.
    This function presents raw repeated times, median times, named speedups,
    and maximum prediction differences for researcher inspection.
    """
    for row in rows:
        print(row["case"])
        print(f"  X={row['xshape']}, Y={row['yshape']}")
        print(
            f"  hyperparameters: n={row['n']}, p={row['p']}, "
            f"k={row['k']}, q={row['q']}, repeats={row['repeats']}"
        )
        for runner in runners:
            label = runner["label"]
            rounded_times = [round(t, 4) for t in row["times_by_label"][label]]
            print(f"  {label} times: {rounded_times} s")
        median_text = ", ".join(
            f"{runner['label']}={row['medians'][runner['label']]:.4f}s"
            for runner in runners
        )
        for runner in runners[1:]:
            baseline = runners[0]["label"]
            label = runner["label"]
            median_text += (
                f", {label}_speedup_vs_{baseline}={row['speedups'][label]:.2f}x"
            )
        print(f"  median: {median_text}")
        print(f"  max prediction diff on first samples: {row['max_pred_diff']:.3e}")
        print("")


def write_summary(output_csv, rows, runners, server_name):
    """Write and print the CSV benchmark summary.

    Inputs:
        output_csv: Path where the summary CSV will be saved.
        rows: Case summary dictionaries produced by run_speedtest.
        runners: Ordered list of runner dictionaries.
        server_name: Host or user-specified benchmark server label.
    Output:
        None.
    This function creates the temp output directory when needed and writes the
    same summary rows that are printed to the terminal.
    """
    output_csv = Path(output_csv)

    print("Summary table")
    print(summary_header(runners))
    for row in rows:
        print(summary_line(server_name, row, runners))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write(summary_header(runners) + "\n")
        for row in rows:
            f.write(summary_line(server_name, row, runners) + "\n")
