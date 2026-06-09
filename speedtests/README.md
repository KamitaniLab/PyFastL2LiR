# Speed Tests

This directory contains speed-run scripts for measuring PyFastL2LiR fitting
runtime. These scripts are not the package test suite; correctness tests live
under `tests/`.

The speed runs generate synthetic fMRI-like regression data, fit FastL2LiR on
several target shapes, compare predictions between runners when more than one
runner is used, print timing summaries, and save CSV summaries in this
directory.

## Files

- `speedrun.sh`: Run the current working-tree package with the project Python
  environment. It compares `solver="numpy"` and `solver="numba"`.
- `speedrun.py`: Python entry point used by `speedrun.sh`.
- `speedrun_legacy.sh`: Run the legacy installed `fastl2lir` package from a
  configured legacy Python environment.
- `speedrun_legacy.py`: Python entry point used by `speedrun_legacy.sh`.
- `speedrun_mkl.sh`: Run the current working-tree package from a configured
  MKL-oriented Python environment.
- `speedtest_common.py`: Shared benchmark setup, timing, prediction checks,
  summaries, and CSV writing.
- `local.sh.example`: Template for machine-local environment paths.

## Local Setup

Copy `local.sh.example` to `local.sh` and edit paths for the machine:

```bash
cp speedtests/local.sh.example speedtests/local.sh
```

`local.sh` is ignored by git. Relative paths are resolved from the repository
root.

## Running

Current environment:

```bash
./speedtests/speedrun.sh
```

Legacy environment:

```bash
./speedtests/speedrun_legacy.sh
```

MKL-oriented environment:

```bash
./speedtests/speedrun_mkl.sh
```

For a quick smoke run, restrict the synthetic data size and case list:

```bash
BENCH_N=24 BENCH_P=12 BENCH_K=4 BENCH_REPEATS=1 BENCH_CASES=fc8 ./speedtests/speedrun.sh
```

## Controls

- `BENCH_N`: Number of samples. Default: `1000`.
- `BENCH_P`: Number of input features or voxels. Default: `15000`.
- `BENCH_K`: Number of selected features. Default: `500`.
- `BENCH_REPEATS`: Repeats per benchmark case. Default: `3`.
- `BENCH_CASES`: Comma-separated subset of `fc8`, `fc6`, `conv5`, and
  `conv5_chunk10`.
- `BENCH_RUNNERS`: Comma-separated subset of available runner labels. The
  current speed run supports `numpy` and `numba`; the legacy speed run supports
  `legacy`.
- `BENCH_SERVER`: Label written to the terminal output and CSV. Defaults to the
  hostname.

## Outputs

CSV files are written under `speedtests/`:

- `speedtest_<server>.csv` for the current and MKL-oriented speed runs.
- `speedtest_legacy_<server>.csv` for the legacy speed run.

The CSV columns include the server label, benchmark case, matrix dimensions,
repeat count, median runtime columns, speedup columns when multiple runners are
present, and the maximum prediction difference on a small validation slice.
