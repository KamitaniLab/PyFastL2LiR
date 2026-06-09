"""Numba implementations for PyFastL2LiR."""

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from threadpoolctl import threadpool_info


_WARMED_UP_DTYPES = set()


def _blas_backend_names():
    """Return detected BLAS backend names from threadpoolctl."""
    names = []
    for info in threadpool_info():
        if info.get("user_api") != "blas":
            continue
        internal_api = info.get("internal_api")
        prefix = info.get("prefix")
        filepath = info.get("filepath")
        parts = [part for part in (internal_api, prefix, filepath) if part]
        names.append(" ".join(parts))
    return names


def _uses_mkl():
    """Return True when any detected BLAS backend is Intel MKL."""
    names = _blas_backend_names()
    return any("mkl" in name.lower() for name in names)


def _load_blas_backend_for_detection(dtype=np.float64):
    """Run a tiny BLAS call so threadpoolctl can see the active backend."""
    x = np.ones((1, 1), dtype=dtype)
    np.matmul(x, x)


def check_numba_solver_environment(dtype=np.float64):
    """Raise RuntimeError when solver="numba" is running with Intel MKL."""
    _load_blas_backend_for_detection(dtype)
    if not _uses_mkl():
        return

    detected = ", ".join(_blas_backend_names()) or "unknown"
    raise RuntimeError(
        "solver='numba' is not supported when the detected "
        f"BLAS backend is Intel MKL. Detected backend: {detected}"
    )


def validate_numba_solver(numba_num_threads, dtype=np.float64):
    """Validate optional numba solver dependencies, threads, and BLAS backend."""
    if numba_num_threads is not None and numba_num_threads < 1:
        raise ValueError("numba_num_threads must be a positive integer or None")

    check_numba_solver_environment(dtype)


def fit_selected_ridge_numba(X, C, W0, W1, n_feat, dtype, numba_num_threads=4):
    """Fit selected-feature ridge regression with the optional numba kernel."""
    validate_numba_solver(numba_num_threads, dtype)

    previous_num_threads = None
    if numba_num_threads is not None:
        previous_num_threads = get_num_threads()
        set_num_threads(min(numba_num_threads, previous_num_threads))

    try:
        warmup_numba(dtype)
        W = np.zeros((C.shape[0], X.shape[1] - 1), dtype=dtype)
        b = np.zeros((1, C.shape[0]), dtype=dtype)
        return _fit_selected_ridge_numba(X, C, W, b, W0, W1, n_feat)
    finally:
        if previous_num_threads is not None:
            set_num_threads(previous_num_threads)


@njit(cache=True, parallel=True)
def _fit_selected_ridge_numba(X, C, W, b, W0, W1, n_feat):
    n_outputs = C.shape[0]
    bias_index = X.shape[1] - 1

    for index_outputDim in prange(n_outputs):
        C0 = np.abs(C[index_outputDim, :])
        feat_idx = np.argsort(C0, kind="mergesort")[::-1]
        feat_idx = feat_idx[:n_feat]

        I_with_bias = np.empty(feat_idx.size + 1, dtype=feat_idx.dtype)
        for i in range(feat_idx.size):
            I_with_bias[i] = feat_idx[i]
        I_with_bias[feat_idx.size] = bias_index

        W0_sub = np.zeros((I_with_bias.size, I_with_bias.size), dtype=W0.dtype)
        for i in range(I_with_bias.size):
            for j in range(I_with_bias.size):
                W0_sub[i, j] = W0[I_with_bias[i], I_with_bias[j]]

        rhs = np.zeros((I_with_bias.size, 1), dtype=W1.dtype)
        for i in range(I_with_bias.size):
            rhs[i, 0] = W1[index_outputDim, I_with_bias[i]]

        Wb = np.linalg.solve(W0_sub.astype(np.float64), rhs.astype(np.float64)).astype(
            W.dtype
        )

        for i in range(feat_idx.size):
            W[index_outputDim, feat_idx[i]] = Wb[i, 0]
        b[0, index_outputDim] = Wb[-1, 0]

    return W.T, b


def warmup_numba(dtype=np.float64):
    """Compile or load the numba ridge kernel before fitting real data.

    The feature-selection fit path calls ``_fit_selected_ridge_numba`` once
    per fit. With ``cache=True``, numba can reuse compiled code across Python
    processes, but the first call in a process still pays dispatcher/cache-load
    and sometimes compilation overhead. If that first call happens on the real
    training matrix, benchmark and notebook users see a large first-fit spike.

    Warm up the same kernel once per dtype with tiny arrays so the one-time
    cost is paid before the real solve loop. This does not remove numba's
    startup cost, but it keeps that cost independent of the user's data size
    and makes subsequent feature-selection fits reflect steady-state speed.
    """
    dtype = np.dtype(dtype)
    if dtype in _WARMED_UP_DTYPES:
        return

    X = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=dtype,
    )
    C = np.array(
        [
            [0.3, 0.2],
            [0.1, 0.4],
        ],
        dtype=dtype,
    )
    W = np.zeros((2, 2), dtype=dtype)
    b = np.zeros((1, 2), dtype=dtype)
    W0 = np.matmul(X.T, X) + np.eye(X.shape[1], dtype=dtype)
    W1 = np.array(
        [
            [1.0, 0.5, 1.5],
            [0.5, 1.0, 1.5],
        ],
        dtype=dtype,
    )

    _fit_selected_ridge_numba(X, C, W, b, W0, W1, 1)
    _WARMED_UP_DTYPES.add(dtype)
