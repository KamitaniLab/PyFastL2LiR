"""Tests for the optional numba solver."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

import fastl2lir
import fastl2lir._numba as fastl2lir_numba


class TestNumbaSolver(TestCase):
    """Tests for optional numba feature-selection fitting."""

    def skip_if_mkl_backend(self):
        """Skip solver execution tests when the current BLAS backend is MKL."""
        fastl2lir_numba._load_blas_backend_for_detection()
        if fastl2lir_numba._uses_mkl():
            self.skipTest("numba solver is not supported with Intel MKL")

    def test_numba_solver_matches_numpy_solver(self):
        """Test that the numba feature-selection solver matches numpy."""

        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba is not installed")
        self.skip_if_mkl_backend()

        rng = np.random.default_rng(0)
        X = rng.normal(size=(24, 12))
        Y = rng.normal(size=(24, 5))
        X_test = rng.normal(size=(7, 12))

        model_numpy = fastl2lir.FastL2LiR()
        model_numba = fastl2lir.FastL2LiR()

        model_numpy.fit(X, Y, alpha=0.5, n_feat=4, solver="numpy")
        model_numba.fit(X, Y, alpha=0.5, n_feat=4, solver="numba")

        np.testing.assert_allclose(model_numba.W, model_numpy.W)
        np.testing.assert_allclose(model_numba.b, model_numpy.b)
        np.testing.assert_allclose(
            model_numba.predict(X_test), model_numpy.predict(X_test)
        )

    def test_numba_solver_matches_numpy_solver_float32(self):
        """Test that float32 numba fitting agrees with numpy within float32 tolerance."""

        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba is not installed")
        self.skip_if_mkl_backend()

        rng = np.random.default_rng(3)
        X = rng.normal(size=(32, 14)).astype(np.float32)
        Y = rng.normal(size=(32, 6)).astype(np.float32)
        X_test = rng.normal(size=(9, 14)).astype(np.float32)

        model_numpy = fastl2lir.FastL2LiR()
        model_numba = fastl2lir.FastL2LiR()

        model_numpy.fit(X, Y, alpha=0.5, n_feat=5, dtype=np.float32, solver="numpy")
        model_numba.fit(X, Y, alpha=0.5, n_feat=5, dtype=np.float32, solver="numba")

        self.assertEqual(model_numpy.W.dtype, np.float32)
        self.assertEqual(model_numba.W.dtype, np.float32)
        np.testing.assert_allclose(model_numba.W, model_numpy.W, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(model_numba.b, model_numpy.b, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            model_numba.predict(X_test, dtype=np.float32),
            model_numpy.predict(X_test, dtype=np.float32),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_numba_solver_matches_numpy_solver_with_tied_correlations(self):
        """Test stable feature selection when correlations tie at the cutoff."""

        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba is not installed")
        self.skip_if_mkl_backend()

        h1 = np.array([1, -1, 1, -1, 1, -1, 1, -1.0])
        h2 = np.array([1, 1, -1, -1, 1, 1, -1, -1.0])
        h3 = np.array([1, 1, 1, 1, -1, -1, -1, -1.0])
        h4 = np.array([1, -1, -1, 1, 1, -1, -1, 1.0])
        X = np.column_stack([h1, h2, h3, h4])
        Y = np.column_stack([2 * h1 + h2 + h3, -h1 + h2 + h3])
        X_test = X[[0, 2, 4], :]

        model_numpy = fastl2lir.FastL2LiR()
        model_numba = fastl2lir.FastL2LiR()

        model_numpy.fit(X, Y, alpha=0.5, n_feat=2, solver="numpy")
        model_numba.fit(X, Y, alpha=0.5, n_feat=2, solver="numba")

        np.testing.assert_allclose(model_numba.W, model_numpy.W)
        np.testing.assert_allclose(model_numba.b, model_numpy.b)
        np.testing.assert_allclose(
            model_numba.predict(X_test), model_numpy.predict(X_test)
        )

    def test_solver_is_validated_before_fit_path_selection(self):
        """Test that invalid solver names are rejected even without feature selection."""

        rng = np.random.default_rng(4)
        X = rng.normal(size=(16, 5))
        Y = rng.normal(size=(16, 3))

        with self.assertRaisesRegex(ValueError, "solver must be"):
            fastl2lir.FastL2LiR().fit(X, Y, n_feat=0, solver="typo")

    def test_numba_solver_warns_when_falling_back_to_numpy_fit_path(self):
        """Test that numba requests warn when the numba path is not used."""

        rng = np.random.default_rng(5)
        X = rng.normal(size=(16, 5))
        Y = rng.normal(size=(16, 3))

        with self.assertWarnsRegex(UserWarning, "solver='numba' is only used"):
            fastl2lir.FastL2LiR().fit(X, Y, n_feat=0, solver="numba")

        with self.assertWarnsRegex(UserWarning, "solver='numba' is only used"):
            fastl2lir.FastL2LiR().fit(
                X, Y, n_feat=3, solver="numba", save_select_feat=True
            )

    def test_numba_solver_restores_thread_count(self):
        """Test that the numba solver does not leak thread-count changes."""

        try:
            from numba import get_num_threads, set_num_threads
        except ImportError:
            self.skipTest("numba is not installed")
        self.skip_if_mkl_backend()

        rng = np.random.default_rng(1)
        X = rng.normal(size=(24, 12))
        Y = rng.normal(size=(24, 5))
        original_num_threads = get_num_threads()

        try:
            fastl2lir.FastL2LiR().fit(X, Y, n_feat=4, solver="numba")
            self.assertEqual(get_num_threads(), original_num_threads)

            set_num_threads(1)
            fastl2lir.FastL2LiR().fit(
                X, Y, n_feat=4, solver="numba", numba_num_threads=None
            )
            self.assertEqual(get_num_threads(), 1)
        finally:
            set_num_threads(original_num_threads)

    def test_numba_solver_rejects_mkl_backend(self):
        """Test that numba fitting is rejected with an Intel MKL backend."""

        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("numba is not installed")

        rng = np.random.default_rng(2)
        X = rng.normal(size=(24, 12))
        Y = rng.normal(size=(24, 5))

        with patch.object(
            fastl2lir_numba,
            "threadpool_info",
            return_value=[
                {
                    "user_api": "blas",
                    "internal_api": "mkl",
                    "prefix": "libmkl_rt",
                    "filepath": "/example/libmkl_rt.so",
                }
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "Intel MKL"):
                fastl2lir.FastL2LiR().fit(X, Y, n_feat=4, solver="numba")

    def test_numba_solver_allows_non_mkl_backend(self):
        """Test that numba environment validation allows non-MKL BLAS."""

        with (
            patch.object(
                fastl2lir_numba,
                "_load_blas_backend_for_detection",
                return_value=None,
            ),
            patch.object(
                fastl2lir_numba,
                "threadpool_info",
                return_value=[
                    {
                        "user_api": "blas",
                        "internal_api": "blis",
                        "prefix": "libblis",
                        "filepath": "/example/libblis.so",
                    }
                ],
            ),
        ):
            fastl2lir_numba.check_numba_solver_environment()
