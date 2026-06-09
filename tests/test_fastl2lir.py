"""Tests for PyFastL2LiR"""

from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

import fastl2lir


class TestFastL2LiR(TestCase):
    """Tests for FastL2LiR"""

    def test_basic(self):
        """Basic Test."""

        data = np.load("./tests/testdata_basic.npz")

        model_1d = fastl2lir.FastL2LiR()
        model_2d = fastl2lir.FastL2LiR()

        model_1d.fit(data["x_tr"], data["y_1d"])
        model_2d.fit(data["x_tr"], data["y_2d"])

        yp_1d = model_1d.predict(data["x_te"])
        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_1d.W, data["w_1d"])
        np.testing.assert_array_almost_equal(model_1d.b, data["b_1d"])
        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_1d, data["yp_1d"])
        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_alpha(self):
        """Test for alpha."""

        data = np.load("./tests/testdata_alpha01.npz")

        model_1d = fastl2lir.FastL2LiR()
        model_2d = fastl2lir.FastL2LiR()

        model_1d.fit(data["x_tr"], data["y_1d"], alpha=0.1)
        model_2d.fit(data["x_tr"], data["y_2d"], alpha=0.1)

        yp_1d = model_1d.predict(data["x_te"])
        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_1d.W, data["w_1d"])
        np.testing.assert_array_almost_equal(model_1d.b, data["b_1d"])
        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_1d, data["yp_1d"])
        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_nfeat(self):
        """Test for n_feat."""

        data = np.load("./tests/testdata_nfeat.npz")

        model_1d = fastl2lir.FastL2LiR()
        model_2d = fastl2lir.FastL2LiR()

        model_1d.fit(data["x_tr"], data["y_1d"], n_feat=20)
        model_2d.fit(data["x_tr"], data["y_2d"], n_feat=20)

        yp_1d = model_1d.predict(data["x_te"])
        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_1d.W, data["w_1d"])
        np.testing.assert_array_almost_equal(model_1d.b, data["b_1d"])
        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_1d, data["yp_1d"])
        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_nfeat_no_feature_selection(self):
        """Test for n_feat when X.shape[1] < n_feat (more samples than features)."""

        data = np.load("./tests/testdata_basic.npz")

        model_1d = fastl2lir.FastL2LiR()
        model_2d = fastl2lir.FastL2LiR()

        model_1d.fit(data["x_tr"], data["y_1d"], n_feat=500)
        model_2d.fit(data["x_tr"], data["y_2d"], n_feat=500)

        yp_1d = model_1d.predict(data["x_te"])
        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_1d.W, data["w_1d"])
        np.testing.assert_array_almost_equal(model_1d.b, data["b_1d"])
        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_1d, data["yp_1d"])
        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_nfeat_no_feature_selection_wide(self):
        """Test for n_feat when X.shape[1] < n_feat (more features than samples)."""

        data = np.load("./tests/testdata_wide.npz")

        model_1d = fastl2lir.FastL2LiR()
        model_2d = fastl2lir.FastL2LiR()

        model_1d.fit(data["x_tr"], data["y_1d"], n_feat=500)
        model_2d.fit(data["x_tr"], data["y_2d"], n_feat=500)

        yp_1d = model_1d.predict(data["x_te"])
        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_1d.W, data["w_1d"])
        np.testing.assert_array_almost_equal(model_1d.b, data["b_1d"])
        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_1d, data["yp_1d"])
        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_chunk(self):
        """Test for chunk_size."""

        data = np.load("./tests/testdata_chunk.npz")

        model_2d = fastl2lir.FastL2LiR()

        model_2d.fit(data["x_tr"], data["y_2d"], chunk_size=10)

        yp_2d = model_2d.predict(data["x_te"])

        np.testing.assert_array_almost_equal(model_2d.W, data["w_2d"])
        np.testing.assert_array_almost_equal(model_2d.b, data["b_2d"])

        np.testing.assert_array_almost_equal(yp_2d, data["yp_2d"])

    def test_reshape(self):
        """Test for reshaping."""
        Y_shape = (200, 10, 10, 5)

        X = np.random.rand(200, 100)
        Y = np.random.rand(*Y_shape)

        alpha = 1.0
        n_feat = 50

        model_test = fastl2lir.FastL2LiR()
        model_test.fit(X, Y, alpha, n_feat)
        pred_test = model_test.predict(X)

        np.testing.assert_array_equal(model_test.W.shape, (100,) + Y_shape[1:])
        np.testing.assert_array_equal(model_test.b.shape, (1,) + Y_shape[1:])
        np.testing.assert_array_equal(pred_test.shape, Y_shape)

    def test_save_select_feat_default_select_sample(self):
        """save_select_feat=True with default select_sample=None should not raise."""
        data = np.load("./tests/testdata_nfeat.npz")
        model = fastl2lir.FastL2LiR()
        model.fit(data["x_tr"], data["y_1d"], n_feat=20, save_select_feat=True)


if __name__ == "__main__":
    test_suite = TestLoader().loadTestsFromTestCase(TestFastL2LiR)
    TextTestRunner(verbosity=2).run(test_suite)
