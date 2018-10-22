'''Tests for PyFastL2LiR'''


import unittest
import pickle

import numpy as np
from numpy.testing import assert_array_equal

import fastl2lir


class TestFastL2LiR(unittest.TestCase):
    '''Tests for FastL2LiR'''

    def __init__(self, *args, **kwargs):
        super(TestFastL2LiR, self).__init__(*args, **kwargs)

    def test_basic(self):
        '''Basic Test.'''

        with open('test/test_x.pkl', 'rb') as f:
            X = pickle.load(f)

        with open('test/test_y.pkl', 'rb') as f:
            Y = pickle.load(f)

        with open('test/test_model.pkl', 'rb') as f:
            model_true = pickle.load(f)

        with open('test/test_predicted.pkl', 'rb') as f:
            predicted_true = pickle.load(f)

        alpha = 1.0
        n_feat = 50

        model_test = fastl2lir.fastL2LiR()
        model_test.fit(X, Y, alpha, n_feat)
        predicted_test = model_test.predict(X)

        np.testing.assert_array_equal(model_test.W, model_true.W)
        np.testing.assert_array_equal(model_test.b, model_true.b)
        np.testing.assert_array_equal(predicted_test, predicted_true)


if __name__ == "__main__":
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFastL2LiR)
    unittest.TextTestRunner(verbosity=2).run(test_suite)
