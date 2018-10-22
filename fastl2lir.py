# At v2, the input dimensions with corrlations = NaN are excluded.


import numpy as np
from numpy.matlib import repmat


class FastL2LiR():
    '''Fast L2-regularized linear regression class.'''

    def __init__(self):
        self.__W = np.array([])
        self.__b = np.array([])

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    def fit(self, X, Y, alpha=0, n_feat=0):

        if n_feat == 0:
            n_feat = X.shape[1]

        feature_selection = X.shape[1] == n_feat

        if feature_selection:
            # Without feature selection
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            W = np.linalg.solve(np.matmul(X.T, X)+alpha *
                                np.eye(X.shape[1]), np.matmul(X.T, Y))
            self.__W = W[0:-1, :]
            self.__b = W[-1, :]
        else:
            # With feature selection
            self.__W = np.zeros((X.shape[1], Y.shape[1]))
            self.__b = np.zeros((1, Y.shape[1]))
            I = np.nonzero(np.var(X, axis=0) < 0.00000001)
            C = corrmat(X, Y, 'col')
            C[I, :] = 0.0
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            W0 = np.matmul(X.T, X) + alpha * np.eye(X.shape[1])
            W1 = np.matmul(X.T, Y)
            for index_outputDim in range(Y.shape[1]):
                C0 = abs(C[:, index_outputDim])
                I = np.argsort(C0)
                I = I[::-1]
                I = I[0:n_feat]
                I = np.hstack((I, X.shape[1]-1))
                W = np.linalg.solve(W0[I][:, I], W1[I][:, index_outputDim])
                for index_selectedDim in range(n_feat):
                    self.__W[I[index_selectedDim],
                             index_outputDim] = W[index_selectedDim]
                self.__b[0, index_outputDim] = W[-1]
            return self

    def predict(self, X):
        predicted = np.matmul(X, self.__W) + \
            np.matmul(np.ones((X.shape[0], 1)), self.__b)
        return predicted


# Functions ##################################################################

def corrmat(x, y, var='row'):
    """
    Returns correlation matrix between `x` and `y`

    Parameters
    ----------
    x, y : array_like
        Matrix or vector
    var : str, 'row' or 'col'
        Specifying whether rows (default) or columns represent variables

    Returns
    -------
    rmat
        Correlation matrix
    """

    # Fix x and y to represent variables in each row
    if var == 'row':
        pass
    elif var == 'col':
        x = x.T
        y = y.T
    else:
        raise ValueError('Unknown var parameter specified')

    nobs = x.shape[1]

    # Subtract mean(a, axis=1) from a
    def submean(a): return a - np.matrix(np.mean(a, axis=1)).T

    cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / \
        np.dot(np.matrix(np.std(x, axis=1, ddof=1)).T,
               np.matrix(np.std(y, axis=1, ddof=1)))

    return np.array(cmat)
