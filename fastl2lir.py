# At v2, the input dimensions with corrlations = NaN are excluded.

import math

import numpy as np
from numpy.matlib import repmat


class FastL2LiR():
    '''Fast L2-regularized linear regression class.'''

    def __init__(self, W=np.array([]), b=np.array([])):
        self.__W = W
        self.__b = b

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W):
        self.__W = W

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b):
        self.__b == b

    def fit(self, X, Y, alpha=0, n_feat=0, chunk_size=0, cache_dir='./cache'):
        '''Fit the L2-regularized linear model with the given data.

        Parameters
        ----------
        X, Y : array_like
            Training inputs (data and targets).
        alpha: float
            Regularization parameter (coefficient for L2-norm).
        n_feta: int
            The number of selected input features.

        Returns
        -------
        self
            Returns an instance of self.
        '''

        # Reshape Y
        reshape_y = Y.ndim > 2

        if reshape_y:
            Y_shape = Y.shape
            Y = Y.reshape(Y.shape[0], -1, order='F')

        # Feature selection settings
        if n_feat == 0:
            n_feat = X.shape[1]

        no_feature_selection = X.shape[1] == n_feat

        # Chunking
        if chunk_size > 0:
            chunks = self.__get_chunks(range(Y.shape[1]), chunk_size)
            print('Num chunks: %d' % len(chunks))
            w_list = []
            b_list = []
            for i, chunk in enumerate(chunks):
                W, b = self.__sub_fit(X, Y[0:, chunk], alpha=alpha, n_feat=n_feat, use_all_features=no_feature_selection)
                w_list.append(W)
                b_list.append(b)
            W = np.hstack(w_list)
            b = np.hstack(b_list)
        else:
            W, b = self.__sub_fit(X, Y, alpha=alpha, n_feat=n_feat, use_all_features=no_feature_selection)

        self.__W = W
        self.__b = b

        if reshape_y:
            Y = Y.reshape(Y_shape, order='F')
            self.__W = self.__W.reshape((self.__W.shape[0],) + Y_shape[1:], order='F')
            self.__b = self.__b.reshape((1,) + Y_shape[1:], order='F')

        return self

    def predict(self, X):
        '''Predict with the fitted linear model.

        Parameters
        ----------
        X : array_like

        Returns
        -------
        Y : array_like
        '''

        # Reshape
        reshape_y = self.__W.ndim > 2
        if reshape_y:
            Y_shape = self.__W.shape
            W = self.__W.reshape(self.__W.shape[0], -1, order='F')
            b = self.__b.reshape(self.__b.shape[0], -1, order='F')
        else:
            W = self.__W
            b = self.__b

        # Prediction
        Y = np.matmul(X, W) + np.matmul(np.ones((X.shape[0], 1)), b)

        if reshape_y:
            Y = Y.reshape((Y.shape[0],) + Y_shape[1:], order='F')

        return Y

    def __sub_fit(self, X, Y, alpha=0, n_feat=0, use_all_features=True):
        if use_all_features:
            # Without feature selection
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            Wb = np.linalg.solve(np.matmul(X.T, X)+alpha *
                                np.eye(X.shape[1]), np.matmul(X.T, Y))
            W = W[0:-1, :]
            b = W[-1, :]
        else:
            # With feature selection
            W = np.zeros((X.shape[1], Y.shape[1]))
            b = np.zeros((1, Y.shape[1]))
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
                Wb = np.linalg.solve(W0[I][:, I], W1[I][:, index_outputDim])
                for index_selectedDim in range(n_feat):
                    W[I[index_selectedDim], index_outputDim] = Wb[index_selectedDim]
                b[0, index_outputDim] = Wb[-1]

        return W, b

    def __get_chunks(self, a, chunk_size):
        n_chunk = int(math.ceil(len(a) / float(chunk_size)))

        chunks = []
        for i in range(n_chunk):
            index_start = i * chunk_size
            index_end = (i + 1) * chunk_size
            index_end = len(a) if index_end > len(a) else index_end
            chunks.append(a[index_start:index_end])

        return chunks


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
