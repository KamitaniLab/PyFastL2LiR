"""PyFastL2LiR: Fast L2-regularized Linear Regression."""

import math
from time import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from scipy import linalg as sp_linalg


class FastL2LiR(object):
    """Fast L2-regularized linear regression class."""

    def __init__(self, W=np.array([]), b=np.array([]), verbose=False, solver='scipy'):
        self.__W = W
        self.__b = b
        self.__verbose = verbose
        self.__solver = solver

        # Choose linear solver once to avoid branching at call sites
        if solver == 'scipy':
            def _solve(a, b):
                return sp_linalg.solve(a, b, assume_a='sym', check_finite=False)
        elif solver == 'numpy':
            def _solve(a, b):
                return np.linalg.solve(a, b)
        else:
            raise ValueError('Unknown solver: %s' % solver)
        self.__solve = _solve

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
        self.__b = b

    @property
    def S(self):
        return self.__S

    @S.setter
    def S(self, S):
        self.__S = S

    def fit(
        self,
        X,
        Y,
        alpha=1.0,
        n_feat=0,
        save_select_feat=False,
        spatial_norm=None,
        select_sample=None,
        chunk_size=0,
        cache_dir="./cache",
        dtype=np.float64,
    ):
        """Fit the L2-regularized linear model with the given data.

        Parameters
        ----------
        X, Y : array_like
            Training inputs (data and targets).
        alpha: float
            Regularization parameter (coefficient for L2-norm).
        n_feta: int
            The number of selected input features.
        save_select_feat: bool
            Save bool matrix indicating selected voxel for each unit.
            Since fitting is performed for each unit, the amount of memory
            required at runtime can be reduced (On the other hand, computation
            time and storage requirements increase).
        spatial_norm: str (None, 'norm1', 'norm2', 'std1', 'std1mean0', 'norm1mean0', or 'norm2mean0')
            Perform spatial normalization (sample unit) on the voxel selected
            for each unit. Selecting this automatically sets 'save_select_feat'
            to True because it is necessary to save the index matrix of the
            selected voxel
        select_sample: str ('nan_remove' or None)
            Specify how to select training samples
            Selecting this automatically sets 'save_select_feat' to True
            because this is an operation for each unit.
            (The sample selection operation itself does not essentially need
            to record the selected voxel.)

        Returns
        -------
        self
            Returns an instance of self.
        """

        if X.dtype != dtype:
            X = X.astype(dtype)
        if Y.dtype != dtype:
            Y = Y.astype(dtype)

        # Reshape Y
        reshape_y = Y.ndim > 2

        if reshape_y:
            Y_shape = Y.shape
            Y = Y.reshape(Y.shape[0], -1, order="F")

        # Feature selection settings
        if n_feat == 0:
            n_feat = X.shape[1]

        no_feature_selection = X.shape[1] == n_feat

        if n_feat > X.shape[1]:
            warnings.warn(
                "X has less features than n_feat (X.shape[1] < n_feat). Feature selection is not applied."
            )
            no_feature_selection = True

        # # Save selected voxel mode
        if not save_select_feat:
            if (spatial_norm is not None) or (select_sample is not None):
                save_select_feat = True

        # Chunking
        if chunk_size > 0:
            chunks = self.__get_chunks(range(Y.shape[1]), chunk_size)

            if self.__verbose:
                print("Num chunks: %d" % len(chunks))

            w_list = []
            b_list = []
            s_list = []
            for i, chunk in enumerate(chunks):
                start_time = time()
                if save_select_feat:
                    W, b, S = self.__sub_fit_save_select_feat(
                        X,
                        Y[0:, chunk],
                        alpha=alpha,
                        n_feat=n_feat,
                        spatial_norm=spatial_norm,
                        use_all_features=no_feature_selection,
                        select_sample=select_sample,
                        dtype=dtype,
                    )
                    s_list.append(S)
                else:
                    W, b = self.__sub_fit(
                        X,
                        Y[0:, chunk],
                        alpha=alpha,
                        n_feat=n_feat,
                        use_all_features=no_feature_selection,
                        dtype=dtype,
                    )
                w_list.append(W)
                b_list.append(b)

                if self.__verbose:
                    print("Chunk %d (time: %f s)" % (i + 1, time() - start_time))

            W = np.hstack(w_list)
            b = np.hstack(b_list)
            if save_select_feat:
                S = np.hstack(s_list)
        else:
            if save_select_feat:
                W, b, S = self.__sub_fit_save_select_feat(
                    X,
                    Y,
                    alpha=alpha,
                    n_feat=n_feat,
                    spatial_norm=spatial_norm,
                    use_all_features=no_feature_selection,
                    select_sample=select_sample,
                    dtype=dtype,
                )
            else:
                W, b = self.__sub_fit(
                    X,
                    Y,
                    alpha=alpha,
                    n_feat=n_feat,
                    use_all_features=no_feature_selection,
                    dtype=dtype,
                )

        self.__W = W
        self.__b = b
        if save_select_feat:
            self.__S = S

        if reshape_y:
            Y = Y.reshape(Y_shape, order="F")
            self.__W = self.__W.reshape((self.__W.shape[0],) + Y_shape[1:], order="F")
            self.__b = self.__b.reshape((1,) + Y_shape[1:], order="F")
            if save_select_feat:
                self.__S = self.__S.reshape(
                    (self.__S.shape[0],) + Y_shape[1:], order="F"
                )

        return self

    def predict(self, X, dtype=np.float64, save_select_feat=False, spatial_norm=None):
        """Predict with the fitted linear model.

        Parameters
        ----------
        X : array_like
        save_select_feat: bool
            Load bool matrix indicating selected voxel for each unit.
            If save_select_feat is True during training, it must be true
            during testing as well.
        spatial_norm: str (None, 'norm1', 'norm2', 'std1', 'std1mean0', 'norm1mean0', or 'norm2mean0')
            Perform spatial normalization (sample unit) on the voxel selected
            for each unit. It is necessary to specify the same spatial_norm
            method as during training.
        Returns
        -------
        Y : array_like
        """
        if X.dtype != dtype:
            X = X.astype(dtype)

        # Save selected voxel mode
        if not save_select_feat:
            if spatial_norm is not None:
                save_select_feat = True

        # Reshape
        reshape_y = self.__W.ndim > 2
        if reshape_y:
            Y_shape = self.__W.shape
            W = self.__W.reshape(self.__W.shape[0], -1, order="F")
            b = self.__b.reshape(self.__b.shape[0], -1, order="F")
            if save_select_feat:
                S = self.__S.reshape(self.__S.shape[0], -1, order="F")
        else:
            W = self.__W
            b = self.__b
            if save_select_feat:
                S = self.__S

        # Prediction
        if save_select_feat:
            Y = np.zeros((X.shape[0], W.shape[1]), dtype=dtype)
            for si in range(W.shape[1]):  # Loop for feature
                selected_voxel = S[:, si]
                newX = X[:, selected_voxel]  # extract selected features

                # Perform the sample normalization.
                newX = self.__apply_spatial_normalization(newX, spatial_norm)

                # Predict
                newW = W[:, si].reshape(-1, 1)
                newW = newW[selected_voxel, :].reshape(
                    -1, 1
                )  # extract selected features
                Y[:, si] = (np.matmul(newX, newW) + b[:, si]).flatten()
        else:
            Y = np.matmul(X, W) + np.matmul(np.ones((X.shape[0], 1), dtype=dtype), b)

        if reshape_y:
            Y = Y.reshape((Y.shape[0],) + Y_shape[1:], order="F")

        return Y

    def __sub_fit(
        self, X, Y, alpha=0, n_feat=0, use_all_features=True, dtype=np.float64
    ):
        if use_all_features:
            # Without feature selection
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=dtype)))

            # Choose the more efficient method based on matrix dimensions
            if X.shape[0] > X.shape[1]:
                # Use primal form for tall matrices (more samples than features)
                Wb = self.__solve(
                    np.matmul(X.T, X) + alpha * np.eye(X.shape[1], dtype=dtype),
                    np.matmul(X.T, Y),
                )
            else:
                # Use dual form for wide matrices (more features than samples)
                Wb = np.matmul(
                    X.T,
                    self.__solve(
                        np.matmul(X, X.T) + alpha * np.eye(X.shape[0], dtype=dtype), Y
                    ),
                )
            W = Wb[0:-1, :]
            b = Wb[-1, :][np.newaxis, :]  # Returning b as a 2D array
        else:
            # With feature selection
            W = np.zeros((Y.shape[1], X.shape[1]), dtype=dtype)
            b = np.zeros((1, Y.shape[1]), dtype=dtype)
            zero_var_idx = np.nonzero(np.var(X, axis=0) < 0.00000001)
            C = corrmat(X, Y, "col")
            C[zero_var_idx, :] = 0.0
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=dtype)))
            W0 = np.matmul(X.T, X) + alpha * np.eye(X.shape[1], dtype=dtype)
            W1 = np.matmul(Y.T, X)
            C = C.T

            with threadpool_limits(limits=1, user_api="blas"):
                for index_outputDim in tqdm(range(Y.shape[1])):
                    C0 = abs(C[index_outputDim, :])
                    feat_idx = np.argsort(C0)
                    feat_idx = feat_idx[::-1]
                    feat_idx = feat_idx[0:n_feat]
                    feat_idx = np.hstack((feat_idx, X.shape[1] - 1))
                    W0_sub = (
                        W0.ravel()[
                            (
                                feat_idx + (feat_idx * W0.shape[1]).reshape((-1, 1))
                            ).ravel()
                        ]
                    ).reshape(feat_idx.size, feat_idx.size)
                    Wb = self.__solve(W0_sub, W1[index_outputDim, feat_idx])
                    W[index_outputDim, feat_idx[:-1]] = Wb[:-1]
                    b[0, index_outputDim] = Wb[-1]
                W = W.T

        return W, b

    def __sub_fit_save_select_feat(
        self,
        X,
        Y,
        alpha=0,
        n_feat=0,
        spatial_norm=None,
        use_all_features=True,
        select_sample=None,
        dtype=np.float64,
    ):
        """
        Execute fitting for each unit.
        Enables spatial normalization for selected voxels and selection of
        training samples.
        """
        # Prepare the matixes to save.
        W = np.zeros((Y.shape[1], X.shape[1]), dtype=dtype)  # feature size x voxel size
        b = np.zeros((1, Y.shape[1]), dtype=dtype)  # feautre size
        S = np.zeros(
            (Y.shape[1], X.shape[1]), dtype=np.bool_
        )  # feature size x voxel size

        with threadpool_limits(limits=1, user_api="blas"):
            for index_outputDim in tqdm(range(Y.shape[1])):
                # Select training samples
                if select_sample is None:
                    selector = slice(None)
                elif (
                    select_sample == "remove_nan"
                ):  # Delete sample with nan value in unit
                    selector = np.logical_not(np.isnan(Y[:, index_outputDim].flatten()))
                else:
                    raise RuntimeError(
                        "Not implemented selection method:", select_sample
                    )
                selX = X[selector, :]
                selY = Y[selector, index_outputDim].reshape(-1, 1)

                # Select voxels
                if use_all_features:
                    feat_idx = np.arange(selX.shape[1])
                else:
                    C0 = abs(corrmat(selX, selY, "col")).ravel()
                    feat_idx = np.argsort(C0 * -1)
                    feat_idx = feat_idx[0:n_feat]
                newX = selX[:, feat_idx]  # sample_num x voxel_num
                S[index_outputDim, feat_idx] = True

                # Perform the spatial normalization
                newX = self.__apply_spatial_normalization(newX, spatial_norm)

                # Fit
                newX = np.hstack(
                    (newX, np.ones((newX.shape[0], 1), dtype=dtype))
                )  # Add one column to rightmost column
                W0 = np.matmul(newX.T, newX) + alpha * np.eye(
                    newX.shape[1], dtype=dtype
                )
                rhs = np.matmul(selY.ravel(), newX)
                Wb = self.__solve(W0, rhs)
                W[index_outputDim, feat_idx] = Wb[:-1]
                b[0, index_outputDim] = Wb[-1]
            W = W.T
            S = np.asarray(S.T, dtype=np.bool_)  # Transpose and convert to bool type

        return W, b, S

    def __get_chunks(self, a, chunk_size):
        n_chunk = int(math.ceil(len(a) / float(chunk_size)))

        chunks = []
        for i in range(n_chunk):
            index_start = i * chunk_size
            index_end = (i + 1) * chunk_size
            index_end = len(a) if index_end > len(a) else index_end
            chunks.append(a[index_start:index_end])

        return chunks

    def __apply_spatial_normalization(self, X, spatial_norm):
        """
        Perform the spatial normalization
        """
        if spatial_norm is None:
            pass
        elif spatial_norm == "norm1":  # L1norm (Divide by L1norm on each sample)
            X = X / np.sum(np.abs(X), axis=1).reshape(X.shape[0], 1)
        elif spatial_norm == "norm2":  # L2norm (Divide by L2norm on each sample)
            X = X / np.sqrt(np.sum(np.square(X), axis=1)).reshape(X.shape[0], 1)
        elif spatial_norm == "std1":  # Normalize with STD=1
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(
                X, axis=1, ddof=1, keepdims=True
            ) + np.mean(X, axis=1, keepdims=True)
        elif spatial_norm == "std1mean0":  # Mean correction + Normalize with STD=1
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(
                X, axis=1, ddof=1, keepdims=True
            )
        elif (
            spatial_norm == "norm1mean0"
        ):  # Mean correction + L1norm (Divide by L1norm on each sample)
            X = X - np.mean(X, axis=1, keepdims=True)
            X = X / np.sum(np.abs(X), axis=1).reshape(X.shape[0], 1)
        elif (
            spatial_norm == "norm2mean0"
        ):  # Mean correction + L2norm (Divide by L2norm on each sample)
            X = X - np.mean(X, axis=1, keepdims=True)
            X = X / np.sqrt(np.sum(np.square(X), axis=1)).reshape(X.shape[0], 1)
        else:
            raise RuntimeError(
                "Not implemented spatial normalization method:", spatial_norm
            )
        return X


# Functions ##################################################################


def corrmat(x, y, var="row"):
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
    if var == "row":
        pass
    elif var == "col":
        x = x.T
        y = y.T
    else:
        raise ValueError("Unknown var parameter specified")

    nobs = x.shape[1]

    # Subtract mean(a, axis=1) from a
    def submean(a):
        return a - np.matrix(np.mean(a, axis=1)).T

    cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / np.dot(
        np.matrix(np.std(x, axis=1, ddof=1)).T, np.matrix(np.std(y, axis=1, ddof=1))
    )

    return np.array(cmat)
