'''PyFastL2LiR: Fast L2-regularized Linear Regression.'''


import math
import sys
from time import time
import warnings

import numpy as np
from numpy.matlib import repmat
from tqdm import tqdm

pv = sys.version_info
python_version = float('{}.{}'.format(pv.major, pv.minor))

if python_version >= 3.5:
    from threadpoolctl import threadpool_limits
print("python_version:", python_version)

class FastL2LiR(object):
    '''Fast L2-regularized linear regression class.'''

    def __init__(self, W=np.array([]), b=np.array([]), #S=np.array([]), 
                 verbose=False):
        self.__W = W
        self.__b = b
        #self.__S = S
        self.__verbose = verbose

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

    def fit(self, X, Y, alpha=1.0, n_feat=0, save_select_feat=False, spatial_norm=None, select_sample=None, chunk_size=0, cache_dir='./cache', dtype=np.float64):
        '''Fit the L2-regularized linear model with the given data.

        Parameters
        ----------
        X, Y : array_like
            Training inputs (data and targets).
        alpha: float
            Regularization parameter (coefficient for L2-norm).
        n_feta: int
            The number of selected input features.
        save_select_feat: bool
            unitごとに選択されたvoxelのbool matrixを保存
            unitごとにfittingを行うため，実行時メモリ量を減らすことができる（計算時間は長くなり，ストレージ必要量は多くなる点に注意）
        spatial_norm: str (norm1, norm2, std1, std1mean0, norm1mean0, norm2mean0)
            sample単位かつ選択されたvoxelでのspatial normalizationを行う
            unitごとに選択されたvoxelのindex matrixを保存する必要性があるため，これを選択すると自動的にsaveMemory modeになる
        select_sample: str (nan_remove...今後増やす)
            training sampleを選択する方法を指定する
            これも一応saveMemoryモードにしておく．
            
        Returns
        -------
        self
            Returns an instance of self.
        '''
        print("alpha:", alpha)
        print("n_feat:", n_feat)
        print("spatial_norm:", spatial_norm)
        print("select_sample:", select_sample)

        if X.dtype != dtype: X = X.astype(dtype)
        if Y.dtype != dtype: Y = Y.astype(dtype)

        # Reshape Y
        reshape_y = Y.ndim > 2

        if reshape_y:
            Y_shape = Y.shape
            Y = Y.reshape(Y.shape[0], -1, order='F')

        # Feature selection settings
        if n_feat == 0:
            n_feat = X.shape[1]

        no_feature_selection = X.shape[1] == n_feat

        if n_feat > X.shape[1]:
            warnings.warn('X has less features than n_feat (X.shape[1] < n_feat). Feature selection is not applied.')
            no_feature_selection = True

        # # Save selected voxel mode
        if not save_select_feat:
            if (spatial_norm is not None) or (select_sample is not None):
                save_select_feat = True

        # Chunking
        if chunk_size > 0:
            chunks = self.__get_chunks(range(Y.shape[1]), chunk_size)

            if self.__verbose:
                print('Num chunks: %d' % len(chunks))

            w_list = []
            b_list = []
            s_list = []
            for i, chunk in enumerate(chunks):
                start_time = time()
                if save_select_feat:
                    print("Run sample selection mode.")
                    W, b, S = self.__sub_fit_save_select_feat(X, Y[0:, chunk], alpha=alpha, n_feat=n_feat, 
                                                           spatial_norm=spatial_norm, 
                                                           use_all_features=no_feature_selection, 
                                                           select_sample=select_sample, 
                                                           dtype=dtype)
                    s_list.append(S)
                else:
                    W, b = self.__sub_fit(X, Y[0:, chunk], alpha=alpha, n_feat=n_feat, 
                                          use_all_features=no_feature_selection, 
                                          dtype=dtype)
                w_list.append(W)
                b_list.append(b)

                if self.__verbose:
                    print('Chunk %d (time: %f s)' % (i + 1, time() - start_time))

            W = np.hstack(w_list)
            b = np.hstack(b_list)
            if save_select_feat:
                S = np.hstack(s_list)
        else:
            if save_select_feat:
                print("Run sample selection mode.")
                W, b, S = self.__sub_fit_save_select_feat(X, Y, alpha=alpha, n_feat=n_feat, 
                                                       spatial_norm=spatial_norm, 
                                                       use_all_features=no_feature_selection, 
                                                       select_sample=select_sample, 
                                                       dtype=dtype)
            else:
                W, b = self.__sub_fit(X, Y, alpha=alpha, n_feat=n_feat, 
                                      use_all_features=no_feature_selection, 
                                      dtype=dtype)

        self.__W = W
        self.__b = b
        if save_select_feat:
            self.__S = S

        if reshape_y:
            Y = Y.reshape(Y_shape, order='F')
            self.__W = self.__W.reshape((self.__W.shape[0],) + Y_shape[1:], order='F')
            self.__b = self.__b.reshape((1,) + Y_shape[1:], order='F')
            if save_select_feat:
                self.__S = self.__S.reshape((self.__S.shape[0],) + Y_shape[1:], order='F')

        return self

    def predict(self, X, dtype=np.float64, save_select_feat=False, spatial_norm=None):
        '''Predict with the fitted linear model.

        Parameters
        ----------
        X : array_like

        Returns
        -------
        Y : array_like
        '''
        if X.dtype != dtype: X = X.astype(dtype)

        # Save selected voxel mode
        if not save_select_feat:
            if spatial_norm is not None:
                save_select_feat = True

        # Reshape
        reshape_y = self.__W.ndim > 2
        if reshape_y:
            Y_shape = self.__W.shape
            W = self.__W.reshape(self.__W.shape[0], -1, order='F')
            b = self.__b.reshape(self.__b.shape[0], -1, order='F')
            if save_select_feat:
                S = self.__S.reshape(self.__S.shape[0], -1, order='F')
        else:
            W = self.__W
            b = self.__b
            if save_select_feat:
                S = self.__S

        # Prediction
        if save_select_feat:
            print("Prediction under save_select_feat")
            Y = np.zeros((X.shape[0], W.shape[1]), dtype=dtype)
            for si in range(W.shape[1]): # Loop for feature
                selected_voxel = S[:, si]
                newX = X[:, selected_voxel] # extract selected features

                # Perform the sample normalization.
                newX = self.__apply_spatial_normalization(newX, spatial_norm)

                # Predict
                newW = W[:, si].reshape(-1, 1)
                newW = newW[selected_voxel, :].reshape(-1, 1) # extract selected features
                Y[:, si] = (np.matmul(newX, newW) + b[:, si]).flatten() 
        else:
            Y = np.matmul(X, W) + np.matmul(np.ones((X.shape[0], 1), dtype=dtype), b)

        if reshape_y:
            Y = Y.reshape((Y.shape[0],) + Y_shape[1:], order='F')

        return Y

    def __sub_fit(self, X, Y, alpha=0, n_feat=0, use_all_features=True, dtype=np.float64):
        if use_all_features:
            # Without feature selection
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=dtype)))
            Wb = np.linalg.solve(np.matmul(X.T, X) + alpha * np.eye(X.shape[1], dtype=dtype), np.matmul(X.T, Y))
            W = Wb[0:-1, :]
            b = Wb[-1, :][np.newaxis, :]  # Returning b as a 2D array
        else:
            # With feature selection
            W = np.zeros((Y.shape[1], X.shape[1]), dtype=dtype)
            b = np.zeros((1, Y.shape[1]), dtype=dtype)
            I = np.nonzero(np.var(X, axis=0) < 0.00000001) # ここで非ゼロのindex値を返している (not bool)
            C = corrmat(X, Y, 'col')
            C[I, :] = 0.0
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=dtype)))
            W0 = np.matmul(X.T, X) + alpha * np.eye(X.shape[1], dtype=dtype)
            W1 = np.matmul(Y.T, X)
            C = C.T

            # TODO: refactoring
            if python_version >= 3.5:
                with threadpool_limits(limits=1, user_api='blas'):
                    for index_outputDim in tqdm(range(Y.shape[1])):
                        C0 = abs(C[index_outputDim,:])
                        I = np.argsort(C0)
                        I = I[::-1]
                        I = I[0:n_feat]
                        I = np.hstack((I, X.shape[1]-1))
                        W0_sub = (W0.ravel()[(I + (I * W0.shape[1]).reshape((-1,1))).ravel()]).reshape(I.size, I.size)
                        Wb = np.linalg.solve(W0_sub, W1[index_outputDim][I].reshape(-1,1))
                        for index_selectedDim in range(n_feat):
                            W[index_outputDim, I[index_selectedDim]] = Wb[index_selectedDim]
                        b[0, index_outputDim] = Wb[-1]
                    W = W.T
            else:
                for index_outputDim in tqdm(range(Y.shape[1])):
                    C0 = abs(C[index_outputDim,:])
                    I = np.argsort(C0)
                    I = I[::-1]
                    I = I[0:n_feat]
                    I = np.hstack((I, X.shape[1]-1))
                    W0_sub = (W0.ravel()[(I + (I * W0.shape[1]).reshape((-1,1))).ravel()]).reshape(I.size, I.size)
                    Wb = np.linalg.solve(W0_sub, W1[index_outputDim][I].reshape(-1,1))
                    for index_selectedDim in range(n_feat):
                        W[index_outputDim, I[index_selectedDim]] = Wb[index_selectedDim]
                    b[0, index_outputDim] = Wb[-1]
                W = W.T

        return W, b

    def __sub_fit_spatial_norm(self, X, Y, alpha=0, n_feat=0, 
                               spatial_norm=None, 
                               use_all_features=True, 
                               dtype=np.float64):
        """
        spatial_normalization ありの fitting を実行
        選択された voxel index を S に保存する
        __sub_fit_save_select_feat から training sample selection 機能を除去したもの
        現在未使用
        """
        if use_all_features:
            # Without feature selection                                      
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=dtype)))
            Wb = np.linalg.solve(np.matmul(X.T, X) + alpha * np.eye(X.shape[1], dtype=dtype), np.matmul(X.T, Y))
            W = Wb[0:-1, :]
            b = Wb[-1, :][np.newaxis, :]  # Returning b as a 2D array        
            S = np.ones((Y.shape[1], X.shape[1] - 1), dtype=np.bool) # bias分の1列をマイナスしておく
            #　W は転置不要
            S = S.T # 転置し， <voxSize x featSize>
        else:
            # With feature selection
            W = np.zeros((Y.shape[1], X.shape[1]), dtype=dtype)
            b = np.zeros((1, Y.shape[1]), dtype=dtype)
            S = np.zeros((Y.shape[1], X.shape[1]), dtype=np.bool)
            I = np.nonzero(np.var(X, axis=0) < 0.00000001) # ここで非ゼロのindex値を返している (not bool)                                      
            C = corrmat(X, Y, 'col')
            C[I, :] = 0.0
            C = C.T

            # TODO: refactoring     
            # if python_version >= 3.5:
            with threadpool_limits(limits=1, user_api='blas'):
                for index_outputDim in tqdm(range(Y.shape[1])):                    
                    # Select voxels 
                    C0 = abs(C[index_outputDim,:])
                    I = np.argsort(C0)
                    I = I[::-1]
                    I = I[0:n_feat]	
                    newX = X[:, I]
                    S[index_outputDim, I] = True

                    # Perform the spatial normalization
                    newX = self.__apply_spatial_normalization(newX, spatial_norm)
                    
                    # Fit
                    newX = np.hstack((newX, np.ones((newX.shape[0], 1), dtype=dtype))) # 最左列にoneの列を追加                                   
                    W0 = np.matmul(newX.T, newX) + alpha * np.eye(newX.shape[1], dtype=dtype)
                    W1 = np.matmul(Y.T[index_outputDim], newX).reshape(-1,1)
                    Wb = np.linalg.solve(W0, W1)
                    for index_selectedDim in range(n_feat):
                        W[index_outputDim, I[index_selectedDim]] = Wb[index_selectedDim]
                    b[0, index_outputDim] = Wb[-1]                    
                W = W.T
                S = np.asarray(S.T, dtype=np.bool) # 転置してbool型に直しておく

        return W, b, S

    def __sub_fit_save_select_feat(self, X, Y, alpha=0, n_feat=0, 
                                spatial_norm=None, 
                                use_all_features=True, 
                                select_sample=None, 
                                dtype=np.float64):
        """
        Execute fitting for each unit
        Enables spatial normalization for selected voxels and selection of training samples.
        """
        # Prepare the matixes to save.
        W = np.zeros((Y.shape[1], X.shape[1]), dtype=dtype) # feature size x voxel size
        b = np.zeros((1, Y.shape[1]), dtype=dtype) # feautre size
        S = np.zeros((Y.shape[1], X.shape[1]), dtype=np.bool) # feature size x voxel size

        if not python_version >= 3.5:
            raise RuntimeError("Python version requires 3.5 or more.")
        
        with threadpool_limits(limits=1, user_api='blas'):
            for index_outputDim in tqdm(range(Y.shape[1])):
                # Select training samples
                if select_sample is None:
                    pass
                elif select_sample == "remove_nan": # unit が nan 値の sample を削除
                    selector = np.logical_not(np.isnan(Y[:, index_outputDim].flatten())) 
                else:
                    raise RuntimeError("Not implemented selection method:", select_sample)
                selX = X[selector, :]
                selY = Y[selector, index_outputDim].reshape(-1, 1)

                # Select voxels 
                if use_all_features:
                    I = np.arange(selX.shape[1])
                else:
                    C0 = abs(corrmat(selX, selY, 'col')).ravel()
                    I = np.argsort(C0 * -1) 
                    I = I[0:n_feat]                
                newX = selX[:, I] # sample_num x voxel_num
                S[index_outputDim, I] = True

                # Perform the spatial normalization
                newX = self.__apply_spatial_normalization(newX, spatial_norm)

                # Fit
                newX = np.hstack((newX, np.ones((newX.shape[0], 1), dtype=dtype))) # 最左列にoneの列を追加                                   
                W0 = np.matmul(newX.T, newX) + alpha * np.eye(newX.shape[1], dtype=dtype)
                W1 = np.matmul(selY.ravel(), newX).reshape(-1,1)
                Wb = np.linalg.solve(W0, W1)
                for index_selectedDim in range(n_feat):
                    W[index_outputDim, I[index_selectedDim]] = Wb[index_selectedDim]
                b[0, index_outputDim] = Wb[-1]                    
            W = W.T
            S = np.asarray(S.T, dtype=np.bool) # 転置してbool型に直しておく

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
        elif spatial_norm == "norm1": # L1norm (Divide by L1norm on each sample)
            X = X / np.sum(np.abs(X), axis=1).reshape(X.shape[0], 1)
        elif spatial_norm == "norm2": # L2norm (Divide by L2norm on each sample)
            X = X / np.sqrt(np.sum(np.square(X), axis=1)).reshape(X.shape[0], 1)
        elif spatial_norm == "std1": # Normalize with STD=1
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, ddof=1, keepdims=True) + np.mean(X, axis=1, keepdims=True)
        elif spatial_norm == "std1mean0": # Mean correction + Normalize with STD=1
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, ddof=1, keepdims=True)
        elif spatial_norm == "norm1mean0": # Mean correction + L1norm (Divide by L1norm on each sample)
            X = X - np.mean(X, axis=1, keepdims=True)
            X = X / np.sum(np.abs(X), axis=1).reshape(X.shape[0], 1) 
        elif spatial_norm == "norm2mean0": # Mean correction + L2norm (Divide by L2norm on each sample)
            X = X - np.mean(X, axis=1, keepdims=True)
            X = X / np.sqrt(np.sum(np.square(X), axis=1)).reshape(X.shape[0], 1)
        else:
            raise RuntimeError("Not implemented spatial normalization method:", spatial_norm)
        return X

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
