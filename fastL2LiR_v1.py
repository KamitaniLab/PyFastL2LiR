
# coding: utf-8

# In[289]:

import numpy as np
from numpy.matlib import repmat


# In[290]:

"""
Function to calculate correlation

This file is priginally a part of BdPy, and created by Shuntaro Aoki.
"""
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
    submean = lambda a: a - np.matrix(np.mean(a, axis=1)).T
    
    cmat = (np.dot(submean(x), submean(y).T) / (nobs - 1)) / np.dot(np.matrix(np.std(x, axis=1, ddof=1)).T, np.matrix(np.std(y, axis=1, ddof=1)))
    
    return np.array(cmat)


# In[291]:

class fastL2LiR():
    def __init__(self):
        print "fast L2 linear regression"
        
    def fit(self,X,Y,alpha,numInputFeatures):
        if False:#X.shape[1]==numInputFeatures: #No feature selection
            X=np.hstack((X,np.ones((X.shape[0],1))))
            W=np.linalg.solve(np.matmul(X.T,X)+alpha*np.eye(X.shape[1]),np.matmul(X.T,Y))
            self.W=W[0:-1,:]
            self.b=W[-1,:]
        else:#with feature selection
            self.W=np.zeros((X.shape[1],Y.shape[1]))
            self.b=np.zeros((1,Y.shape[1]))
            I=np.nonzero(np.var(X,axis = 0)<0.001)
            C=corrmat(X,Y,'col')
            C[I,:]=0.0
            X=np.hstack((X,np.ones((X.shape[0],1))))
            W0=np.matmul(X.T,X)+alpha*np.eye(X.shape[1])
            W1=np.matmul(X.T,Y)
            for index_outputDim in range(Y.shape[1]):
                C0=abs(C[:,index_outputDim])
                I=np.argsort(C0)
                I=I[::-1]
                I=I[0:numInputFeatures]
                I=np.hstack((I,X.shape[1]-1))
                W=np.linalg.solve(W0[I][:,I],W1[I][:,index_outputDim])
                for index_selectedDim in range(numInputFeatures):
                    self.W[I[index_selectedDim],index_outputDim]=W[index_selectedDim]
                self.b[0,index_outputDim]=W[-1]
                print str(index_outputDim) + '-th output dim finished'
            return self
    def predict(self,X):
        predicted = np.matmul(X,self.W)+np.matmul(np.ones((X.shape[0],1)),self.b)
        return predicted

