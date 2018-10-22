# PyFastL2LiR: Fast L2-regularized Linear Regression

PyFastL2LR is an fast implementation of ridge regression (regression with L2 normalization) that is developed for predicting neural netowrk units from fMRI data. This method is 5x faster than ordinary implementations of ridge regression, and can be used with feature selection.

## Core functions & demo codes

* `fastl2lir.py`: FastL2LR module
* `demoFastL2LiR_20180906.py`: Demo code of FastL2LR

## How to use

```
import fastl2lir


model = fastl2lir.FastL2LiR()
model.fit(X, Y, alpha, n_feat)
predicted = model.predict(X)
```

Here,

* `X`: A matrix (# of training samples x # of voxels).
* `Y`: A vector including label information (# of training samples x # of cnn features).
* `alpha`: Regularization term of L2 normalization.
* `n_faet`: # of features to be selected (feature selection is based on correlation coefficient).

## Notice

* Do not add bias term to `X`.
* Feature selection is included. Please do not perform it by yourself.
* `X` and `Y` should be z-scored with mean and variance of training data.
