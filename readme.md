# PyFastL2LiR: Fast L2-regularized Linear Regression

PyFastL2LR is fast implementation of ridge regression (regression with L2 normalization) that is developed for predicting neural netowrk unit activities from fMRI data. This method is five times faster than ordinary implementations of ridge regression, and can be used with feature selection.

## Installation

```
$ pip install PyFastL2LiR
```

## Usage

```
import fastl2lir


model = fastl2lir.FastL2LiR()
model.fit(X, Y, alpha, n_feat)
Y_predicted = model.predict(X)
```

Here,

* `X`: A matrix (# of training samples x # of voxels).
* `Y`: A vector including label information (# of training samples x # of cnn features).
* `alpha`: Regularization term of L2 normalization.
* `n_feat`: # of features to be selected (feature selection is based on correlation coefficient).

See `demo.py` for more examples.

## Notice

* You don't need to add bias term in `X`; `FastL2LiR` automatically adds the bias term in the input data.
* `FastL2LiR.fit()` automatically performs feature selection. You don't need to select features by yourself.
* `X` and `Y` should be z-scored with mean and standard deviation of training data.
