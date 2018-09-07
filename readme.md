# Python implementation of fastL2LiR

fastL2LR is an fast implementation of ridge regression (regression with L2 normalization) that is developed for predicting neural netowrk units from fMRI data. This method is 5x faster than ordinary implementations of ridge regression, and can be used with feature selection. 

## Core functions & demo codes

* `fastL2LiR_v1.py`: Function for fastL2LR training  
* `demoFastL2LiR_20180906.py`: Demo code of fastL2LR   
* `demoFastL2LiR_20180906.ipynb`: Jupyter notebook for demo code    

## How to use

```
model = fastL2LiR_v1.fastL2LiR()
model.fit(X,Y,alpha,numUsedInputFeatures)
```
Here, 
* `X`: A matrix (# of training samples x # of voxels).  
* `Y`: A vector including label information (# of training samples x # of cnn features).  
* `alpha`: Regularization term of L2 normalization.  
*  `numUsedInputFeatures`: # of features to be selected (feature selection is based on correlation coefficient).  

## Notice 

* Do not add bias term to `X`.   
* Feature selection is included. Please do not perform it by yourself.  
* `X` and `Y` should be z-scored with mean and variance of training data. 
