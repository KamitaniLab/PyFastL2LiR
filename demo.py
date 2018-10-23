'''Demo for PyFastL2LiR'''


import numpy
import time

import fastl2lir


# Simple example #############################################################

# Preparation for simulation data
X = numpy.random.rand(600, 1000)  # The number of samples x The number of voxels
Y = numpy.random.rand(600, 500)   # The number of samples x The number of CNN features

# Regularization parameter (coefficient for L2-norm)
alpha = 1.0

# The number of voxels after feature selection with correlation coefficients.
n_feat = 500

# Prepare a model object (like scikit-learn functions)
model = fastl2lir.FastL2LiR()

# Training
start_t = time.time()
model.fit(X, Y, alpha, n_feat)
print('Time for decoder training: ' + str(time.time() - start_t) + ' seconds')

# Prediction
Y_predicted = model.predict(X)


# Multi-dimensional target features ##########################################

X = numpy.random.rand(600, 1000)

# The target feature array is 3D.
Y = numpy.random.rand(600, 16, 16, 4)

# Training and prediction
model = fastl2lir.FastL2LiR()

start_t = time.time()
model.fit(X, Y, alpha, n_feat)
print('Time for decoder training: ' + str(time.time() - start_t) + ' seconds')

Y_predicted = model.predict(X)

# Check predicted Y shape
print('Predicted Y shape: %s' % (Y_predicted.shape,))


# Chunking example ###########################################################

X = numpy.random.rand(600, 1000)
Y = numpy.random.rand(600, 16, 16, 4)

chunk_size = 16

# Training and prediction
model = fastl2lir.FastL2LiR()

start_t = time.time()
model.fit(X, Y, alpha, n_feat, chunk_size=chunk_size)
print('Time for decoder training: ' + str(time.time() - start_t) + ' seconds')

Y_predicted = model.predict(X)

# Check predicted Y shape
print('Predicted Y shape: %s' % (Y_predicted.shape,))
