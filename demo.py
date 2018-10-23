'''Demo for PyFastL2LiR'''


import numpy
import time
import pickle

import fastl2lir


# Preparation for simulation analysis
X = numpy.random.rand(1200, 5000)  # The number of samples x The number of voxels
Y = numpy.random.rand(1200, 1000)  # The number of samples x The number of CNN features

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
predicted = model.predict(X)
