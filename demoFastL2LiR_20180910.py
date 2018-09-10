
# coding: utf-8

# In[1]:

#Module import
import fastL2LiR_v2
import numpy
import time


# In[2]:

#Preparation for simulation analysis
X=numpy.random.rand(1200,5000)# The number of samples x The number of voxels
Y=numpy.random.rand(1200,1000)# The number of samples x The number of CNN features
alpha=1.0 #Regularization parameter (coefficient for L2-norm)
numUsedInputFeatures=500 # The number of voxels after feature selection with correlation coefficients.
#Prepare a model object (like scikit-learn functions)
model = fastL2LiR_v2.fastL2LiR()


# In[3]:

#Training start
startTime=time.time()
model.fit(X,Y,alpha,numUsedInputFeatures)
print 'Time for decoder training: ' + str(time.time()-startTime) + ' seconds'

#Prediction
predicted=model.predict(X)

