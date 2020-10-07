#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn import svm
trainSet= pd.read_csv("training_data.csv")
testSet= pd.read_csv("test_data.csv")


# In[ ]:





# In[139]:


trainSet.head()


# In[140]:


labelEncoder=LabelEncoder()
#categoricalFeatureMask = trainSet.dtypes==object
#oneHotEncoder = OneHotEncoder(categorical_features=categoricalFeatureMask,sparse=False)
trainSet['REGION']= labelEncoder.fit_transform(trainSet['REGION']) 

trainSet['YEAR']= labelEncoder.fit_transform(trainSet['YEAR']) 

trainSet['TRX_TYPE']= labelEncoder.fit_transform(trainSet['TRX_TYPE']) 
trainSet['IDENTITY']= labelEncoder.fit_transform(trainSet['IDENTITY']) 


# In[141]:


trainSet.dtypes


# In[142]:



# In[ ]:





# In[143]:


trainX=trainSet.iloc[:,0:6].values
trainY=trainSet.iloc[:,6].values
trainTrX,trainTeX,trainTrY,trainTeY=train_test_split(trainX,trainY,test_size=0.2,random_state=1)


# In[144]:


testSet


# In[145]:


testX=testSet.iloc[:,0:6]
testX


# In[146]:


sc=StandardScaler()
trainTrX=sc.fit_transform(trainTrX)
trainTeX=sc.fit_transform(trainTeX)


# In[148]:


print("Random Forest Regressor")
regRFR= RandomForestRegressor(n_estimators=1000,random_state=0, oob_score=True)
regRFR.fit(trainTrX,trainTrY)
yTePred=regRFR.predict(trainTeX)
yTePred
print('Mean Absolute Error:', metrics.mean_absolute_error(trainTeY, yTePred))
print('Mean Squared Error:', metrics.mean_squared_error(trainTeY, yTePred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(trainTeY, yTePred)))
#Labelladım error arttı


# In[149]:


print("Random Forest Regressor with rounding")
yTePred=np.around(yTePred, decimals=0)

print('Mean Absolute Error:', metrics.mean_absolute_error(trainTeY, yTePred))
print('Mean Squared Error:', metrics.mean_squared_error(trainTeY, yTePred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(trainTeY, yTePred)))



# In[150]:


print("Gradient Boosting Regressor")
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.99]

for learning_rate in lr_list:
    regGB = GradientBoostingRegressor(n_estimators=200, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    regGB.fit(trainTrX, trainTrY)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(trainTrX, trainTrY)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(trainTeX, trainTeY)))


# In[151]:


regGB2 = GradientBoostingRegressor(n_estimators=10000, min_samples_split=300, min_samples_leaf=300,learning_rate=0.99, max_features=2, max_depth=2, random_state=0)
regGB2.fit(trainTrX, trainTrY)
yTePred = regGB2.predict(trainTeX)
yTePred=np.around(yTePred, decimals=0)
print('Mean Absolute Error:', metrics.mean_absolute_error(trainTeY, yTePred))
print('Mean Squared Error:', metrics.mean_squared_error(trainTeY, yTePred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(trainTeY, yTePred)))


# In[152]:


print("Ada Boost Regressor")

regADA = AdaBoostRegressor( n_estimators=1000,random_state=0)
regADA.fit(trainTrX, trainTrY)
yTePred=regADA.predict(trainTeX)
yTePred=np.around(yTePred, decimals=0)
print('Mean Absolute Error:', metrics.mean_absolute_error(trainTeY, yTePred))
print('Mean Squared Error:', metrics.mean_squared_error(trainTeY, yTePred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(trainTeY, yTePred)))


# In[153]:


print("SVM Regressor")

regSVM= svm.SVR(C=0.5, epsilon=0.3,degree = 4, gamma = 'auto',shrinking = True)
regSVM.fit(trainTrX, trainTrY)
yTePred=regSVM.predict(trainTeX)
yTePred=np.around(yTePred, decimals=0)
print('Mean Absolute Error:', metrics.mean_absolute_error(trainTeY, yTePred))
print('Mean Squared Error:', metrics.mean_squared_error(trainTeY, yTePred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(trainTeY, yTePred)))


# In[154]:



testPred=regRFR.predict(testSet)


# In[155]:


df = pd.DataFrame({"TRX_COUNT":testPred}) 
df


# In[157]:


#df.to_csv("test_predictions.csv")

