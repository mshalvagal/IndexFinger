# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:44:51 2016

@author: manu
"""
import numpy as np
import matplotlib.pyplot as plt
#import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

fileTrain = open("fingerDataTrain.dat",'r')
fileTest = open("fingerDataTest.dat",'r')
trainingSet = np.loadtxt(fileTrain)
testSet = np.loadtxt(fileTest)
fileTrain.close()
fileTest.close()

trainX = trainingSet[:,:13]
trainY = trainingSet[:,14:]
testX = testSet[:,:13]
testY = testSet[:,14:]

regr = linear_model.LinearRegression()
regr.fit(trainX,trainY)

plt.scatter(testX[:,2], testY[:,2],  color='black')
plt.plot(testX[:,2], regr.predict(testX)[:,2], color='blue',linewidth=3)