# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:44:51 2016

@author: manu
"""
import numpy as np
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

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(trainX,trainY)