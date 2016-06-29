# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:44:51 2016

@author: manu
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from datetime import datetime

startTime = datetime.now()

fileTrain = open("fingerDataTrain.dat",'r')
fileVal = open("fingerDataVal.dat",'r')
trainingSet = np.loadtxt(fileTrain)
valSet = np.loadtxt(fileVal)
fileTrain.close()
fileVal.close()

trainX = trainingSet[:,:13]
trainY = trainingSet[:,14:]
valX = valSet[:,:13]
valY = valSet[:,14:]

for i in range(trainX.shape[1]):
    m = trainX[:,i].mean()
    s = trainX[:,i].std()
    trainX[:,i] = (trainX[:,i]-m)/s
    valX[:,i] = (valX[:,i]-m)/s


ann = MLPRegressor()
ann.fit(trainX,trainY)
sqError = ((ann.predict(valX)-valY)**2).mean()

plt.scatter(valX[:,1], valY[:,3],  color='black')
plt.plot(valX[:,1], ann.predict(valX)[:,3], color='blue', linewidth=3)

print datetime.now() - startTime