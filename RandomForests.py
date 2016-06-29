# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:16:11 2016

@author: manu
"""

from sklearn.ensemble import RandomForestRegressor
import numpy as np
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

forest = RandomForestRegressor(n_estimators=25,n_jobs=-1)
forest.fit(trainX,trainY)
sqErrorForest = ((forest.predict(valX)-valY)**2).mean()

plt.scatter(valX[:,1], valY[:,3],  color='black')
plt.plot(valX[:,1], forest.predict(valX)[:,3], color='blue', linewidth=3)

print datetime.now() - startTime