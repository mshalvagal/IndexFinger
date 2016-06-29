# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:41:02 2016

@author: manu
"""

from sklearn.tree import DecisionTreeRegressor
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

tree = DecisionTreeRegressor()
tree.fit(trainX,trainY)
sqErrorTree = ((tree.predict(valX)-valY)**2).mean()

plt.scatter(valX[:,1], valY[:,3],  color='black')
plt.plot(valX[:,1], tree.predict(valX)[:,3], color='blue', linewidth=3)

print datetime.now() - startTime