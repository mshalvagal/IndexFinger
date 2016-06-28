# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:44:51 2016

@author: manu
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

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
    
def Evaluate(n):
    neigh = KNeighborsRegressor(n)
    neigh.fit(trainX,trainY)
    return ((neigh.predict(valX)-valY)**2).mean()

sqErrors2 = Parallel(n_jobs=num_cores)(delayed(Evaluate)(n) for n in range(1,11))