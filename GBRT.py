# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:59:18 2016

@author: manu
"""

from sklearn import ensemble
import numpy as np
#import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime

startTime = datetime.now()

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
    
def Evaluate(i):
    gbrt = ensemble.GradientBoostingRegressor()
    gbrt.fit(trainX,trainY[:,i])
    return ((gbrt.predict(valX)-valY[:,i])**2).mean()

sqErrorsgbrt = Parallel(n_jobs=num_cores)(delayed(Evaluate)(n) for n in range(trainY.shape[1]))

#plt.scatter(valX[:,1], valY[:,3],  color='black')
#plt.plot(valX[:,1], forest.predict(valX)[:,3], color='blue', linewidth=3)

print datetime.now() - startTime