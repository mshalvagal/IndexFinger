# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:46:30 2016

@author: manu
"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import random

startTime = datetime.now()

sess = tf.InteractiveSession()

fileTrain = open("fingerDataTrain.dat",'r')
fileVal = open("fingerDataVal.dat",'r')
trainingSet = np.loadtxt(fileTrain)
valSet = np.loadtxt(fileVal)
fileTrain.close()
fileVal.close()

x = tf.placeholder(tf.float32, [None, 13])
W1 = tf.Variable(tf.truncated_normal([13, 8]))
b1 = tf.Variable(tf.zeros([8]))
W2 = tf.Variable(tf.truncated_normal([8, 4]))
b2 = tf.Variable(tf.zeros([4]))
y_ = tf.placeholder(tf.float32, [None, 4])

sess.run(tf.initialize_all_variables())
y = tf.matmul(tf.matmul(x, W1) + b1,W2) + b2

sq_error = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(sq_error)

for i in range(1000):
  batch = np.array(random.sample(trainingSet,100))
  train_step.run(feed_dict={x: batch[:,:13], y_: batch[:,14:]})
  
sqError = sq_error.eval(feed_dict={x: valSet[:,:13], y_: valSet[:,14:]})

print datetime.now() - startTime