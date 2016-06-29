# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:21:38 2016

@author: manu
"""
import numpy as np
import sys
import random

factor = np.pi/180

R = np.array([[1.11,1.19,0.37,0.93,0.66,-0.9,-0.86],
         [0.79,0.62,0,-0.18,-0.26,-0.26,-0.28],
         [0.41,0,0,-0.07,-0.16,-0.19,-0.22],
         [-0.11,0.17,-0.61,-0.48,0.58,0.13,-0.02]])
         
Rstd = np.array([[0.11,0.07,0.14,0.21,0.21,0.13,0.16],
        [0.11,0.1,0,0.13,0.08,0.11,0.11],
        [0.14,0,0,-0.07,-0.16,-0.19,-0.22],
        [0.17,0.2,0.21,0.16,0.17,0.16,0.25]])
        
qMCPAbdAdd = np.linspace(-6.6,50.4,10)*factor
qMCP = np.linspace(-18.4,85.3,10)*factor
qPIP = np.linspace(-11.7,89.5,10)*factor
qDIP = np.linspace(-6.6,50.36,10)*factor

Q = np.meshgrid(qMCP,qPIP,qDIP,qMCPAbdAdd)
Q = np.transpose(np.reshape(Q,[4,-1]))

Fmuscle = np.linspace(0,20,10)
Fmuscle = np.meshgrid(Fmuscle,Fmuscle,Fmuscle,Fmuscle,Fmuscle,Fmuscle,Fmuscle)
Fmuscle = np.transpose(np.reshape(Fmuscle,[7,-1]))

def JacobianInv(q):
    l = np.array([43.57,24.67,19.67])
    c = [np.cos(q[0]),np.cos(q[1]+q[0]),np.cos(q[1]+q[2]+q[0])]
    s = [np.sin(q[0]),np.sin(q[1]+q[0]),np.sin(q[1]+q[2]+q[0])]
    temps = np.array([-np.dot(l,s),-np.dot(l[1:],s[1:]),-l[-1]*s[-1]])
    tempc = np.array([-np.dot(l,c),-np.dot(l[1:],c[1:]),-l[-1]*c[-1]])
    J = np.empty([4,4])
    J[0,] = np.hstack((temps*np.cos(q[-1]),tempc[0]*np.sin(q[-1])))
    J[1,] = np.hstack((temps*np.sin(q[-1]),-tempc[0]*np.cos(q[-1])))
    J[2,] = np.hstack((tempc,0))
    J[3,] = [1,1,1,0]
    return J

count = 0
fileTrain = open("fingerDataTrain.dat",'w')
fileVal = open("fingerDataVal.dat",'w')
fileTest = open("fingerDataTest.dat",'w')
for q in Q:
    L = -np.dot(q,R)
    J = JacobianInv(q)
    if np.linalg.cond(J) < 1/sys.float_info.epsilon:
        J = np.linalg.inv(J)
        count = count + 1
    else:
        continue
    fMuscle = random.sample(Fmuscle,50)
    H = np.dot(np.transpose(R),J)
    Fend = np.dot(fMuscle,H)
    np.savetxt(fileTrain,np.hstack((np.tile(L,[np.size(fMuscle[:40],axis=0),1]),fMuscle[:40],Fend[:40])),fmt = '%2.3g')
    np.savetxt(fileVal,np.hstack((np.tile(L,[np.size(fMuscle[41:45],axis=0),1]),fMuscle[41:45],Fend[41:45])),fmt = '%2.3g')
    np.savetxt(fileTest,np.hstack((np.tile(L,[np.size(fMuscle[46:],axis=0),1]),fMuscle[46:],Fend[46:])),fmt = '%2.3g')

fileTrain.close()
fileVal.close()
fileTest.close()
print count