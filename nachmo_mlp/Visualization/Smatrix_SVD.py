#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:50:10 2024

@author: andreyvlasenko
"""



import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import numpy as np
import torch
import os
import sys
import copy



S_verwer = torch.tensor(
       [[ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  1.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0., -1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  1., -1.,  1.],
       [ 0.,  0.,  0., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  1.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  2.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,
         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  1.,  1.,  0.,
        -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  1.],
       [ 1., -1., -1.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,
        -1.,  0.,  0.,  0.,  2.,  0., -1.,  0.,  0.,  0.,  0.,  0.]],dtype=torch.float32)




path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/same_distribution/"

max_c = np.load(path+"max_c.npy")

[u,s,v]    = np.linalg.svd(S_verwer) 

Smatrix = copy.copy(S_verwer)
for i in range(0,20):
        Smatrix[i, :] = Smatrix[i, :] / max_c[i]



[um,sm,vm]    = np.linalg.svd(Smatrix) 


fig, ax = plt.subplots()
cs = ax.pcolormesh(u) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Left vectors of Smatrix's SVD decomposition for Verwer mechanism",  fontsize=22)
cbar = fig.colorbar(cs)

plt.show()


fig, ax = plt.subplots()
cs = ax.pcolormesh(v) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Right vectors of Smatrix's SVD decomposition for Verwer mechanism",  fontsize=22)
cbar = fig.colorbar(cs)

plt.show()


fig, ax = plt.subplots()
cs = ax.plot(s) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Eigenvalues",  fontsize=22)

plt.show()



fig, ax = plt.subplots()
cs = ax.pcolormesh(um) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Left vectors of the !!!Normalized!!!  Smatrix's SVD decomposition for Verwer mechanism",  fontsize=22)
cbar = fig.colorbar(cs)

plt.show()


fig, ax = plt.subplots()
cs = ax.pcolormesh(vm) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Right vectors of the !!!Normalized!!! Smatrix's SVD decomposition for Verwer mechanism",  fontsize=22)
cbar = fig.colorbar(cs)

plt.show()


fig, ax = plt.subplots()
cs = ax.plot(sm) # linear mapping
#cs = ax.contourf(X, Y, Z, 50, locator=ticker.LogLocator(), cmap=cm.get_cmap('jet')) # log mapping
plt.title("Eigenvalues of the !!!Normalized!!! Smatrix",  fontsize=22)

plt.show()






