#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts model weights from .npy formant into txt fortran readable format

@author: andreyvlasenko
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys




path = your_path_to_weights + "net."

nlayers = # Number of saved layers 
    
for l in range(1,nlayers):
    layerw = path + "layers."  + str(l) +".weight.npy"
    Lw = np.load(layerw)
    layerw = path + "layers."  + str(l) +".weight.txt"
    np.savetxt(layerw, np.ndarray.flatten(np.asarray(Lw), order="F"))
    

    
  
    layerb = path + "layers."  + str(l) +".bias.npy"
    Lb = np.load(layerb)
    layerw = path + "layers."  + str(l) +".bias.txt"
    np.savetxt(layerw, np.ndarray.flatten(np.asarray(Lw), order="F"))


layern = path + "nonlinearity.weight.npy"
Ln = np.load(layern)
layerw = path + "nonlinearity.weight.txt"
np.savetxt(layerw, np.ndarray.flatten(np.asarray(Lw), order="F"))

