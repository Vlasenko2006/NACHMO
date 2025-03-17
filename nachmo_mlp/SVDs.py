#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:03:37 2024

@author: andrey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:31:10 2024

@author: andreyvlasenko
"""

import sys



import numpy as np
import os

#sys.path.append(os.path.abspath(os.path.join('..', '')))
from utilities import load_data
from matplotlib import pyplot as plt
import torch
from chemical_constants_and_parameters import S_verwer


def SVD32_64(data32,data64, path):

    #diffs64 =[np.diff(v, axis=0) for v in data64]
    diffs64 = [v - v.mean(axis=0) for v in data64]
    z = np.concatenate(diffs64, axis=0)


    #diffs32 = [np.diff(v, axis=0) for v in data32]
    diffs32 = [v - v.mean(axis=0) for v in data32]

    z32 = np.concatenate(diffs32, axis=0)

    u64, s64, v64 = np.linalg.svd(z.T, full_matrices=False)
    u32, s32, v32 = np.linalg.svd(z32.T, full_matrices=False)

    np.save(path+"u32",u32)
    np.save(path+"u64",u64)
    np.save(path+"s32",s32)
    np.save(path+"s64",s64)
#    np.save(path+"v32",v32)
#    np.save(path+"v64",v64)




data_path = '/gpfs/work/vlasenko/NACHMO_data/verwer_long2/'
species = ["CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"]

data64, max_c = load_data(data_path,dtype=torch.float64, normalize=True, species=species)
data32, max_c32 = load_data(data_path,dtype=torch.float32, normalize=True, species=species)
data32 = data32[:,:84000,:]
data64 = data64[:,:84000,:]

print("data_loaded")

dlen = len(data32)

path = "/gpfs/work/vlasenko/NACHMO_prog/Smatrix3/nachmo_coding/nachmo_mlp/SVDs_smean/"

for i in range(10,11):
    print("i = ", i)
    L = int(dlen * i *0.1)
    output = path + str(i) + "_percent_"
    SVD32_64(data32[:L,:,:],data64[:L,:,:], output)











#plt.semilogy(np.arange(1,21), s64, '.-')
#plt.semilogy(np.arange(1,21), s32, '*')




#diff = S @ np.linalg.solve(S, u) - u  # should be close to zero

#plt.imshow(u)
#plt.colorbar()
#plt.ylabel('species')
#plt.xlabel('vector space basis')
