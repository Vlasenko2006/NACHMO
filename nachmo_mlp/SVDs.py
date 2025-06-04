#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:03:37 2024

@author: andrey
"""

import sys
import numpy as np
import os

#sys.path.append(os.path.abspath(os.path.join('..', '')))
from utilities import load_data
from matplotlib import pyplot as plt
import torch
from chemical_constants_and_parameters import S_verwer

# SVD decomposition of the concentration time series for float 32 and float 64 precision
def SVD32_64(data32,data64, path):

    diffs64 = [v - v.mean(axis=0) for v in data64]
    z = np.concatenate(diffs64, axis=0)

    diffs32 = [v - v.mean(axis=0) for v in data32]
    z32 = np.concatenate(diffs32, axis=0)

    u64, s64, v64 = np.linalg.svd(z.T, full_matrices=False)
    u32, s32, v32 = np.linalg.svd(z32.T, full_matrices=False)

    np.save(path+"u32",u32)
    np.save(path+"u64",u64)
    np.save(path+"s32",s32)
    np.save(path+"s64",s64)
