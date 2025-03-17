#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:54:46 2024

@author: andreyvlasenko
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import pandas as pd

import tensorboard as tb




species = ["OH", "HO2", "H2O2"]

species = ["CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"]


path = "/Users/andreyvlasenko/tst/data/data_for_paper/TBEvents/events.out.tfevents.1705612052.g001.211343.0"


#a = pd.read_csv(path)

with open(path, 'rb') as f:
  contents = f.read()


# experiment = tb.data.experimental.ExperimentFromDev(path)
# df = experiment.get_scalars()
# df
