#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:57:14 2024

@author: andreyvlasenko
"""

import numpy as np
import pickle
from drawing_mean_relative_error import drawing_mean_relative_error, drawing_cost
import matplotlib as plt
from passmaker import passmaker 
from simple_plotter import simple_plotter
path = "/Users/andreyvlasenko/tst/data/data_for_paper/TBEvents/List_of_events_100"


with open(path, "rb") as fp:   # Unpickling
   Y = pickle.load(fp)


species = []
y = []
for i in range(0,len(Y)-2):
    species = species + [Y[i][0]]
    y = y + [Y[i][1]]


y = np.transpose(np.asarray(y))

epochs = len(y)

xticks = ['0',str(int(epochs/2)), str(epochs)]
title = "Evolution of the buased mean relative error over epochs "
Ylabel = "Value"
Xlabel = "Epochs"

pass_to_figures = "/Users/andreyvlasenko/tst/Figs_Verwer_same/"  
passf = pass_to_figures # creating difectory containing figures

passmaker(passf = pass_to_figures) # creating difectory containing figures


# namef = "L1"

# passF = pass_to_figures  +"Cost/"

# drawing_mean_relative_error(y, species, xticks = xticks, passf = passF, title = title , xlabel = Xlabel, ylabel = Ylabel, namef = namef, fx = 5, fy = 4 )


# train_cost = np.squeeze(np.asarray([Y[-2][1]])) 
# test_cost   = np.squeeze(np.asarray([Y[-1][1]])) 

# train_cost[0:2] = np.nan   # tblogger starts wrighting  from the third element 
# train_cost = train_cost[1:] #
# test_cost[0] = np.nan

# namef = "Costs"

# passF = pass_to_figures  +"Cost/"

# drawing_cost(train_cost, test_cost,  xticks = xticks, passf = passF, title = title , xlabel = Xlabel, ylabel = Ylabel, namef = namef, fx = 1, fy = 1 )


path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/Fast_growing_mode/Fast_growing_modes2.npy"

fgem = np.load(path)
fgem = fgem[:,0:-1:10,:]

passF = passf + "FGEM"


xticks = ['0',str(int(epochs/2)), str(epochs)]
init = 4
xlabel = "month"
xticks = ["0","6","12"]

simple_plotter(fgem[init,:8600,:], species, xticks, passf = passF, title = "Fast Growing Error Mode initiated by " + species[init] , xlabel =  xlabel, ylabel = None, namef = "FGEM_" + species[init],  fx = 5, fy = 4 )
               
               
