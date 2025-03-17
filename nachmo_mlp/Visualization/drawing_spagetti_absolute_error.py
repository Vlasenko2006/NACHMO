#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:20:14 2023

@author: andreyvlasenko
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from  shadded_plotter import  shadded_plotter


def drawing_spagetti_absolute_error(species, absolute_error, passf = None, xticks = None, xlabel = "default steps", ylabel = "Error in ppb ", title = None, namef = None):
    
    for i in range(0,len(species)):
        if not title:
            titleR = "Absolute error in " + species[i]  
        else:
            titleR = title + species[i] 
        if not namef: namefR = "absolute_error_" + species[i]          

        shadded_plotter(absolute_error[:,:,i], passf = passf, title = titleR, xticks = xticks, xlabel = xlabel, ylabel = ylabel , namef = namefR)

