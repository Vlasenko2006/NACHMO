#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:42:00 2023

@author: andreyvlasenko
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from passmaker import passmaker
from plotter import plotter



def drawing_spagetti_relative_error(species,relative_error, passf = None, xticks = None, xlabel = "default steps", ylabel = "Error in %" , title = None, namef = None):
    
    
    
    
    for i in range(0,len(species)):
    
        if not title:
            titleE = "Relative error in " + species[i]  
        else:
            titleE = title + species[i] 
        if not namef: namefE = "relative_error_" + species[i]    
    
        plotter(relative_error[:,:,i], passf = passf, title = titleE, xticks = xticks, xlabel = xlabel, ylabel = ylabel , namef = namefE)