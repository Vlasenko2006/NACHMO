#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:40:23 2023

@author: andreyvlasenko
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from passmaker import passmaker
from plotter import plotter


def drawing_spagetti(species, c_est, c_ref = None, passf = None, xticks = None, xlabel = "default steps", ylabel = 'Concentrations in ppb', title = None, namef = None ):
    
    
    
    
    for i in range(0,len(species)):
    
        if not title:
            titleE = "Estimated concentrations of " + species[i]  
        else:
            titleE = title + species[i] 
        if not namef: namefE = "estimated_concentraton_" + species[i]        
     
        plotter(c_est[:,:,i], passf = passf , title = titleE, xticks = xticks, xlabel = xlabel, ylabel = ylabel, namef = namefE)


        if not title:
            titleR = "Reference concentrations of " + species[i]  
        else:
            titleR = title + species[i] 
        if not namef: namefR = "reference_concentraton_" + species[i]  
        
        plotter(c_ref[:,:,i], passf = passf, title = titleR, xticks = xticks, xlabel = xlabel, ylabel = ylabel, namef = namefR )
