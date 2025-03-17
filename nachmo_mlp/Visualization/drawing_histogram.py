#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:50:48 2023

@author: andreyvlasenko
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


from passmaker import passmaker


def drawing_histogram(species, c, passf = None, xticks = None, xlabel = "concentration in ppb", ylabel = "No of cases ", title = None, namef = None):
    


    for i in range(0,len(species)):
        if not title:
            titleH = "ICs for " + species[i]  
        else:
            titleH = titleH + species[i] 
        if not namef: namefH = "histogram_for_" + species[i]    
                
        print(species[i], "shape", c.shape)
        fig, axs = plt.subplots()
        axs.hist(c[:,i], 150)
        axs.set_title(titleH, fontsize=30)
        axs.set_xlabel(xlabel, fontsize=30)
        axs.set_ylabel(ylabel, fontsize=30)
    
        if passf != None:
            passmaker(passf,namefH, fig)
