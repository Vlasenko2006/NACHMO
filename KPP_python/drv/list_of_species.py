"""
Created on Fri Jan 12 15:48:30 2024

@author: andreyvlasenko
"""

from scipy.io import FortranFile
import numpy as np


def list_of_species(filename):
    file = open(filename, "r")
    content = file.readlines()
    file.close()
    aux = []
    flag = 0
    for i in content:
        i = i.strip()
        if i == "#DEFVAR": flag = 1
        if i == '#DEFFIX': flag = 0    

        if flag == 1 and i != "#DEFVAR":   aux = aux + [i]


    for i in aux:
        if i == '': aux.remove('') 




    species = []

    for i in aux:
        s = ""
        for j in i:   
            if j == ' ' or j =='$' or j == '\t':
               break
            else:
               s = s+j
        species = species + [s]
    return species

