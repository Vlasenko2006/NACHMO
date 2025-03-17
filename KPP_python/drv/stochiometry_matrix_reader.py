"""
Created on Fri Jan 12 15:48:30 2024

@author: andreyvlasenko
"""

from scipy.io import FortranFile
import numpy as np


def stochiometry_matrix_reader(filename,passf):
    file = open(filename + "_StoichiomSP.f90", "r")
    content = file.readlines()
    file.close()
    rows = []
    colls = []
    matrix = []
    Rflag = 0
    Cflag = 0
    Mflag = 0
    for i in content:
        i = i.strip().split()
        
        for j in i:

            if j == "!": break

            if j == "IROW_STOICM": Rflag = 1
            if Rflag == 1: rows = rows + [j.replace(",", "")]
  

            if j == "ICOL_STOICM": Cflag = 1
            if Cflag == 1: colls = colls + [j.replace(",", "")]

            if j == "STOICM": Mflag = 1
            if Mflag == 1: j =j.replace("e+00_dp", "") 
            if Mflag == 1: matrix = matrix + [j.replace(",", "")]


         #   print(rows =)


            if j == '/)': 
               Rflag = 0 
               Cflag = 0
               Mflag = 0  


    IROW_STOICM = [] 
    for j in rows:
        if j != "&" and j != "/)" and j != "=" and j != "IROW_STOICM" and j != "(/":
            IROW_STOICM = IROW_STOICM + [int(j)]

    ICOL_STOICM = [] 
    for j in colls:
        if j != "&" and j != "/)" and j != "=" and j != "ICOL_STOICM" and j != "(/":
            ICOL_STOICM = ICOL_STOICM + [int(j)]


    STOICM = [] 
    for j in matrix:
        if j != "&" and j != "/)" and j != "=" and j != "STOICM" and j != "(/":
            STOICM = STOICM + [float(j)]

    IROW_STOICM = np.asarray(IROW_STOICM)
    ICOL_STOICM = np.asarray(ICOL_STOICM)
    STOICM      = np.asarray(STOICM)


    S = np.zeros([np.max(IROW_STOICM),np.max(ICOL_STOICM)])
    

    for i in range(0, len(STOICM)):
        S[IROW_STOICM[i]-1,ICOL_STOICM[i]-1] = STOICM[i]  
    
    filename = passf+ "/S"
    np.save(filename, S)


    print("S = ", S)
