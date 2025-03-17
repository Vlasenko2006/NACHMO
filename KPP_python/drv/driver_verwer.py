#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:04:47 2023

@author: andreyvlasenko
"""

import numpy as np
from scipy.io import FortranFile
import numpy as np
import os
import time

from list_of_species import list_of_species
from Initialize_py import Initialize_py 
from prepare_data_directory import prepare_data_directory
from binary_reader import binary_reader
from stochiometry_matrix_reader import  stochiometry_matrix_reader
    

passf = "Data"
mechanism = "verwer"
KPP_run_file = mechanism + ".exe"
times = 40


def initv():

#    INITIALIZE YOUR RUNTIME VARIABLES HERE!    
    
  TSTART = 0. 
  TEND = 3600*20*24.
  DT = int(TEND/60)
  TEND = TEND 
  TEMP = 298

  TSTEPS = np.linspace(TSTART,TEND, DT, endpoint=True)


# ~~~ Define scale factor for units
  CFACTOR = 24600000000
    
# ~~~ Zero C array
      
  C = np.zeros(23,dtype = np.float64)
  RCONST = np.zeros(25,dtype = np.float64) 
  Random_array = np.random.rand(len(C))   # Random values for perturbing concentration ICs
    
# ~~~ Set initial species concentrations
  C[0] = 300
  C[5] = 7
  C[7] = 10
  C[12] = 100
  C[13] = 40
  C[17] = 200
  C[20] = 1.0e+6
  C[21] = 2.1e+4
  C[22] = 6.0e+3
  CFIX = 18
    
    
# constant rate coefficients
  RCONST[1] = 1.8e-20
  RCONST[2] = 8.1e-18
  RCONST[5] = 1e-17
  RCONST[7] = 1.6e-17
  RCONST[8] = 1.1e-17
  RCONST[9] = 6.1e-18
  RCONST[10] = 0.00037
  RCONST[11] = 8.1e-18
  RCONST[12] = 0.031
  RCONST[13] = 1.1e-17
  RCONST[14] = 80000
  RCONST[17] = 1.7e+06
  RCONST[18] = 7.4e+09
  RCONST[19] = 8.4e-19
  RCONST[22] = 3.1e-23
  RCONST[23] = 1.2e-18
  RCONST[24] = 0.052
# END constant rate coefficients


#  END OF INITIALIZE YOUR RUNTIME VARIABLES HERE! 

  return C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array, CFIX

def main():

  prepare_data_directory(passf)

  for i in range(0,times):
      C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array, CFIX = initv()
      print("iteration = ", i)
      print("C = ", C[:])
      start = time.time()
      Initialize_py(C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array, CFIX)
      os.system("./" + mechanism + ".exe")
      print("computation is finished")
      new_name = passf + "/" + mechanism + str(i) +'.dat'
      os.replace(mechanism + '.dat', new_name)

      new_name = passf + "/" + mechanism + str(i) +'.bin'
      os.replace(mechanism + '.bin', new_name)
      end = time.time()
      print("elapced time = ", end-start)

  species = ["CO","HNO3","SO4","XO2","O1D","SO2","O3P","ALD2","PAN","CH3O","N2O5","NO3","HCHO","O3","C2O3","HO2","NO2","NO","CH3O2","OH"] #list_of_species(mechanism + ".spc")

  binary_reader(TSTEPS, passf, mechanism, species,times)

  stochiometry_matrix_reader(mechanism, passf + "_npy/")
  print(species)



if __name__ == "__main__":
    main()

