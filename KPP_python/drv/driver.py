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
mechanism = "dynho"
KPP_run_file = mechanism + ".exe"
times = 10


def initv():

#    INITIALIZE YOUR RUNTIME VARIABLES HERE!    
    
  TSTART = 0. 
  TEND = 3600*2.
  DT = int(TEND/0.01)
  TEND = TEND 
  TEMP = 298

  TSTEPS = np.linspace(TSTART,TEND, DT, endpoint=True)


# ~~~ Define scale factor for units
  CFACTOR = 24600000000
    
# ~~~ Zero C array
      
  C = np.zeros(5,dtype = np.float64)
  RCONST = np.zeros(4,dtype = np.float64) 
  Random_array = np.random.rand(len(C))   # Random values for perturbing concentration ICs
    
# ~~~ Set initial species concentrations
  C[0] = 2.0e-3
  C[1] = 2.0e-1
  C[2] = 2.0e+3
  C[3] = 6.0e+3
  C[4] = 2.1e+4
    
    
# constant rate coefficients
  RCONST[1] = 1.7e-12
  RCONST[2] = 3.1e-12
  RCONST[3] = 1.1e-10
# END constant rate coefficients


#  END OF INITIALIZE YOUR RUNTIME VARIABLES HERE! 

  return C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array

def main():

  prepare_data_directory(passf)

  for i in range(0,times):
      C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array = initv()
      print("iteration = ", i)
      print("C = ", C[:])
      start = time.time()
      Initialize_py(C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array)
      os.system("./" + mechanism + ".exe")
      print("computation is finished")
      new_name = passf + "/" + mechanism + str(i) +'.dat'
      os.replace(mechanism + '.dat', new_name)

      new_name = passf + "/" + mechanism + str(i) +'.bin'
      os.replace(mechanism + '.bin', new_name)
      end = time.time()
      print("elapced time = ", end-start)

  species = list_of_species(mechanism + ".spc")

  binary_reader(TSTEPS, passf, mechanism, species,times)

  stochiometry_matrix_reader(mechanism, passf + "_npy/")




if __name__ == "__main__":
    main()

