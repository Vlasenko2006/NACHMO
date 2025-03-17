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

def Initialize_py(passf):
    if not os.path.exists(passf):
        print("Creating directory " + passf)
        os.mkdir(passf)
    else:
        print("Directory " + passf + " exists")
    
    
    
    
    
    TSTART = 0. 
    TEND = 1.
    DT = int(TEND/0.01)
    TEND = TEND 
    TEMP = 298
    
    
    # ~~~ Define scale factor for units
    CFACTOR = 2.460000
    
    # ~~~ Zero C array
      
    C = np.zeros(5,dtype = np.float64)
    RCONST = np.zeros(4,dtype = np.float64) 
    CINDEX = np.zeros(len(C), dtype=np.int32)
    RINDEX = np.zeros(len(RCONST), dtype=np.int32)
    Random_array = np.random.rand(len(C))
    
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
    
    
    
    # Set nonzero ciocnentration's random initialization its corresponding indexes here
    
    for i in range(0, len(C)):
        C[i] = C[i] * CFACTOR * Random_array[i]
        CINDEX[i] = i + 1  # Accounting difference between python and fortran indexing 
        print("C = ", C[i], "CINDEX = ", CINDEX[i])
    
    
    # Set nonzero RCONST indexes (RINDEX) here
    
    for i in range(0, len(RCONST)-1):  # Note the nonzero part of RCONST is from 1 to 3
        RINDEX[i] = i + 2 # Accounting difference between python and fortran indexing 

    print( "RINDEX = ", RINDEX    )
    
    
    
    TSTEPS = np.linspace(TSTART,TEND, DT, endpoint=True)
    
    
    
    
    f = FortranFile('tsteps.nml', 'w')
    f.write_record(TSTEPS)
#    print(np.asarray([TSTEPS]))
    f.close()
    
    
    f = FortranFile('runtime_data.nml', 'w')
    f.write_record(np.asarray([TSTART,TEND,TEMP, CFACTOR], dtype = np.float64 ) )
#    print(np.asarray([TSTART,TEND,TEMP]))
    f.close()

    f = FortranFile('RCONST.nml', 'w')
    f.write_record(np.asarray([RCONST], dtype = np.float64 ) )
    f.close()
    
    f = FortranFile('RINDEX.nml', 'w')
    f.write_record(np.asarray([RINDEX], dtype = np.int32 ) )
    f.close()
    
    
    f = FortranFile('C.nml', 'w')
    f.write_record( C )
    f.close()
    
    f = FortranFile('CINDEX.nml', 'w')
    f.write_record( CINDEX )
    f.close()
    

passf = "Data"
common_file_name_of_your_results = "dynho"
KPP_run_file = "dynho.exe"
times = 1
for i in range(0,times):
    Initialize_py(passf)
    os.system("./dynho.exe")
    new_name = passf + "/" + common_file_name_of_your_results + str(i) +'.dat'
    os.replace(common_file_name_of_your_results + '.dat', new_name)



