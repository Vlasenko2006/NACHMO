import numpy as np
from scipy.io import FortranFile
import numpy as np
import os


# Function writes ICs, timestepping for the KPP 


def Initialize_py(C, TSTART, TEND, TSTEPS, TEMP, CFACTOR, RCONST, Random_array, CFIX):

   
    CINDEX = np.zeros(len(C), dtype=np.int32)
    RINDEX = np.zeros(len(RCONST), dtype=np.int32)    
    
    # Set nonzero ciocnentration's random initialization its corresponding indexes here
    
    for i in range(0, len(C)):
        if i == CFIX: Random_array[i] = 1.
        C[i] = C[i] * CFACTOR * Random_array[i]
        CINDEX[i] = i + 1  # Accounting difference between python and fortran indexing 
        print("C = ", C[i], "CINDEX = ", CINDEX[i])
    
    
    # Set nonzero RCONST indexes (RINDEX) here
    
    for i in range(0, len(RCONST)-1):  # Note the nonzero part of RCONST is from 1 to 3
        RINDEX[i] = i + 2 # Accounting difference between python and fortran indexing 

    print( "RINDEX = ", RINDEX    )
    
    
   
    
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

    return None
