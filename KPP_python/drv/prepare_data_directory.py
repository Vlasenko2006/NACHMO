
import numpy as np
from scipy.io import FortranFile
import numpy as np
import os




def prepare_data_directory(passf):
    if not os.path.exists(passf):
        print("Creating directory " + passf)
        os.mkdir(passf)
    else:
        print("Directory " + passf + " exists")

    passf = passf+"_npy"
    
    if not os.path.exists(passf):
        print("Creating directory " + passf)
        os.mkdir(passf)
    else:
        print("Directory " + passf + " exists")
