from scipy.io import FortranFile
import numpy as np

# Reads binary files and prepares them fot the NN.


def binary_reader(TSTEPS, passf, name, species, times):

    for i in range(0,times): 
        file = passf+ "/"+ name + str(i) + ".bin"
        array_data = np.fromfile(file, dtype=np.float64) 
        nx = len(TSTEPS)
        ny = int(len(array_data)/nx)
        array_data = array_data.reshape([nx, ny])
        if i == 0:
           concentrations = np.zeros([times, nx, ny])
           concentrations[0,:,:] = array_data[:,:]
        else:
           concentrations[i,:,:] = array_data[:,:]

    for i in range(0,10):
        print("a = ", array_data[i,:])


    #np.save(passl,array_data[:,i+1])
    passf = passf+"_npy/"
    np.save(passf + "Time" ,concentrations[:,:,0]) 
    for i in range(0,len(species)):
        passl= passf+ species[i] 
        np.save(passl,concentrations[:,:,i+1])




