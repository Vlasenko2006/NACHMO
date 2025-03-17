import numpy as np


path = "../../../NACHMO_data/dyn_OH/"
path_IC = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/IC_"
path_ = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/"



species = ["OH", "HO2", "H2O2"]



max_c = []


for i in species:
    path_c = path + i + ".npy"
    a = np.load(path_c)
    max_c = max_c + [(np.max(a))]
   # a = a/np.max(a)
    np.save(path_ +i,a[:50,:])
    np.savetxt(path_IC + i + ".txt", np.ndarray.flatten(np.asarray(a[:50,0]), order="F"))

    print(i, a.shape)


path_c = path + "max_c.npy"

np.savetxt(path_ + "max_c.txt", np.ndarray.flatten(np.asarray(max_c), order="F"))

np.save(path_ + "max_c",max_c)
#np.savetxt(layerw, np.ndarray.flatten(np.asarray(Lw), order="F"))
    
#     fn = np.ndarray.flatten(np.asarray(Lw), order="F")
