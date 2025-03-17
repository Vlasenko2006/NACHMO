import numpy as np
import torch

def Ssurrogate(data, max_c, cut, dtype, subtract_mean=True, Error_removal_projection=True, normalize = False):

    if subtract_mean:
      diffs = [v - v.mean(axis=0) for v in data]
    else:
      diffs = [np.diff(v, axis=0) for v in data]

    z = np.concatenate(diffs, axis=0)

    u, s, v = np.linalg.svd(z.T, full_matrices=False)

    if Error_removal_projection == True:
        if cut > 0: u[:,:-cut] = 0 #s[-1*cut:] = 0  # in the opposite case we check if full Ssur can reproduce chemistry 
        Ssur =  np.eye(len(s),len(s)) - u @ u.T #np.diag(s) #u @ np.diag(s)
    else:
        print("cut = ", cut)
        if cut > 0: u[:,-cut :] = 0 #s[-1*cut:] = 0  # in the opposite case we check if full Ssur can reproduce chemistry
        Ssur = u @ np.diag(s)
   
    return torch.tensor(Ssur.T, requires_grad=False, dtype = dtype)
