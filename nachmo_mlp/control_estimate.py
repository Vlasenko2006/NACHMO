import torch
import numpy as np
from copy import deepcopy
import timeit


def control_estimate(data, QPL,device ):
    
    c = data[0, :100, :]
    dc = np.diff(c, axis = 0)

    dc = torch.tensor(dc).to(device)
    c  = torch.tensor(c).to(device)
    
    
    dc_u = deepcopy(dc)
    c_u = deepcopy(c)
    start = timeit.default_timer()
    c_est = torch.cat([QPL(dc[t, :], c[t, :]) for t in range(dc.shape[0])])
    end = timeit.default_timer()
    print("Elapsed time = ", end - start)
    #exit()    
    return c_u.to("cpu"), c_est.to("cpu")

