
import torch
from models import no_grad_stepper
import timeit
import numpy as np



def MSE_counter_slice_fun(
                    stepper, 
                    step_slices,
                    data,
                    rollout_length = 100,
                    random_starts = False,    
                    strategy = "ddp",
                    top_device = "cpu",
                    accelerator = "cpu", #"gpu"
                    num_of_nodes = 1,
                    devices = 1
                    ):



    #MSE = torch.zeros((rollout_length, n_species), dtype=data.dtype) #, device=device)
    MSE = []
    MSE_std = []
    MSE_min= []
    MSE_max= []

    if top_device == "cuda":
        stepper = stepper.to('cuda')
        stepper.QPL = stepper.QPL.to("cuda")
        y0 = data.to("cuda")

    for t_start in step_slices:
        print("running rollout number: ",t_start )

        targets = data[:, t_start:t_start + rollout_length, :]

        # compute rollouts
        if top_device == "cuda":

            start = timeit.default_timer()  
            pred = no_grad_stepper(stepper, y0[:, t_start:t_start + 1], top_device, n_timepts=rollout_length).to("cpu")
            end = timeit.default_timer()
            print("Time = ", end - start) 
        
        else:
            pred = no_grad_stepper(stepper, data[:, t_start:t_start + 1], top_device, n_timepts=rollout_length)

        # compute MSE with average over simulations. these are of size (rollout_length, n_species)
        MSE += [((pred[:,-1,:] - targets[:,-1,:]) ** 2).detach().mean()] # -1 ensures that we compute a slice #/ tries
        MSE_std += [((pred[:,-1,:] - targets[:,-1,:]) ** 2).detach().mean(-1)] # -1 ensures that we compute a slice #/ tries
        MSE_min += [((pred[:,-1,:] - targets[:,-1,:]) ** 2).detach().mean(-1).min()] # -1 ensures that we compute a slice #/ tries
        MSE_max += [((pred[:,-1,:] - targets[:,-1,:]) ** 2).detach().mean(-1).max()] # -1 ensures that we compute a slice #/ tries


    rmse = np.sqrt(np.asarray(MSE))
    rmse_min = np.sqrt(np.asarray(MSE_min))
    rmse_max = np.sqrt(np.asarray(MSE_max))
    rmse_std = np.std(np.sqrt(np.asarray(MSE_std)))

    return rmse,rmse_min,rmse_max,rmse_std, len(MSE_std)



