from typing import Any
import torch
import numpy as np
from models import no_grad_stepper
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import timeit

class Lit_valid(pl.LightningModule):
    def __init__(
        self,
        model = None,
        rollout_length= 1,
        device = "cpu"
        ):

        super(Lit_valid, self).__init__()
        self.model = model
        self.rollout_length = rollout_length
   
    def forward(self, x) -> Any:
        y =  no_grad_stepper(self.model, x, self.device, n_timepts=self.rollout_length)
        return y


    def validation_step(self, batch, batch_idx):
        x = batch
        x_out = no_grad_stepper(self.model, x, self.device, n_timepts=self.rollout_length)


def MSE_counter_fun(
                    stepper, 
                    data,
                    rollout_length = 100,
                    tries= 40,
                    random_starts = False,    
                    strategy = "ddp",
                    top_device = "cpu",
                    accelerator = "cpu", #"gpu"
                    num_of_nodes = 1,
                    devices = 1
                    ):


    n_species = data.shape[2]
    total_steps = data.shape[1]
    final_point = total_steps - rollout_length

    MSE = torch.zeros((rollout_length, n_species), dtype=data.dtype) #, device=device)

    model = Lit_valid(
                          model = stepper, 
                          rollout_length = rollout_length,
                          device=top_device
                          )


    validator = pl.Trainer(
            strategy=strategy,
            devices=devices,
            accelerator=accelerator,
            num_nodes=num_of_nodes,
            max_epochs=1
        )



    if top_device == "cuda":
        stepper = stepper.to('cuda')
        stepper.QPL = stepper.QPL.to("cuda")
        y0 = data.to("cuda")

    for n_try in range(0,tries):
        print("running rollout number: ",n_try)
        if random_starts:
            t_start = 0 if n_try == 0 else random.randint(0, final_point)
        else:
            t_start = n_try

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
        MSE += ((pred - targets) ** 2).mean(axis=0) / tries


    rmse = torch.sqrt(MSE).detach().numpy()

    return rmse



