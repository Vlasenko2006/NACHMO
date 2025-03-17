from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.optim as optim

from loss import loss_fun
from metrics import error_metrics
from models import no_grad_stepper

from visualization import scatter_plots_verwer
import tensorboard
import time


def train_one_epoch(model, optimizer, train_loader, valid_loader, first_loss_step=1, penalty_factor=0.0):
    model.train()
    train_loss, valid_loss, valid_batchsize = [], [], []

    for i, data in enumerate(train_loader):
        c, targets = data

        optimizer.zero_grad()

        rollout = model(c)
        loss = loss_fun(rollout, targets, penalty_factor, first_loss_step)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # check valid data loss once per epoch
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            c, targets = data

            rollout = model(c)
            loss = loss_fun(rollout, targets, penalty_factor, first_loss_step)

            valid_loss.append(loss.item())
            valid_batchsize.append(c.shape[0])

    if len(valid_loss) > 0:
        mean_valid_loss = (np.array(valid_loss) * np.array(valid_batchsize)).sum() / np.sum(valid_batchsize)
    else:
        mean_valid_loss = np.nan

    model.eval()  # FIXME need this?
    return train_loss, mean_valid_loss


class Lit_train(pl.LightningModule):
    def __init__(
        self,
        model,
        y_ref,
        parameters_loader,
        device="cpu",
        max_c=None,
        cfg=None,
        n_epochs=1,
        lr=2e-4,
        first_loss_step=1,
        penalty_factor=0.0,
        nsteps=None,
        ncells=None,
        alp=None,
        concentrations_or_tendencies=None,
        test_output_frequency=1,
    ):
        super().__init__()
       
        self.model = model
        self.stepper = model.stepper
        self.y_ref = y_ref.to(device)
        self.penalty_factor = penalty_factor
        self.nsteps = nsteps
        self.ncells = ncells
        self.first_loss_step = first_loss_step
        self.lr = lr
        self.cfg = cfg
        self.my_loss = 0.0
        self.max_c = max_c
        self.test_output_frequency = test_output_frequency
        self.my_counter = 0
        self.parameters_loader = parameters_loader
        self.species = self.cfg.data_config.species
        self.my_weights = parameters_loader[3].dataset.tensors[0]

        _, self.n_timepts, _ = self.y_ref.shape
        self.L1_norm = ["hp/BMRE averaged over " + str(self.n_timepts) + " steps for " + i for i in self.species]
        self.L1_old = np.zeros([len(max_c)])
        self.y_ref_copy = torch.Tensor.cpu(self.y_ref)
        self.max_c = max_c

    def forward(self, c) -> Any:
        return self.model(c)

    def training_step(self, batch, batch_idx):
        c, targets = batch
        rollout = self.forward(c)
        loss = loss_fun(rollout, targets, self.penalty_factor, self.my_weights, self.max_c, self.first_loss_step)
        self.my_loss = loss
        self.log("train_loss", self.my_loss, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        c, targets = batch
        rollout = self.model(c)
        loss = loss_fun(rollout, targets, self.penalty_factor, self.first_loss_step)
        self.log("test_loss", loss, on_step=True, sync_dist=True)
        self.logger.experiment


    def on_fit_start(self) -> None:
      
        if "tensorboard" in self.loggers[0].__class__.__name__.lower():
            test_params = self.cfg 
            values = np.zeros(len(self.species))

            self.loggers[0].log_hyperparams(
                test_params, dict(zip(self.L1_norm, values))
            )

    def validation_step(self, batch, batch_idx):
        c, targets = batch
        rollout = self.model(c)
        loss = loss_fun(rollout, targets, self.penalty_factor, my_weights = [False], first_step = self.first_loss_step)
        self.log("validation_loss", loss)
  

    def on_train_epoch_end(self):

        if self.current_epoch % self.test_output_frequency == 0:
           
            time_start = time.time()
            y = no_grad_stepper(self.stepper, self.y_ref[:,:,:], self.device, self.n_timepts) 


            print("Estimation time = ", time_start - time.time())
            print("y_ref.shape", self.y_ref.shape, 'y.shape', y.shape)

            y_copy = torch.Tensor.cpu(y)
          
            print("======  Saving data =======")

            print("Path to Y", self.cfg.path_to_estimates + "/Y" + str(self.current_epoch))
            np.save(self.cfg.path_to_estimates + "/Y" + str(self.current_epoch), y_copy)
            if self.current_epoch == 0:
                np.save(self.cfg.path_to_estimates + "/Y_ref", self.y_ref_copy)
                np.save(self.cfg.path_to_estimates + "/max_c", self.max_c)
            torch.save(self.model.state_dict(), self.cfg.path_to_estimates +"/model" +str(self.current_epoch) +".pt")
        

            discrete_steps, absolute_error, L1 = error_metrics(
                y, self.y_ref, steps=self.n_timepts, device=self.device, L1_old = self.L1_old
            )

            self.L1_old = L1

            tensorboard = self.logger.experiment
            scatter_plots = scatter_plots_verwer(y_copy,self.y_ref_copy,self.species, self.cfg)
            tensorboard.add_figure("Concentration scatter plots for 1/2 and full period. Output frequency = " +str(self.test_output_frequency), scatter_plots, global_step = self.current_epoch)

            print("Y_ref = ", self.y_ref_copy[1,:,0])
           

            print("TRAIN LOSS = ", self.my_loss)
            for i in range(0,len(self.species)):
                print("L1[",i,"] =", L1[i])
                self.log(self.L1_norm[i], L1[i],sync_dist=True)

            self.logger.experiment


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        return optimizer
