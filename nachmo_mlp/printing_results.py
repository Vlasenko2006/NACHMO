#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon Feb 27 13:36:38 2023.

@author: andreyvlasenko
"""
import os
import numpy as np
import shutil
import torch
import timeit
import yaml

from torch.utils.data import random_split
from chemical_constants_and_parameters import Smatrix
from dataset import RolloutTensorDataset
from dataloader import prepare_dataloader
from models import MLP, ChemicalTimeStepper, RolloutModel, no_grad_stepper
from train import Lit_train

from utilities import load_data
import lightning.pytorch as pl

from argparse import ArgumentParser

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from omegaconf import DictConfig, OmegaConf
import hydra

from visualization import ode_plot
from visualization import metrics_surface_figs
from metrics import error_metrics


#from tensorboard.plugins.hparams import api as hp
#import tensorflow as tf


dtype = torch.float32
#loss_config, stepper_config, net_config, data_config, loader_config, train_config, experiment_config, visualization_config = default_config()
data_path = '../../concentrations_val/'              # path to the concentration data for training and testing
action = 'save'
model_dir  = 'models/'
model_name = 'NN_test'
lightning = True




@hydra.main(version_base=None, config_path=".", config_name="config")



def main(cfg: DictConfig):


	DEVICES = cfg.hardw_settings.devices
	ACCELERATOR = cfg.hardw_settings.accelerator
	STRATEGY = cfg.hardw_settings.strategy
	NUM_OF_NODES = cfg.hardw_settings.num_of_nodes

	loss_config = cfg.loss_config 
	stepper_config = cfg.stepper_config
	net_config = cfg.net_config
	data_config = cfg.data_config
	loader_config = cfg.loader_config
	train_config = cfg.train_config
	experiment_config = cfg.experiment_config
	visualization_config = cfg.visualization_config


	    

	print('trajectory_lenth = ', data_config.trajectory_length)
#	Exp = "Gate_PReLU_acc_"+ ACCELERATOR + \
#	"_dev_" + str(DEVICES) + \
#	"_strategy_"  + STRATEGY + \
#	"_Num_of_nodes_" + str(NUM_OF_NODES) + \
#	"_batch_" + str(loader_config.batch_size) + \
#	"_trajectory_length_" + str(data_config.trajectory_length) + \
#        "_num_of_layers_" + str(len(net_config.n_hidden)) + \
#        "_factor_" + str(3) + \
#        "_lr_" + str(train_config.lr) + \
#        "_activation_" + net_config.activation + \
#	"_learn_rates_" + str(stepper_config.learn_rates) + \
#	"_learn_updates_" + str(stepper_config.learn_updates) + \
#        "_Stochiometry_" + str(stepper_config.stoichiometry_matrix)
        
	Exp = "Exp_gate_8_no_positivity_prelu/" 



	cfg.depth = len(net_config.n_hidden) 
	
	cfg1 = {}
	cfg1["layers"] = {}  
	cfg1["activations"] = {}  
	for i in range(0,len(net_config.n_hidden)):
	    cfg1["layers"][str("layer " + str(i+1))]  =  net_config.n_hidden[i]  
	cfg1["activations"][str(net_config.activation)] = True

	cfg["layers"] = cfg1["layers"]


	cfg["lr"]=train_config.lr 

	if net_config.activation == "Sigmoid": cfg["activations"]["Sigmoid"] = 1
	if net_config.activation == "ReLU": cfg["activations"]["ReLU"] = 1
	if net_config.activation == "PReLU": cfg["activations"]["PReLU"] = 1



	

	in_features = len(data_config['species'])  
	out_features = 4 if stepper_config['learn_rates'] else in_features   # FIXME out_features=4

	data, max_c = load_data(data_path, dtype=dtype, species=data_config['species'])

	for i in range(len(Smatrix)):
	    Smatrix[i,:] = Smatrix[i,:]/max_c[i]

	start = timeit.default_timer()
	net = MLP(in_features, out_features, dtype=dtype, **net_config)
	PATH = "tb_gate/" + Exp +"/version_0/checkpoints/epoch=15-step=1478816.ckpt"

	print("PATH = ", PATH)
	#net.load_weights(torch.load(PATH))
	#net.load_state_dict(torch.load(PATH))


	torch_checkpoint = torch.load(PATH)
	#print(torch_checkpoint["state_dict"]['model.stepper.net.layers'] ) #.model.stepper.net.layers
	aux = torch_checkpoint["state_dict"]
	#print(aux) 
	#model = Lit_train.load_from_checkpoint(torch.load(PATH),strict=False)
	#print(model)
	stepper = ChemicalTimeStepper(net, 
                                      dtype=dtype,
                                      species_list=data_config['species'], 
                                      **stepper_config)

	c = 0
	for i in stepper.net.layers:
	    layer_weight = "model.stepper.net.layers."+str(c+0)+".weight"
	    layer_bias = "model.stepper.net.layers."+str(c+0)+".bias"
	    if c> 0:

	      #  print("i = ",  aux[layer_weight])
	        stepper.net.layers[c].weight = torch.nn.Parameter(aux[layer_weight])
	        stepper.net.layers[c].bias = torch.nn.Parameter(aux[layer_bias])
	       # print("i = ", stepper.net.layers[c].weight)
	    c+=1


	#model = RolloutModel(stepper, data_config['trajectory_length'])



	#model.eval()
	steps = 240000
	y_ref=data[:,:steps,:]
	_, n_timepts_long, _ = y_ref.shape
	y = no_grad_stepper(stepper,y_ref)
	np.save('figs/y',np.asarray(y))
	np.save('figs/y_ref',np.asarray(y_ref))
	#y_ref = y_ref[:,0::100,:]

	print("Y_ref.shape", y_ref.shape)
	print("Y.shape", y.shape)
	print("sin(1) = ", torch.sin(torch.tensor(1.)))
	print("sin = " , torch.sin(torch.tensor(3.1428/2)))
	err_grow, f2p_grow, discrete_steps, max_err_grow, absolute_error, y_outliers, L1 = error_metrics(y, y_ref, steps)  

	concentration_plots_long, scatter_plots_long, absolute_error_plots_long, relative_error_plots_long, outlier_plots_long  = ode_plot(stepper, 
	max_c, 
	y,
	y_ref,
	y_outliers, 
	n_timepts_long, 
	ncells=10)

	concentration_plots, scatter_plots, absolute_error_plots, relative_error_plots, outlier_plots  = ode_plot(stepper, 
	max_c, 
	y,
	y_ref,
	y_outliers, 
	nsteps = 20, 
	ncells=10)


	c = 0
	for i in concentration_plots:
	    i.savefig("figs/concentrations_of_"+data_config['species'][c]+".png")
	    c+=1


	c = 0
	for i in concentration_plots_long:
	    i.savefig("figs/concentrations_long_of_ "+data_config['species'][c]+".png")
	    c+=1
	c = 0
	for i in scatter_plots_long:
	    i.savefig("figs/scatter_plots"+data_config['species'][c]+".png")
	    c+=1
	c = 0
	for i in scatter_plots:
	    i.savefig("figs/scatter_plots_long"+data_config['species'][c]+".png")
	    c+=1
	for i in scatter_plots:
	    i.savefig("figs/scatter_plots_long"+data_config['species'][c]+".png")
	    c+=1


	#scatter_plots_long[0].show()
if __name__ =="__main__":
	main()


