#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:20:05 2023

@author: andrey
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Local imports
from passmaker import passmaker
from drawing_spagetti import drawing_spagetti
from drawing_spagetti_absolute_error import drawing_spagetti_absolute_error
from drawing_spagetti_relative_error import drawing_spagetti_relative_error
from drawing_scatter_plot_dyn_oh import drawing_scatter_plot_dyn_oh
from drawing_histogram import drawing_histogram
from mean_val_plotter import mean_val_plotter
from group_of_scatterplots import group_of_scatterplots
from utilities import load_data
from drawing_relative_and_absolute_errors import drawing_relative_and_absolute_errors
from drawing_the_best_and_worst_results import drawing_the_best_and_worst_results
from compute_SVD import compute_SVD
from simple_plotter_2_speceis import simple_plotter_2_speceis

# Ensure base directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join('..', '')))

# ---------------------- User Configuration ---------------------- #

epoch = 2

# Specify your species to visualize, example: ['O3', 'NO2', 'CO']
species = []  # <-- Fill with your list, e.g. ['O3', 'NO2']

# Paths
pass_to_figures = "/Users/andreyvlasenko/tst/FIGS/Figs_Verwer_same/"
data_path = "/Users/andreyvlasenko/tst/data/data_for_paper/Verwer_paper/same_distribution/"

reffile = os.path.join(data_path, "reference.npy")
estfile = os.path.join(data_path, "estimates.npy")
max_c_file = os.path.join(data_path, "max_c.npy")

# ------------------- Data Loading and Preparation ------------------- #

if not species:
    raise ValueError("Please specify the list of species to visualize, e.g. species = ['O3', 'NO2'].")

# Load estimated data
aux = np.load(estfile)

if os.path.isfile(reffile):
    aux1 = np.load(reffile)
    max_c1 = np.load(max_c_file)
else:
    print("Reference file does not exist, loading using utilities.load_data...")
    aux1, max_c1 = load_data(data_path, dtype=torch.float32, species=species)

# Downsample aux data for visualization if needed
aux = np.asarray(aux[:, :-1:10, :])
aux1 = np.asarray(aux1)
scheme = "Verwer"

# Axis/label settings
xticks = ['0(0)', '1(60)', '2(12)']
xticks2M = ['0(0)', '1(4.5K)', '2(9K)']
xlabel = "hours(steps)"
fx, fy = 5, 4
ylabel = "ppb"

max_c = np.load(max_c_file)

ncells, timesteps0, nconc = aux.shape
_, timesteps1, _ = aux1.shape
timesteps = min(timesteps0, timesteps1)
timesteps = (timesteps // 12) * 2

# Scale concentrations by maximum values
for i in range(len(species)):
    aux[:, :, i] *= max_c[i]
    aux1[:, :, i] *= max_c1[i]

# Choose skip step for plotting
skip = max(1, timesteps // 400)

c_est = aux[:, 0:timesteps:skip, :]
c_ref = aux1[:, 0:timesteps:skip, :]

abs_err = c_est - c_ref
rel_err = np.zeros_like(abs_err)

ncells, timesteps, nconc = abs_err.shape
c_est = np.nan_to_num(c_est, nan=0.0)

# Calculate relative error (in percent)
for i in range(ncells):
    for j in range(timesteps):
        for k in range(nconc):
            rel_err[i, j, k] = 100 * abs_err[i, j, k] / (c_ref[i, j, k] + 0.001 * max_c[k])

# ---------------------- Plotting and Visualization ---------------------- #

passmaker(passf=pass_to_figures)  # Ensure figure directory exists

# Concentration plots
passF = os.path.join(pass_to_figures, 'conc')
drawing_spagetti(
    species, c_est, c_ref, passf=passF,
    xticks=xticks, xlabel=xlabel, ylabel="Concentrations (ppb)"
)

# Absolute error plots
passF = os.path.join(pass_to_figures, 'error')
drawing_spagetti_absolute_error(
    species, abs_err, passf=passF,
    xticks=xticks, xlabel=xlabel, ylabel="Error in ppb"
)

# Scatter plots (flattened)
passF = os.path.join(pass_to_figures, 'scatter_plots')
C_ref = c_ref.reshape(-1, 1, len(species))
C_est = c_est.reshape(-1, 1, len(species))

passF_group = os.path.join(pass_to_figures, 'Group_of_scatterplots')

# SVD and eigenvalue plot
U, S, V, RU, RS, RV = compute_SVD(c_est, c_ref)
simple_plotter_2_speceis(
    np.squeeze(RS[:, :]), np.squeeze(S[:, :]),
    xticks=['0', str(int(len(S) / 2)), str(int(len(S)))],
    xlabel="", ylabel="", title="Eigenvalues of concentrations",
    species=species, fx=fx, fy=fy
)

simple_plotter_2_speceis( 
   c_ref[l,...], c_est[l,...], 
   xticks = Xticks, xlabel = "Time" , 
   ylabel = "",
   title = "Concentrations", 
   species = species, 
   fx = fx, fy = fy 
)






