#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Wed Dec 21 12:08:45 2022.

@author: andrey

This code contains nonlinear perceptron emulating stiff system of chemical kinetic equations.
The reacting gases are h2o2, ho2 and oh


"""

import numpy as np
import torch
import os


def load_data(path, dtype=torch.float32, normalize=True, species=None, ntimesteps_in_data_set = None, **kwargs):
    """Load data from data files for each chemical species.

    returned data has size (n_trajectories, n_timepts, n_species)
    """
    assert ntimesteps_in_data_set != None, "Specify number of timesteps in the dataset!"

    c_all = [np.load(os.path.join(path, s) + ".npy") for s in species]  # load chemical concentrations

    max_c =[]
    for d in c_all:
        assert d.ndim == 2
        print("d.shape", d.shape)
        max_c = max_c + [d.max()]
  #  max_c = [d.max() for d in c_all]  # FIXME for some reason this stopped working. We substitute it with the loop above
    if normalize==True:
        c_out = np.stack([c / m for c, m in zip(c_all, max_c)], axis=2)
    else:
        c_out = np.stack([ci for ci in c_all], axis=2)

    c_out = torch.tensor(c_out, dtype=dtype)
    print("max_c.shape = ", len(max_c), "c_all.shape = ", len(c_all))
    return c_out, max_c



