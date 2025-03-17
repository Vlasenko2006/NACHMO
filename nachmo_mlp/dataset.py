#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Mon Feb 27 13:20:04 2023.

@author: andreyvlasenko
"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset
from utilities import batched_max


class RolloutDataset(Dataset):
    def __init__(self,
                 data_files,
                 rollout_length=1,
                 use_mmap=True,
                 normalization=None,
                 init_norm=True,
                 max_timepoints=None,
                 noise = 0,
                 **kwargs
                 ):
        """
        Creates a dataset from data files, which support streamming (if use_memmap=True) and rollouts.

        The data files should be numpy files with dimensions (n_trajectories, n_timepts, ...)
        """
        super(RolloutDataset, self).__init__(**kwargs)
        assert isinstance(rollout_length, int) and rollout_length > 0
        if isinstance(data_files, str):
            data_files = [data_files]
        self.rollout_length, self.use_mmap, self.data_files, self.max_timepoints = \
            rollout_length, use_mmap, data_files, max_timepoints
        assert np.all([isinstance(f, str) for f in data_files]), "data_files must be a string or list of strings"
        assert np.all([os.path.exists(f) for f in data_files]), "one or more data files not found"
        self.data = [self.load_file(f) for f in data_files]

        # count the number of possible training data points for each file
        self.n_each_file = [d.shape[0] * (d.shape[1] - rollout_length) for d in self.data]

        self.cum_n = np.cumsum(self.n_each_file)
        self.normalization = 'max' if normalization is None else normalization
        self.norm_data = dict()
        if init_norm:
            self.init_norm()

    def load_file(self, file):
        d = np.load(file, mmap_mode='r') if self.use_mmap else np.load(file)
        return d[:, :self.max_timepoints, ...] if self.max_timepoints is not None else d

    def init_norm(self):
        batch_size = 1024
        if self.normalization == 'none':
            return
        elif self.normalization == 'max':
            self.norm_data = dict(max=batched_max(self.data))
        else:
            raise NotImplementedError

    def norm(self, x):
        if self.normalization == 'none':
            return x
        elif self.normalization == 'max':
            return x / self.norm_data['max']
        else:
            raise NotImplementedError

    def denorm(self, x):
        if self.normalization == 'none':
            return x
        elif self.normalization == 'max':
            return x * self.norm_data['max']
        else:
            raise NotImplementedError

    def check_data(self):
        s = self.data[0].shape
        assert len(s) > 2, "data must have at least 3 dimensions (sequence index, time, ...)"
        for d in self.data:
            # TODO: allow different numbers time points
            assert d.ndim == len(s) and np.all(d.shape[1:] == s[1:]), "dimensions of all sequences must match"

    def __getitem__(self, item):
        i_dataset = np.nonzero(self.cum_n > item)[0][0]
        n_prev = 0 if i_dataset == 0 else self.cum_n[i_dataset - 1]
        d = self.data[i_dataset]

        i = item - n_prev  # index into all datapoints of the current file
        points_per_seq = d.shape[1] - self.rollout_length
        i_seq = i // points_per_seq  # sequence index within current file
        t = i - i_seq * points_per_seq  # time index within current sequence

        x = d[i_seq, t]  # inputs
        y = d[i_seq, t + 1: t + self.rollout_length + 1]  # outputs

        x, y = self.norm(x), self.norm(y)

        y = np.moveaxis(y,0, -1)  # move time axis from first to last

        x, y = torch.tensor(x), torch.tensor(y)

        samples,species = x.shape
  
        if noise != 0:
            for i in range(0,species):
                x[:,i] = x[:,i] + noise * torch.rand(samples)

        return x, y

    def __len__(self):
        return np.sum(self.n_each_file)


def RolloutTensorDataset(data, trajectory_length=1, verbose=False, noise = 0):
    """Prepares a dataset consisting of input (chemical states) and outputs (chemical state sequences)"""
    if isinstance(data, torch.utils.data.Dataset):
        data = data[:]
    assert isinstance(data, torch.Tensor)
    n_cells, n_timepts, n_species = data.shape
    if verbose:
        print("We have a dataset consisting of", f"{n_timepts} time points, {n_cells} grid cells, {n_species} species")

    K = trajectory_length
    print("trajectory_length = ", trajectory_length, "type(trajectory_length) = ",trajectory_length  )
    print("type(data) = ",type(data))
    print("type(K) = ",  type(K))
    print("type(n_species) = ", n_species)
    x = data[:, :-K, :].reshape((-1, n_species))  # input concentrations (single time step)
    y = torch.dstack([data[:, j : n_timepts - K + j, :].reshape(-1, n_species) for j in range(1, K + 1)])  # outputs
    n_datapts = x.shape[0]
    assert y.shape[0] == n_datapts and y.shape[2] == K

    samples,species = x.shape
  
    if noise != 0:
        for i in range(0,species):
            x[:,i] = x[:,i] + noise * torch.rand(samples)

    return TensorDataset(x, y)
