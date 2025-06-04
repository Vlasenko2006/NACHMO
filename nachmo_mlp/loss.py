#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:00:28 2023.

@author: andreyvlasenko
"""

import torch


def chemistry_loss(
    y_pred,
    y_true,
    penalty_factor=0.0,
    constraint_weights=None,
    max_c=None,
    first_step=1,
):
    """
    Custom loss function for chemistry time series prediction.

    Args:
        y_pred (torch.Tensor): Predicted values, shape (batch, nspecies, nsteps).
        y_true (torch.Tensor): Ground truth values, shape (batch, nspecies, nsteps).
        penalty_factor (float): Factor for penalizing non-physical predictions (positivity).
        constraint_weights (list or None): Optional weights for constraint penalties.
        max_c (optional): Not used, for API compatibility.
        first_step (int): Index for starting step (1-based).

    Returns:
        torch.Tensor: Loss value.
    """
    # Ensure weights is a list or None
    if constraint_weights is None:
        constraint_weights = [0, 0, 0, 0]

    # Align predictions and targets to first_step
    # Convert first_step to zero-based index
    y_pred = y_pred[:, :, first_step - 1 :]
    y_true = y_true[:, :, first_step - 1 :]
    residual = y_pred - y_true

    # Initialize constraint penalties
    penalty_HCHO = torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device)
    penalty_NO = torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device)
    penalty_CO = torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device)
    penalty_ALD = torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device)

    if constraint_weights and all(constraint_weights):
        # Example constraints:
        # HCHO (index 12) should not increase
        penalty_HCHO = (
            constraint_weights[0]
            * torch.nn.functional.relu(-1 * torch.diff(y_pred[:, 12, :]))
        )
        # NO (index 17) should not increase
        penalty_NO = (
            constraint_weights[1]
            * torch.nn.functional.relu(-1 * torch.diff(y_pred[:, 17, :]))
        )
        # CO (index 0) should not decrease
        penalty_CO = (
            constraint_weights[2]
            * torch.nn.functional.relu(torch.diff(y_pred[:, 0, :]))
        )
        # ALD (index 7) should not increase
        penalty4 = (
            constraint_weights[3]
            * torch.nn.functional.relu(-1 * torch.diff(y_pred[:, 7, :]))
        )

    # Base loss: mean squared error
    mse_loss = (residual ** 2).mean()

    # Add constraint penalties (squared mean for each)
    constraint_loss = (
        (penalty_HCHO ** 2).mean()
        + (penalty_NO ** 2).mean()
        + (penalty_CO ** 2).mean()
        + (penalty_ALD ** 2).mean()
    )

    # Penalize negative values in predictions (enforce non-negativity)
    positivity_violation = -torch.clamp(y_pred, max=0.0).mean()

    total_loss = mse_loss + constraint_loss + positivity_violation * penalty_factor
    return total_loss
