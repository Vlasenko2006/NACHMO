import torch
from torch.utils.data import TensorDataset, DataLoader

def constants_and_parameters_dataloader(
    device,
    Smatrix,
    Ssur=False,
    penalty_weights=None,
    cost_weights=None,
    dtype=torch.float32
):
    """
    Prepares dataloaders for constants and parameters.

    Args:
        device: The device to move tensors to (e.g., 'cpu' or 'cuda').
        Smatrix: The main matrix for S.
        Ssur: The surrogate matrix or value for Ssur (default: False).
        penalty_weights: List of penalty weights (required).
        cost_weights: List of cost weights (required).
        dtype: Data type for tensors (default: torch.float32).

    Returns:
        List containing dataloaders for [epsilon_dl, S_dl, Ssur_dl, penalty_weights_dl, cost_weights_dl].
    """
    if penalty_weights is None:
        penalty_weights = [False]
    if cost_weights is None:
        raise ValueError("Specify cost weights!")

    S_tensor = torch.tensor(Smatrix, dtype=dtype, device=device)
    epsilon_tensor = torch.tensor([1e-9], dtype=dtype, device=device)
    Ssur_tensor = torch.tensor(Ssur, dtype=dtype, device=device)
    penalty_weights_tensor = torch.tensor(penalty_weights, dtype=dtype, device=device)
    cost_weights_tensor = torch.tensor(cost_weights, dtype=dtype, device=device)

    S_dl = DataLoader(TensorDataset(S_tensor))
    epsilon_dl = DataLoader(TensorDataset(epsilon_tensor))
    Ssur_dl = DataLoader(TensorDataset(Ssur_tensor))
    penalty_weights_dl = DataLoader(TensorDataset(penalty_weights_tensor))
    cost_weights_dl = DataLoader(TensorDataset(cost_weights_tensor))

    return [epsilon_dl, S_dl, Ssur_dl, penalty_weights_dl, cost_weights_dl]
