import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#from chemical_constants_and_parameters import Smatrix


def constants_and_parameters_dataloader(device, Smatrix, Ssur = False, penalty_weights = [False], cost_weights = [False], dtype = torch.float32 ):
    S = TensorDataset(torch.tensor(Smatrix,dtype = dtype).to(device))
    epsilon = TensorDataset(torch.tensor([0.000000001],dtype = dtype).to(device))
    Ssur = TensorDataset(torch.tensor(Ssur,dtype = dtype).to(device))

    assert cost_weights, "Specify cost weights!"

    penalty_weights = TensorDataset(torch.tensor(penalty_weights,dtype = dtype).to(device))
    cost_weights = TensorDataset(torch.tensor(cost_weights,dtype = dtype).to(device))


    #if not any(weights): 
    #   weights = TensorDataset(torch.tensor([False],dtype = dtype).to(device))
    #   print("No cost weights")
    #elif all(weights):
    #   weights = TensorDataset(torch.tensor(weights,dtype = dtype).to(device))
    #   print("Setting cost weights")

    S_dl = DataLoader(S)
    epsilon_dl = DataLoader(epsilon)
    Ssur_dl = DataLoader(Ssur)
    penalty_weights = DataLoader(penalty_weights)
    cost_weights = DataLoader(cost_weights)        

    return [epsilon_dl, S_dl, Ssur_dl, penalty_weights, cost_weights]

