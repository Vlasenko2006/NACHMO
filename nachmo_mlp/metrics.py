import numpy as np
import torch





def error_metrics(c, c_ref, steps=1499, device="cpu", L1_old = np.zeros([1]) ):
    # Mean abs erroror
    # Standard deviation on an erroror
    # Size of the last decintile

    # histogram of an erroror
    # fit to purpose outliers

    # measure whether the erroror is growing





    n_cells, n_timepts, n_species = c_ref[:, :steps, :].shape
    scells = int(np.sqrt(n_cells))
    cells = scells * scells
    c_ref = c_ref[:cells, :steps, :]
    c = c[:cells, :steps, :]

    inc0 = int(steps / 4)


  
    discrete_steps = []

    relative_error = torch.zeros([cells, 5, n_species]).to(device)
    absolute_error = torch.zeros([cells, 5, n_species]).to(device)
    epsilon = torch.tensor(0.001)#.to(device)


    relative_error_mean = np.zeros([n_species])

    for i in range(0,n_species):
        relative_error_mean[i] = torch.mean(torch.abs(c[:,:,i] - c_ref[:,:,i]) / (c_ref[:,:, i] + epsilon))


    inc = 1
    for i in range(0, 5):
        print("i = ",i , "inc = ", inc, "c.shape = ", c.shape, "c_ref.shape", c_ref.shape)
        absolute_error[:,i,:] = c[:, inc, :] - c_ref[:, inc, :]
        relative_error[:,i,:] = absolute_error[:,i,:] / (c_ref[:, inc, :] + epsilon)
        discrete_steps = discrete_steps + [inc]
        inc = inc0 * (i + 1) - 1   #inc0 = 50K



    max_error_grow = []
    outliers = torch.zeros([2, n_timepts, n_species])
    outliers_reference = torch.zeros([2, n_timepts, n_species])
    mean = np.zeros([n_species])
    L1 = np.zeros([n_species])

    for i in range(0, n_species):
        mean[i] = torch.mean(c_ref[:, -1, i])
        for j in range(0, len(c_ref[:, -1, i])):
            if c_ref[j, -1, i] < 0.05 * mean[i]:
                relative_error[j, -1, i] = 0.0

        relative_error[:, -1, i] = torch.nan_to_num(relative_error[:, -1, i], nan=10e10)


    absolute_error_copy = torch.Tensor.cpu(absolute_error)
    absolute_error_copy = np.reshape(absolute_error_copy[:, :, :], [scells, scells, 5, n_species])
    #absolute_error_copy = torch.Tensor.cpu(absolute_error)

    return discrete_steps, absolute_error, relative_error_mean
