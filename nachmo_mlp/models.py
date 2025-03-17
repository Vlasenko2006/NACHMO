import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from qpth.qp import QPFunction

#from chemical_constants_and_parameters import Stochiometry_matrix


class RolloutModel(nn.Module):
    """This class defines a module that iteratively applies a time stepping model."""

    def __init__(self, stepper, n_steps=1, device="cpu"):
        super(RolloutModel, self).__init__()
        self.stepper, self.n_steps, self.device = stepper, n_steps, device

    def rollout(self, x, n_steps, save_every=1):
        rollout = []
        for i in range(n_steps):
            x = self.stepper(x)

            if i % save_every == 0:
                rollout.append(x)
        return torch.dstack(rollout)

    def forward(self, x):
        return self.rollout(x, self.n_steps)


def no_grad_stepper(stepper, c_ref, device, save_every = 1, n_timepts = 1):
    if n_timepts == 1:
        _, n_timepts, _ = c_ref.shape
    model = RolloutModel(stepper, device=device)
    with torch.no_grad():
        ic = c_ref[:, 0, :]  # initial conditions for each cell, all specie
        model_output = model.rollout(ic, n_timepts - 1, save_every = 1)
        model_output = torch.movedim(model_output, 1, 2)
        c = torch.cat((c_ref[:, 0:1, :], model_output), 1)  # rollout including ic
    return c


class OptLayer(nn.Module):
    def __init__(self, Smatrix, max_c = None, device = "cpu", reg_fac=1e-6):
        super().__init__()

        self.nrates, self.nspecies = Smatrix.shape
        self.reg_fac = reg_fac
        self.device = device
    
        self.Smatrix_rank = torch.linalg.matrix_rank(Smatrix)  # fix numerical issues here?        
        Smatrix = deepcopy(Smatrix).to(dtype=torch.float64)
        self.Smatrix = Smatrix.to(device,dtype=torch.float64)

        if max_c is not None:
            self.max_c = torch.tensor(max_c).to(device, dtype = self.Smatrix.dtype)
            self.Smatrix = self.Smatrix @ torch.diag(1.0 / self.max_c)
        self.Q = 2.0 * self.Smatrix @ self.Smatrix.T
        u,s,v = torch.linalg.svd(self.Smatrix.T, full_matrices=True)

        self.null_space_dim = self.nrates - self.Smatrix_rank
        self.null_space = v[-self.null_space_dim:, :].T.to(device)
        self.Q += reg_fac * self.null_space @ self.null_space.T.to(device)


        self.A = torch.zeros([0, self.nrates], dtype=Smatrix.dtype).to(device)
        self.G = torch.cat(( -Smatrix.T ,-1 * torch.eye(self.nrates, dtype=Smatrix.dtype)), 0).to(device)
        self.b = torch.zeros(0, dtype=Smatrix.dtype).to(device)

    def forward(self, dc_in, c_in, verbose=False):  
        assert dc_in.ndim == 1
#        torch.set_printoptions(precision=9)
        dc_nn = deepcopy(dc_in).to(self.device)
        c_prev = deepcopy(c_in).to(self.device)

                      
        dc_nn   = (dc_nn   * self.max_c).double().to(self.device)
        c_prev  = (c_prev  * self.max_c).double().to(self.device)

        dc_nn = dc_nn.reshape(1, -1)
        p = (-2 * dc_nn @ self.Smatrix.T).to(self.device)    
        h = torch.cat((c_prev, torch.zeros(self.nrates, dtype=self.Smatrix.dtype).to(self.device)),0).to(self.device)
        rhat = QPFunction(verbose = verbose, maxIter=1000, eps=1e-15)(self.Q, p, self.G, h, self.A, self.b)  

        x = (rhat @ self.Smatrix / self.max_c).to(self.device)
        
        return x


class ChemicalTimeStepper(nn.Module):
    """This class wraps a NN or other model to do time stepping in a chemical system.

    S - stoichiometry matrix. Should have n_processes rows and nspecies columns (we will calculate rS, not Sr)
    """

    def __init__(
        self,
        net,
        QPL = None,
        device="cpu",
        dtype = torch.float32,
        learn_updates=False,  # whether to learn changes in concentration for one time step
        learn_rates=False,  # whether to learn reaction rates instead of concentrations or their changes
        try_Ssur_matrix = None,
        species_list=None,
        parameters_loader = None,
        rectify_outputs=True,
        verbose=True,
        apply_QP_correction = False
    ):
        super(ChemicalTimeStepper, self).__init__()
        if learn_rates:
            assert not learn_updates, "conflicting options"
            assert parameters_loader is not None, "Set Smatrix and epsilon as dataloader"
        self.device = device
        self.parameters_loader = parameters_loader
        self.Ssur = parameters_loader[2].dataset.tensors[0]
        self.Smatrix = parameters_loader[1].dataset.tensors[0]
        self.epsilon = parameters_loader[0].dataset.tensors[0]
        self.net = net
        self.learn_updates = learn_updates
        self.learn_rates = learn_rates
        self.try_Ssur_matrix = try_Ssur_matrix
        self.rectify_outputs = rectify_outputs
        self.dtype = dtype
        self.apply_QP_correction = apply_QP_correction
        self.QPL = QPL


        if species_list is not None:
            assert net.in_features == len(species_list)
            self.species_list = species_list
        else:
            self.species_list = [f"species {j}" for j in range(net.in_features)]
        if self.Smatrix is not None:
            if verbose:
                print("Size of S = ", self.Smatrix.shape)
                print("S = ", self.Smatrix)
                print("Size of epsilon = ", self.epsilon.shape)
                print("epsilon = ", self.epsilon)
            self.Smatrix = torch.transpose(self.Smatrix, 0, 1)

    def forward(self, x):

        x = x.to(self.dtype)
        y = self.net(x).to(self.Ssur.dtype)
        x = x.to(self.Ssur.dtype)

      #  c + torch.matmul(S, output_y.to(config.data_dtype)) 
        if self.learn_updates:

            if self.apply_QP_correction:
                print("Apply QP = ", self.apply_QP_correction)
                exit()

                y = torch.cat([self.QPL(dc,c) for dc ,c in zip(y,x) ])
                y = y.to(self.Ssur.dtype)
            y += x

        elif self.learn_rates:
            y = x + torch.matmul(y, self.Smatrix)
        elif self.try_Ssur_matrix:
            y = x + torch.matmul(y, self.Ssur)
        if self.rectify_outputs:
            y = F.prelu(y, self.epsilon) 
        return y


class OuterProductLayer(nn.Module):
    """Computes all pairwise products of inputs (including self-products) and appends them to the inputs."""

    def __init__(self, in_features):
        super(OuterProductLayer, self).__init__()
        self.in_features = in_features
        self.triu_indices = torch.triu_indices(in_features, in_features, offset=0)
        self.out_features = int(in_features * (in_features + 3) / 2)

    def forward(self, x):
        if x.ndim == 1:
            x = x[None, :]
        products = x.unsqueeze(1) * x.unsqueeze(2)
        products = products[:, self.triu_indices[0], self.triu_indices[1]]  # don't include both a * b and b * a
        products = products.reshape(x.shape[0], -1)
        return torch.cat((x, products), axis=1)


class MLP(nn.Module):
    """This class implements MLPs in Pytorch of an arbitrary number of hidden layers of potentially different sizes."""

    def __init__(
        self,
        in_features,
        out_features,
        n_hidden=None,
        use_bias=True,
        input_products=False,
        nonlinearity=None,
        activation_after_products=True,
        activation="ReLU",
        device="cpu",
        gate=True,
        depth = None,
        debug = None,
        NN_dtype = None,
    ):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.device = device
        self.dtype = eval(NN_dtype)

        # self.nonlinearity = torch.nn.ReLU() if nonlinearity is None else nonlinearity
        if activation == "ReLU":
            self.nonlinearity = torch.nn.ReLU() if nonlinearity is None else nonlinearity
        if activation == "PReLU":
            self.nonlinearity = torch.nn.PReLU() if nonlinearity is None else nonlinearity
        if activation == "Sigmoid":
            self.nonlinearity = torch.nn.Sigmoid() if nonlinearity is None else nonlinearity
        if n_hidden is None:
            n_hidden = []

        layers = []
        n_prev = in_features
        if input_products:
            layers.append(OuterProductLayer(in_features))
            n_prev = layers[-1].out_features

        for c, h in enumerate(n_hidden + [out_features]):
            if gate and c == len(n_hidden):
                h = h * 2
            layers.append(nn.Linear(n_prev, h, bias=use_bias, dtype=self.dtype))
            n_prev = layers[-1].out_features

        self.layers = torch.nn.ModuleList(layers)
        self.in_features, self.out_features, self.activation_after_products, self.input_prdoucts, self.gate = \
            in_features, out_features, activation_after_products, input_products, gate

    def forward(self, x):
        for c, L in enumerate(self.layers[:-1]):
            x = L(x)
            if self.activation_after_products or (not self.input_products) or c > 0:
                x = self.nonlinearity(x)
        x = self.layers[-1](x)

        if self.gate:
            if len(x.shape) == 1:
                gate_vars = x[self.out_features:]
                x = x[0 : self.out_features]
            elif len(x.shape) == 2:
                gate_vars = x[:, self.out_features:]
                x = x[:, 0 : self.out_features]
            else:
                raise ValueError
            x = x * (1 - torch.exp(-gate_vars * gate_vars))

        return x
