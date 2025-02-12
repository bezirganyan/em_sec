import numpy as np
import torch

class Distance_layer(torch.nn.Module):
    def __init__(self, n_prototypes, n_feature_maps):
        super(Distance_layer, self).__init__()
        self.units = n_prototypes
        # Create a trainable weight parameter of shape (units, input_dim)
        # Using torch.randn produces values from a standard normal distribution,
        # which is equivalent to TensorFlow’s "random_normal" initializer.
        self.w = nn.Parameter(torch.randn(n_prototypes, n_feature_maps))

    def forward(self, inputs):
        # Use self.linear.weight rather than self.w
        un_mass_list = []
        for i in range(self.units):
            # Subtract the i-th prototype (shape: (input_dim,)) from inputs
            # Broadcasting makes the subtraction work on each input in the batch.
            diff = self.w[i] - inputs  # shape: (batch_size, input_dim)
            diff_sq = diff ** 2  # element-wise square
            # Sum over the last dimension (input_dim) and keep the summed dimension,
            # matching TF's keepdims=True.
            un_mass_i = diff_sq.sum(dim=-1, keepdim=True)  # shape: (batch_size, 1)
            un_mass_list.append(un_mass_i)

        # Concatenate the distances along the last dimension to get shape: (batch_size, units)
        un_mass = torch.cat(un_mass_list, dim=-1)
        return un_mass

class DistanceActivation_layer(torch.nn.Module):
    def __init__(self, n_prototypes, init_alpha=0, init_gamma=0.1):
        super(DistanceActivation_layer, self).__init__()
        # Initialize xi and eta with shape (1, input_dim) using a normal distribution.
        self.xi = nn.Parameter(torch.randn(1, n_prototypes))
        self.eta = nn.Parameter(torch.randn(1, n_prototypes))

    def forward(self, inputs):
        gamma = self.eta ** 2  # Shape: (1, input_dim)

        # Compute alpha = 1 / (1 + exp(-xi)), which is equivalent to the sigmoid of xi.
        alpha = torch.sigmoid(self.xi)  # Shape: (1, input_dim)

        # Multiply inputs by gamma (broadcasting over the batch dimension)
        si = gamma * inputs  # Shape: (batch_size, input_dim)
        si = -si
        si = torch.exp(si)
        si = si * alpha  # Element-wise multiplication with alpha

        # Normalize by dividing each element by the maximum along the last dimension (plus a small epsilon).
        max_si, _ = torch.max(si, dim=-1, keepdim=True)
        si = si / (max_si + 0.0001)
        return si


class Belief_layer(torch.nn.Module):
    def __init__(self, n_prototypes, num_class):
        super(Belief_layer, self).__init__()
        # Create a trainable parameter beta with shape (input_dim, num_class)
        self.beta = nn.Parameter(torch.randn(n_prototypes, num_class))

    def forward(self, inputs):
        beta_sq = self.beta ** 2
        beta_sum = beta_sq.sum(dim=-1, keepdim=True)
        u = beta_sq / beta_sum
        inputs_new = inputs.unsqueeze(-1)  # (batch_size, input_dim, 1)
        mass_prototype = inputs_new * u.unsqueeze(0)  # (batch_size, input_dim, num_class)
        return mass_prototype



class Omega_layer(torch.nn.Module):
    '''
    verified, give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Omega_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        # Assume inputs has shape: (batch_size, ..., last_dim)
        # 1. Sum along the last dimension while keeping the dimension.
        mass_omega_sum = inputs.sum(dim=-1, keepdim=True)  # shape: (batch_size, ..., 1)

        # 2. Subtract the summed values from 1.
        # In TF, mass_omega_sum[:,:,0] removes the last dimension. In PyTorch we can use squeeze.
        mass_omega_sum = 1.0 - mass_omega_sum.squeeze(-1)  # shape: (batch_size, ...)

        # 3. Expand dims to get back to shape (batch_size, ..., 1)
        mass_omega_sum = mass_omega_sum.unsqueeze(-1)

        # 4. Concatenate along the last dimension.
        mass_with_omega = torch.cat([inputs, mass_omega_sum], dim=-1)

        return mass_with_omega


class Dempster_layer(torch.nn.Module):
    '''
    verified give same results

    '''
    def __init__(self, n_prototypes, num_class):
        super(Dempster_layer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / (torch.sum(combine2_3, dim=-1, keepdim=True) + 1e-4)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1



class DempsterNormalize_layer(torch.nn.Module):
    '''
    verified

    '''
    def __init__(self):
        super(DempsterNormalize_layer, self).__init__()
    def forward(self, inputs):
        mass_combine_normalize = inputs / torch.sum(inputs, dim=-1, keepdim=True)
        return mass_combine_normalize


class Dempster_Shafer_module(torch.nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes):
        super(Dempster_Shafer_module, self).__init__()
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.n_feature_maps = n_feature_maps
        self.ds1 = Distance_layer(n_prototypes=self.n_prototypes, n_feature_maps=self.n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes = self.n_prototypes)
        self.ds2 = Belief_layer(n_prototypes= self.n_prototypes, num_class=self.n_classes)
        self.ds2_omega = Omega_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_dempster = Dempster_layer(n_prototypes= self.n_prototypes,num_class= self.n_classes)
        self.ds3_normalize = DempsterNormalize_layer()

    def forward(self, inputs):
        '''
        '''
        ED = self.ds1(inputs)
        ED_ac = self.ds1_activate(ED)
        mass_prototypes = self.ds2(ED_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        mass_Dempster = self.ds3_dempster(mass_prototypes_omega)
        mass_Dempster_normalize = self.ds3_normalize(mass_Dempster)
        return mass_Dempster_normalize


import torch
from torch import nn


class DM_test(nn.Module):
    """
    DM_test layer in PyTorch. Uses a non-trainable utility matrix to compute expected utilities.
    """
    def __init__(self, num_class, num_set, nu):
        super(DM_test, self).__init__()
        self.num_class = num_class
        self.nu = nu
        # In TF, trainable=False => do not update. We'll store as a buffer in PyTorch
        # so it won't be considered a learnable parameter.
        utility_matrix = torch.randn(num_set, num_class)  # random normal init
        self.register_buffer("utility_matrix", utility_matrix)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): shape (batch_size, num_class+1),
                             where inputs[:, 0:self.num_class] is the distribution across classes,
                             and inputs[:, -1] is the mass for omega.

        Returns:
            Tensor: shape (batch_size, num_set). The expected utilities for each set i.
        """
        # We'll accumulate along dimension -1 across all possible sets (num_set).
        utility = None
        for i in range(self.utility_matrix.size(0)):  # 0..(num_set-1)
            # precise part
            # inputs[:, 0:self.num_class] => shape (batch_size, num_class)
            # self.utility_matrix[i]       => shape (num_class,)
            precise = inputs[:, 0:self.num_class] * self.utility_matrix[i]
            precise = precise.sum(dim=-1, keepdim=True)

            # uncertain part
            omega_1 = inputs[:, -1] * self.utility_matrix[i].max()
            omega_2 = inputs[:, -1] * self.utility_matrix[i].min()
            omega = (self.nu * omega_1 + (1.0 - self.nu) * omega_2).unsqueeze(-1).float()

            # total utility for set i
            utility_i = precise + omega

            # Concatenate along the last dimension for each set
            if utility is None:
                utility = utility_i  # shape (batch_size, 1)
            else:
                utility = torch.cat([utility, utility_i], dim=-1)

        return utility  # shape => (batch_size, num_set)

class DM_pignistic(nn.Module):
    """
    DM_pignistic layer in PyTorch. Converts a mass distribution (with omega) into
    a pignistic probability by distributing the omega mass evenly among classes.
    """
    def __init__(self, num_class):
        super(DM_pignistic, self).__init__()
        self.num_class = num_class

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): shape (batch_size, num_class+1).
                             The last dimension is the omega mass.

        Returns:
            Tensor: shape (batch_size, num_class). The pignistic probability distribution.
        """
        avg_pignistic = inputs[:, -1] / self.num_class
        avg_pignistic = avg_pignistic.unsqueeze(-1)
        # Add evenly-distributed omega to all columns
        pignistic_prob = inputs + avg_pignistic
        # Drop the last column (the new "omega" position after addition)
        pignistic_prob = pignistic_prob[:, 0:-1]
        return pignistic_prob



def tile(a, dim, n_tile, device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index)

#
# class DM(torch.nn.Module):
#     def __init__(self, num_class, nu=0.9, device=torch.device('cpu')):
#         super(DM, self).__init__()
#         self.nu = nu
#         self.num_class = num_class
#
#     def forward(self, inputs):
#         # Assume inputs has shape: (batch_size, num_class+1)
#         # 1. Compute (1 - nu) * last_column
#         upper = (1 - self.nu) * inputs[:, -1]  # shape: (batch_size,)
#         # 2. Expand dims to shape: (batch_size, 1)
#         upper = upper.unsqueeze(-1)
#         # 3. Tile (repeat) upper to shape: (batch_size, num_class+1)
#         upper = upper.repeat(1, self.num_class + 1)
#         # 4. Add the tiled tensor to inputs elementwise
#         outputs = inputs + upper
#         # 5. Remove the last column (columns 0 to -2) to match TF slicing: [:, 0:-1]
#         outputs = outputs[:, :-1]
#         return outputs


class DM(nn.Module):
    def __init__(self, num_class, nu, **kwargs):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class

    def forward(self, inputs):
        # Assume inputs has shape: (batch_size, num_class+1)
        # 1. Compute (1 - nu) * last_column
        upper = (1 - self.nu) * inputs[:, -1]  # shape: (batch_size,)
        # 2. Expand dims to shape: (batch_size, 1)
        upper = upper.unsqueeze(-1)
        # 3. Tile (repeat) upper to shape: (batch_size, num_class+1)
        upper = upper.repeat(1, self.num_class + 1)
        # 4. Add the tiled tensor to inputs elementwise
        outputs = inputs + upper
        # 5. Remove the last column (columns 0 to -2) to match TF slicing: [:, 0:-1]
        outputs = outputs[:, :-1]
        return outputs