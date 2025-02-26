import numpy as np
from torch import nn


class DS1(nn.Module):
    def __init__(self, units, input_dim):
        super(DS1, self).__init__()
        self.units = units
        self.input_dim = input_dim

        w_init = torch.randn(units, input_dim) * 0.05
        self.w = nn.Parameter(w_init)

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        w_expanded = self.w.unsqueeze(0)
        dist = (w_expanded - x_expanded).pow(2).sum(dim=-1)
        return dist


class DS1_activate(nn.Module):
    def __init__(self, input_dim):
        super(DS1_activate, self).__init__()
        xi_init = torch.randn(1, input_dim) * 0.05
        eta_init = torch.randn(1, input_dim) * 0.05
        self.xi = nn.Parameter(xi_init)
        self.eta = nn.Parameter(eta_init)

    def forward(self, x):
        gamma = self.eta.pow(2)
        alpha = 1.0 / (torch.exp(-self.xi) + 1.0)
        si = alpha * torch.exp(-gamma * x)

        si = si / (si.max(dim=-1, keepdim=True)[0] + 1e-4)
        return si


class DS2(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS2, self).__init__()
        beta_init = torch.randn(input_dim, num_class) * 0.05
        self.beta = nn.Parameter(beta_init)

    def forward(self, x):
        beta_sq = self.beta.pow(2)
        beta_sum = beta_sq.sum(dim=-1, keepdim=True) + 1e-8
        u = beta_sq / beta_sum
        x_exp = x.unsqueeze(-1)
        u_exp = u.unsqueeze(0)
        mass_prototype = x_exp * u_exp
        return mass_prototype


class DS2_omega(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS2_omega, self).__init__()

    def forward(self, x):
        mass_omega_sum = 1.0 - x.sum(dim=-1, keepdim=True)
        return torch.cat([x, mass_omega_sum], dim=-1)


class DS3_Dempster(nn.Module):
    def __init__(self, input_dim, num_class):
        super(DS3_Dempster, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        m1 = x[:, 0, :]
        for i in range(1, self.input_dim):
            m2 = x[:, i, :]
            omega1 = m1[:, -1].unsqueeze(-1)
            omega2 = m2[:, -1].unsqueeze(-1)
            combine1 = m1 * m2
            combine2 = m1 * omega2
            combine3 = omega1 * m2
            m_comb = combine1 + combine2 + combine3
            m_comb = m_comb / (m_comb.sum(dim=-1, keepdim=True) + 1e-8)
            m1 = m_comb
        return m1


class DS3_normalize(nn.Module):
    def forward(self, x):
        return x / (x.sum(dim=-1, keepdim=True) + 1e-8)


class DM(nn.Module):
    def __init__(self, nu, num_class):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class

    def forward(self, x):
        upper = (1.0 - self.nu) * x[:, -1].unsqueeze(-1)
        upper = upper.repeat(1, self.num_class + 1)
        out = x + upper
        return out[:, :-1]


class DSNet(nn.Module):
    def __init__(self, prototypes=200, num_class=10, n_features=128, nu=0.9):
        super(DSNet, self).__init__()

        self.ds1 = DS1(prototypes, n_features)
        self.ds1_act = DS1_activate(prototypes)
        self.ds2 = DS2(prototypes, num_class)
        self.ds2_omega = DS2_omega(prototypes, num_class)
        self.ds3_dempster = DS3_Dempster(prototypes, num_class)
        self.ds3_norm = DS3_normalize()

        self.dm = DM(nu, num_class)

    def forward(self, x):
        x = self.ds1(x)
        x = self.ds1_act(x)
        x = self.ds2(x)
        x = self.ds2_omega(x)
        x = self.ds3_dempster(x)
        x = self.ds3_norm(x)
        x = self.dm(x)
        return x


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

        utility_matrix = torch.randn(num_set, num_class)
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

        utility = None
        for i in range(self.utility_matrix.size(0)):

            precise = inputs[:, 0:self.num_class] * self.utility_matrix[i]
            precise = precise.sum(dim=-1, keepdim=True)

            omega_1 = inputs[:, -1] * self.utility_matrix[i].max()
            omega_2 = inputs[:, -1] * self.utility_matrix[i].min()
            omega = (self.nu * omega_1 + (1.0 - self.nu) * omega_2).unsqueeze(-1).float()

            utility_i = precise + omega

            if utility is None:
                utility = utility_i
            else:
                utility = torch.cat([utility, utility_i], dim=-1)

        return utility


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

        pignistic_prob = inputs + avg_pignistic

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
