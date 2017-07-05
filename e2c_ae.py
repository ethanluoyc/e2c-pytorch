import torch
from torch import nn
from pixel2torque.pytorch.e2c import load_config
from pixel2torque.pytorch.configs import Encoder
from pixel2torque.pytorch.losses import kl_bernoulli


class _PendulumEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out)
        ])
        super(_PendulumEncoder, self).__init__(m, dim_in, dim_out)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x


class E2CAE(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u, config='pendulum'):
        super(E2CAE, self).__init__()
        _, trans, dec = load_config(config)

        self.encoder = _PendulumEncoder(dim_in, dim_z)
        self.decoder = dec(dim_z, dim_in)
        self.trans = trans(dim_z, dim_u)

    def forward(self, x, action, x_next):
        self.z = self.encoder(x)
        self.z_next = self.encoder(x_next)
        print(self.decoder)
        self.x_reconst = self.decoder(self.z)
        self.x_next_reconst = self.decoder(self.z_next)

        self.z_next_pred = self.trans(self.z)
        self.x_next_pred_dec = self.decoder(self.z_next_pred)

        return self.x_next_pred_dec


def compute_loss(x, x_next, x_reconst, x_next_reconst,
                 z, z_next, z_next_pred,
                 beta=0.05, sparsity=0.005):
    mse = nn.MSELoss()
    return mse(x_reconst, x) \
        .add(mse(x_next_reconst, x_next)) \
        .add(beta * kl_bernoulli(z, sparsity)) \
        .add(beta * kl_bernoulli(z_next_pred, sparsity)) \
        .add(beta * kl_bernoulli(z_next, sparsity))
