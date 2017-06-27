"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, enc, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.m = enc
        self.dim_int = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        return self.m(x).chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self, dec, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.m = dec
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, z):
        return self.m(z)


class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super(Transition, self).__init__()
        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.dada.cuda()
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)


class PlaneEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 150),
            # nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            # nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            # nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, dim_out*2)
        )
        super(PlaneEncoder, self).__init__(m, dim_in, dim_out)


class PlaneDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, dim_out),
            # nn.BatchNorm1d(dim_out),
            nn.Sigmoid()
        )
        super(PlaneDecoder, self).__init__(m, dim_in, dim_out)


class PlaneTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            # nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(PlaneTransition, self).__init__(trans, dim_z, dim_u)


class PendulumEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 2 * dim_out)
        )
        super(PendulumEncoder, self).__init__(m, dim_in, dim_out)


class PendulumDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out),
            nn.Sigmoid()
        )
        super(PendulumDecoder, self).__init__(m, dim_in, dim_out)


class PendulumTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z * 2),
            nn.BatchNorm1d(dim_z * 2),
            nn.Sigmoid() # Added to prevent nan
        )
        super(PendulumTransition, self).__init__(trans, dim_z, dim_u)


_CONFIG_MAP = {
    'plane': (PlaneEncoder, PlaneTransition, PlaneDecoder),
    'pendulum': (PendulumEncoder, PendulumTransition, PendulumDecoder)
}


def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

from .e2c import NormalDistribution

__all__ = ['load_config']
