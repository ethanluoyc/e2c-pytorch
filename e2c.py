from collections import namedtuple
import pytorch
from pytorch import nn
from pytorch.autograd import Variable
import pytorch.nn.functional as F


class NormalDistribution(object):
    """
    Wrapper class representing a multivariate normal distribution parameterized by
    N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
    Cov=A*(sigma).^2*A', where A = (I+v*r^T).
    """

    def __init__(self, sample, mu, sigma, log_sigma, v=None, r=None):
        self.sample = sample
        self.mu = mu
        self.sigma = sigma
        self.logsigma = log_sigma
        self.v = v
        self.r = r


def KLDGaussian(Q, N, eps=1e-9):
    sum = lambda x: pytorch.sum(x, dim=1)  # convenience fn for summing over features (columns)
    k = float(Q.mu.size()[1])  # dimension of distribution
    mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
    s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
    a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
    b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
    c = 2. * sum(N.logsigma - Q.logsigma)
    return 0.5 * (a + b - k + c)


class Transition(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u):
        super(Transition, self).__init__()
        self.dim_in = dim_in
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.fc_vr = nn.Linear(dim_in, dim_z * 2)
        self.fc_B = nn.Linear(dim_in, dim_z, dim_u)
        self.fc_o = nn.Linear(dim_in, dim_z)

    def forward(self, Q, u):
        h = Q.sample
        v, r = self.fc_vr(h).split(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = pytorch.eye(self.dim_z)
        A = I.add_(v1.mm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples

        u = u.unsqueeze(2)

        d = A.bmm(Q.mean).add(B.bmm(u)).add(o)
        sample = A.bmm(Q.sample).add(B.bmm(u)).add(o)
        return NormalDistribution(sample, d, Q.sigma, Q.logsigma, v, r)


class Encoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.m = nn.Sequential(
            pytorch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            pytorch.nn.Linear(800, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.m(x)


class Decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.m = nn.Sequential(
            pytorch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            pytorch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.m(x)


class E2C(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u):
        super(E2C, self).__init__()
        self.encoder = Encoder(dim_in, 800)
        self.enc_fc_normal = nn.Linear(800, dim_z * 2)

        self.decoder = Decoder(dim_z, 800)
        self.dec_fc_bernoulli = nn.Linear(800, dim_in)
        self.trans = Transition(dim_in, dim_z, dim_u)

    def encode(self, x):
        return self.enc_fc_normal(self.encoder(x)).split(2, dim=1).relu()

    def decode(self, z):
        return self.dec_fc_bernoulli(self.decoder(z))

    def transition(self, z, u):
        return self.trans(z, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = pytorch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z = self.reparam(mean, logvar)
        z_next = self.reparam(mean_next, logvar_next)

        x_dec = self.decode(z)
        x_next_dec = self.decode(z_next)

        z_next_pred = self.transition(z, action)
        x_next_dec_pred = self.decode(z_next_pred)

        def loss():
            x_reconst_loss = nn.MSELoss()(x_dec, x)
            x_next_reconst_loss = nn.MSELoss()(x_next_dec, x_next)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD_element = z.mean.pow(2).add_(z.logvar.exp()).mul_(-1).add_(1).add_(z.logvar)
            KLD = pytorch.sum(KLD_element).mul_(-0.5)

            bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD)
            kl = KLDGaussian(z_next_pred, z_next).mul(0.5)
            loss = bound_loss.add(kl)
            return loss.mean(dim=1)

        return x_next_dec_pred
