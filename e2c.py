import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class NormalDistribution(object):
    """
    Wrapper class representing a multivariate normal distribution parameterized by
    N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
    Cov=A*(sigma).^2*A', where A = (I+v*r^T).
    """

    def __init__(self, mu, sigma, logsigma, *, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r

    @property
    def cov(self):
        """This should only be called when NormalDistribution represents one sample"""
        if self.v is not None and self.r is not None:
            assert self.v.dim() == 1
            dim = self.v.dim()
            v = self.v.unsqueeze(1)  # D * 1 vector
            rt = self.r.unsqueeze(0)  # 1 * D vector
            A = torch.eye(dim) + v.mm(rt)
            return A.mm(torch.diag(self.sigma.pow(2)).mm(A.t()))
        else:
            return torch.diag(self.sigma.pow(2))


def KLDGaussian(Q, N, eps=1e-8):
    """KL Divergence between two Gaussians
        Assuming Q ~ N(mu0, A\sigma_0A') where A = I + vr^{T}
        and      N ~ N(mu1, \sigma_1)
    """
    sum = lambda x: torch.sum(x, dim=1)
    k = float(Q.mu.size()[1])  # dimension of distribution
    mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
    s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
    a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
    b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1. + sum(v * r)))  # ratio-of-determinants term.

    #
    # print('trace: %s' % a)
    # print('mu_diff: %s' % b)
    # print('k: %s' % k)
    # print('det: %s' % c)

    return 0.5 * (a + b - k + c)


class E2C(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u, lambd=0.5,
                 config='pendulum'):
        super(E2C, self).__init__()
        enc, trans, dec = load_config(config)
        self.encoder = enc(dim_in, dim_z)

        self.decoder = dec(dim_z, dim_in)
        self.trans = trans(dim_z, dim_u)
        self.lamdb = lambd

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, Qz, u):
        return self.trans(z, Qz, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, Qz = self.reparam(mean, logvar)
        z_next, Qz_next = self.reparam(mean_next, logvar_next)

        x_dec = self.decode(z)
        x_next_dec = self.decode(z_next)

        z_next_pred, Qz_next_pred = self.transition(z, Qz, action)
        x_next_dec_pred = self.decode(z_next_pred)

        def loss():
            x_reconst_loss = (x_dec - x_next).pow(2).sum(dim=1)
            x_next_reconst_loss = (x_next_dec - x_next).pow(2).sum(dim=1)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            logvar = Qz.logsigma.exp().pow(2).log()
            KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element, dim=1).mul(-0.5)

            bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD)
            kl = KLDGaussian(Qz_next_pred, Qz_next).mul(self.lamdb)
            loss = bound_loss.add(kl)
            return loss.mean()

        return x_next_dec_pred, loss

    def latent_embeddings(self, x):
        return self.encode(x)[0]

from .e2c_configs import load_config
