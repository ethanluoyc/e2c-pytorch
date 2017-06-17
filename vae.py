import torch
from torch.autograd import Variable
from torch import nn


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        self.v = v
        self.r = r


class Encoder(nn.Module):
    def __init__(self, D_in, D_out):
        super(Encoder, self).__init__()
        self.m = nn.Sequential(
            torch.nn.Linear(D_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, D_out),
            nn.BatchNorm1d(D_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.m(x)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(Decoder, self).__init__()
        self.m = nn.Sequential(
            torch.nn.Linear(D_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, D_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.m(x)


class VAE(torch.nn.Module):
    def __init__(self, dim_in, dim_z):
        super(VAE, self).__init__()
        self.encoder = Encoder(dim_in, 800)
        self.decoder = Decoder(dim_z, dim_in)
        self._enc_mu = torch.nn.Linear(800, dim_z)
        self._enc_log_sigma = torch.nn.Linear(800, dim_z)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        self.z_mean = mu
        self.z_sigma = sigma
        self.z_sigma_sq = sigma.pow(2)

        eps = torch.FloatTensor(mu.size()).normal_()
        eps = Variable(eps, requires_grad=False)

        return eps.mul(sigma).add_(mu)  # Reparameterization trick

    def forward(self, state, *input):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(mu, var):
    logvar = var.log()
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1).mul_(-0.5)
    return KLD


def weights_init(m):
    from torch.nn.init import xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        xavier_uniform(m.bias.data)
