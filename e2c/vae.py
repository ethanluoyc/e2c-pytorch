import torch
from torch.autograd import Variable
from torch import nn
from .configs import load_config
from .losses import binary_crossentropy


class VAE(torch.nn.Module):
    def __init__(self, dim_in, dim_z, config='pendulum'):
        super(VAE, self).__init__()
        enc, trans, dec = load_config(config)
        self.encoder = enc(dim_in, dim_z)
        self.decoder = dec(dim_z, dim_in)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        self.z_mean, logvar = self.encoder(x)
        self.z_logsigma = logvar.mul(0.5)

        z = self.reparam(self.z_mean, logvar)
        x_dec = self.decoder(z)

        return x_dec

    def latent_embeddings(self, x):
        return self.encoder(x)[0]


def compute_loss(x_pred, x_true, z_mean, z_logsigma, mse=False):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if mse:
        x_reconst_loss = (x_pred - x_true).pow(2).sum(dim=1)
    else:
        x_reconst_loss = -binary_crossentropy(x_true, x_pred).sum(dim=1)
    logvar = z_logsigma.mul(2)
    KLD_element = z_mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1).mul(-0.5)
    return x_reconst_loss.mean(), KLD.mean()