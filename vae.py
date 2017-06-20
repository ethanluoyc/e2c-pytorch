import torch
from torch.autograd import Variable
from torch import nn
from .e2c_configs import load_config
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

    def forward(self, x, action, x_next):
        mean, logvar = self.encoder(x)
        logsigma = logvar.mul(0.5)

        z = self.reparam(mean, logvar)
        x_dec = self.decoder(z)

        def loss():
            if False:  # TODO refactor this
                x_reconst_loss = (x_dec - x_next).pow(2).sum(dim=1)
            else:
                x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)

            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            logvar = logsigma.exp().pow(2).log()
            KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element, dim=1).mul(-0.5)
            return x_reconst_loss.mean(), KLD.mean()

        return x_dec, loss

    def latent_embeddings(self, x):
        return self.encoder(x)[0]