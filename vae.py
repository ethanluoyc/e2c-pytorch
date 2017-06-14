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
    def __init__(self, D_in, H, D_out):
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
    def __init__(self, D_in, H, D_out):
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
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(800, 3)
        self._enc_log_sigma = torch.nn.Linear(800, 3)

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

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(mu, var):
    logvar = var.log()
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    from torch.nn.init import xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        xavier_uniform(m.bias.data)


if __name__ == '__main__':
    pass
    #
    # input_dim = 28 * 28
    # batch_size = 32
    #
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    # mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)
    #
    # dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                          shuffle=True, num_workers=2)
    #
    # print('Number of samples: ', len(mnist))
    #
    # encoder = Encoder(input_dim, 100, 100)
    # decoder = Decoder(8, 100, input_dim)
    # vae = VAE(encoder, decoder)
    # vae.train()
    # vae.apply(weights_init)
    #
    # criterion = nn.MSELoss()
    #
    # optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    # l = None
    # for epoch in range(100):
    #     for i, data in enumerate(dataloader, 0):
    #         inputs, classes = data
    #         inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
    #         optimizer.zero_grad()
    #         dec = vae(inputs)
    #         ll = latent_loss(vae.z_mean, vae.z_sigma_sq)
    #         loss = criterion(dec, inputs) + ll
    #         loss.backward()
    #         optimizer.step()
    #         l = loss.data[0]
    #     print(epoch, l)
    #
    # plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    # plt.show(block=True)
