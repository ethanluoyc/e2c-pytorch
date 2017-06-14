import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from skimage.color import rgb2gray
from skimage.transform import resize
from torch.autograd import Variable

import torch
from torch import nn
from torch import optim
from pytorch.vae import Encoder, Decoder, VAE, latent_loss


def show(img):
    npimg = img.numpy()
    return plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


class PendulumData(torch.utils.data.Dataset):
    def __init__(self, root, split):
        if split not in ['train', 'test']:
            raise ValueError

        dir = os.path.join(root, split)
        filenames = glob.glob(os.path.join(dir, '*.png'))
        images = []

        for f in filenames:
            img = plt.imread(f)
            img[img != 1] = 0
            images.append(resize(rgb2gray(img), [48, 48], mode='constant'))

        self.images = np.array(images, dtype=np.float32)
        self.images = self.images.reshape([len(images), 48, 48, 1])

        action_filename = os.path.join(root, 'actions.txt')

        with open(action_filename) as infile:
            actions = np.array([float(l) for l in infile.readlines()])

        self.actions = actions[:len(self.images)]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.images[index]


def weights_init(m):
    from torch.nn.init import xavier_uniform
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        xavier_uniform(m.bias.data)


if __name__ == '__main__':
    torch.manual_seed(12345)
    dataset = PendulumData('data/pendulum_data/pendulum_max-speed_0.1-gravity-train-test', 'train')
    testset = PendulumData('data/pendulum_data/pendulum_max-speed_0.1-gravity-train-test', 'test')
    batch_size = 128
    loader = torch.utils.data.DataLoader(dataset, batch_size,
                                         shuffle=True, drop_last=True)

    input_dim = 48 * 48
    encoder = Encoder(input_dim, 800, 800)
    decoder = Decoder(3, 800, input_dim)
    vae = VAE(encoder, decoder)
    weights_init(vae)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(vae.parameters(), lr=0.0003, betas=(0.1, 0.1))

    l = None
    plt.figure(1)
    plt.ion()
    plt.pause(0.02)

    for epoch in range(10000):
        for i, data in enumerate(loader):
            inputs = Variable(data.resize_(batch_size, input_dim))
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs).add(ll)
            loss.backward()
            optimizer.step()
            l = loss.data[0]

        if epoch % 100 == 0:
            data = Variable(torch.from_numpy(dataset[:4]).view(4, 48 * 48))
            predicted = vae(data)
            grid = torchvision.utils.make_grid(predicted.view(4, 1, 48, 48).data, nrow=2)
            show(grid)
            plt.pause(0.02)
            print("Epoch: {}, Loss: {}".format(epoch, l))
