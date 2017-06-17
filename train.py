import torch
from torch import optim
import torchvision
from torch.autograd import Variable
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pixel2torque.pytorch.vae import VAE, latent_loss
from pixel2torque.pytorch.e2c import E2C
from .datasets import PendulumData
from torch.utils.data import Dataset, DataLoader

img_width = 48


def show_and_save(img, path):
    npimg = img.numpy()
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))
    return plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def weights_init(m):
    from torch.nn.init import xavier_uniform
    if isinstance(m, torch.nn.Linear):
        xavier_uniform(m.weight.data)


def compute_loss(reconst, actual, mean, var):
    ll = latent_loss(mean, var)
    reconst_loss = torch.sum(reconst.sub(actual).pow(2), dim=1)
    loss = reconst_loss.add(ll).mean()
    return loss


def parse_args(argv):
    pass


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)

    width = img_width
    height = width
    model_dir = 'checkpoints'

    dataset = PendulumData(
        'data/pendulum_data/pendulum_max-speed_0.1-gravity-train-test', 'all')

    trainset = dataset
    testset = dataset

    # trainset = PendulumData('bitbucket/data/processed/pendulum_max-speed_0.1-gravity-train-test', 'train')
    # testset = PendulumData('bitbucket/data/processed/pendulum_max-speed_0.1-gravity-train-test', 'test')

    batch_size = len(trainset)
    dim_u = 1
    print('batch_size %d' % batch_size)

    loader = DataLoader(trainset, batch_size, shuffle=True, drop_last=False)

    input_dim = width * height
    model = E2C(input_dim, 3, 1)

    weights_init(model)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.1, 0.1))

    l = None
    ll = None
    reconst_loss = None

    plt.figure(1)
    plt.ion()
    plt.pause(0.01)

    # if os.path.exists('checkpoint.pth'):
    #     model = torch.load('checkpoint.pth')
    #     print('checkpoint loaded')

    for epoch in range(1000):
        model.train()
        for i, (x, u, x_next) in enumerate(loader):
            x = Variable(x.resize_(batch_size, input_dim), requires_grad=False)
            u = Variable(u)
            x_next = Variable(
                x_next.resize_(batch_size, input_dim), requires_grad=False)

            optimizer.zero_grad()

            dec, loss = model(x, u, x_next)
            #            loss = compute_loss(dec, x_next, model.z_mean, model.z_sigma)
            loss = loss()
            loss.backward()
            optimizer.step()

            l = loss.data[0]

        if epoch % 100 == 0:
            model.eval()
            x, actions, x_next = testset[:len(testset)]
            x = Variable(
                torch.from_numpy(x).view(len(testset), input_dim),
                requires_grad=False)
            actions = Variable(torch.from_numpy(actions), requires_grad=False)
            x_next = Variable(
                torch.from_numpy(x_next).view(len(testset), input_dim),
                requires_grad=False)
            predicted, _ = model(x, actions, x_next)

            concat = torch.cat([x_next, predicted], 1)
            grid = torchvision.utils.make_grid(
                concat.view(len(testset), 1, img_width * 2, img_width).data, nrow=16)

            # npz = model.z_mean.data.numpy() TODO
            show_and_save(grid,
                          os.path.join(model_dir,
                                       'test-{:04}.png'.format(epoch)))

            plt.pause(0.01)

            # data = Variable(torch.from_numpy(testset[:batch_size]).view(batch_size, 48 * 48))
            # predicted = vae(data)
            # data = predicted.resize(batch_size, 1, 48, 48).data
            # grid = torchvision.utils.make_grid(data)
            # plt.subplot(221)
            # show(grid)
            # plt.pause(0.05)
            print("Epoch: {}, Loss: {}".format(epoch, l))

        if epoch % 200 == 0:
            torch.save(model, 'checkpoint.pth')
            print('checkpoint saved')
