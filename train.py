import torch
from torch import optim
import torchvision
from torch.autograd import Variable
from torch import nn
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from pixel2torque.pytorch.vae import Encoder, Decoder, VAE, latent_loss
from torch.utils.data import Dataset, DataLoader
import pickle


def show(img):
    npimg = img.numpy()
    return plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


class PendulumData(Dataset):
    def __init__(self, root, split):
        if split not in ['train', 'test', 'all']:
            raise ValueError

        dir = os.path.join(root, split)
        filenames = glob.glob(os.path.join(dir, '*.png'))

        if split == 'all':
            filenames = glob.glob(os.path.join(root, 'train/*.png'))
            filenames.extend(glob.glob(os.path.join(root, 'test/*.png')))

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


def compute_loss(reconst, actual, mean, var):
    ll = latent_loss(mean, var)
    reconst_loss = torch.sum(reconst.sub(actual).pow(2), dim=1)
    loss = reconst_loss.add(ll).mean()
    return loss


def load_checkpoint(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)


def save_checkpoint(model, filename):
    with open(filename, 'wb') as outfile:
        return pickle.dump(model, outfile)


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    dataset = PendulumData('data/pendulum_data/pendulum_max-speed_0.1-gravity-train-test', 'all')

    # trainset = PendulumData('bitbucket/data/processed/pendulum_max-speed_0.1-gravity-train-test', 'train')
    # testset = PendulumData('bitbucket/data/processed/pendulum_max-speed_0.1-gravity-train-test', 'test')

    batch_size = len(dataset)

    print('batch_size %d' % batch_size)

    loader = DataLoader(dataset, batch_size,
                        shuffle=True, drop_last=False)

    input_dim = 48 * 48
    vae = VAE(input_dim, 3)

    weights_init(vae)
    criterion = nn.MSELoss()
    criterion.size_average = False

    optimizer = optim.Adam(vae.parameters(), lr=3e-4, betas=(0.1, 0.1))

    l = None
    ll = None
    reconst_loss = None

    plt.figure(1)
    plt.ion()
    plt.pause(0.01)

    if os.path.exists('checkpoints.pt'):
        vae = load_checkpoint('checkpoints.pt')
        print('Checkpoint loaded')

    for epoch in range(1000):
        vae.train()
        for i, data in enumerate(loader):
            inputs = Variable(data.resize_(batch_size, input_dim),
                              requires_grad=False)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma_sq)
            loss = compute_loss(dec, inputs, vae.z_mean, vae.z_sigma_sq)
            loss.backward()
            optimizer.step()
            l = loss.data[0]

        if epoch % 100 == 0:
            vae.eval()
            data = Variable(torch.from_numpy(dataset[np.random.randint(batch_size, size=(4,)), :]).view(4, 48 * 48)
                            , requires_grad=False)
            predicted = vae(data)
            concat = torch.cat([data, predicted], 1)
            grid = torchvision.utils.make_grid(concat.view(4, 1, 48 * 2, 48).data, nrow=4)
            show(grid)
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
            save_checkpoint(vae, 'checkpoint.pt')
            print('Checkpoint saved')
