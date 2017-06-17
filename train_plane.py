import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
from skimage.transform import resize

from pixel2torque.pytorch.e2c import E2C
from pixel2torque.pytorch.vae import VAE, latent_loss
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

img_width = 40


def show_and_save(img, path):
    npimg = img.numpy()
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))


class PendulumData(Dataset):
    def __init__(self, root, split):
        if split not in ['train', 'test', 'all']:
            raise ValueError

        dir = os.path.join(root, split)
        filenames = glob.glob(os.path.join(dir, '*.png'))

        if split == 'all':
            filenames = glob.glob(os.path.join(root, 'train/*.png'))
            filenames.extend(glob.glob(os.path.join(root, 'test/*.png')))

        filenames = sorted(
            filenames, key=lambda x: int(os.path.basename(x).split('.')[0]))

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

        self.actions = actions[:len(self.images)].astype(np.float32)
        self.actions = self.actions.reshape(len(actions), 1)

    def __len__(self):
        return len(self.actions) - 1

    def __getitem__(self, index):
        return self.images[index], self.actions[index], self.images[index]


def weights_init(m):
    from torch.nn.init import xavier_uniform
    if isinstance(m, torch.nn.Linear):
        xavier_uniform(m.weight.data)


def compute_loss(reconst, actual, mean, var):
    ll = latent_loss(mean, var)
    reconst_loss = torch.sum(reconst.sub(actual).pow(2), dim=1)
    loss = reconst_loss.add(ll).mean()
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description='train e2c on plane data')
    parser.add_argument('--batch-size', required=False, default=128, type=int)
    parser.add_argument('--lr', required=False, default=3e-4, type=float)
    parser.add_argument('--seed', required=False, default=1234, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    width = img_width
    height = width
    model_dir = 'checkpoints'

    from pixel2torque.tf_e2c.plane_data2 import PlaneData

    env_path = os.path.join(os.path.dirname(__file__), '../tf_e2c', 'env0.png')
    planedata = PlaneData('plane0.npz', env_path)
    planedata.initialize()

    class PlaneData(torch.utils.data.Dataset):
        def __init__(self, planedata):
            self.planedata = planedata
            self.sampled = planedata.sample(128)

        def __len__(self):
            return len(self.sampled[0])

        def __getitem__(self, index):
            return self.sampled[0][index], self.sampled[1][
                index], self.sampled[2][index]

    dataset = PlaneData(planedata)
    trainset = dataset
    testset = dataset

    batch_size = len(dataset)
    print('batch_size %d' % batch_size)

    loader = DataLoader(trainset, batch_size, shuffle=True, drop_last=False)

    input_dim = width * height
    latent_dim = 2
    action_dim = 2

    model = E2C(input_dim, latent_dim, action_dim)

    print(model)
    weights_init(model)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.1, 0.1))

    l = None
    ll = None
    reconst_loss = None
    fig = plt.figure('Embeddings')
    true_ax = plt.subplot(121)
    pred_ax = plt.subplot(122)

    for epoch in range(1000):
        fig.suptitle('Epoch: {}'.format(epoch))
        model.train()
        for i, (x, u, x_next) in enumerate(loader):
            x = Variable(
                x.resize_(batch_size, input_dim).float(), requires_grad=False)
            u = Variable(u.float())
            x_next = Variable(
                x_next.resize_(batch_size, input_dim).float(),
                requires_grad=False)

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
                torch.from_numpy(x).view(len(testset), input_dim).float(),
                requires_grad=False)
            actions = Variable(
                torch.from_numpy(actions).float(), requires_grad=False)
            x_next = Variable(
                torch.from_numpy(x_next).view(len(testset), input_dim).float(),
                requires_grad=False)
            predicted, _ = model(x, actions, x_next)

            # x = planedata.getPSpace()[:,1]
            # y = planedata.getPSpace()[:,0]
            # cs = np.array([i for i in range(len(x))]) / len(x)
            # plt.scatter(x, y, c=cs)
            # plt.show()

            all_states = []
            for p in planedata.getPSpace():
                all_states.append(np.array(planedata.getXp(p.astype(np.int))))

            all_states = Variable(
                torch.from_numpy(np.array(all_states)).view(
                    len(planedata.getPSpace()), input_dim))
            embeds = model.latent_embeddings(all_states)

            # For each set of style and range settings, plot n random points in the box
            # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            colors = np.array(range(len(all_states))) / len(all_states)

            xs = embeds[:, 0].data.numpy()
            ys = embeds[:, 1].data.numpy()

            true_ax.scatter(
                planedata.getPSpace()[:, 0],
                planedata.getPSpace()[:, 1],
                c=colors)
            pred_ax.scatter(xs, ys, c=colors)

            plt.pause(0.01)
            plt.show(block=False)
            concat = torch.cat([x_next, predicted], 1)
            grid = torchvision.utils.make_grid(
                concat.view(len(testset), 1, img_width * 2, img_width).data,
                nrow=16)

            show_and_save(grid, os.path.join(model_dir, 'test-{:04}.png'.format(epoch)))

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
