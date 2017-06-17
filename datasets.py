import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
from skimage.transform import resize

from torch.utils.data import Dataset
from pixel2torque.tf_e2c.plane_data2 import T, num_t


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


class PlaneDataset(Dataset):
    def __init__(self, planedata):
        self.planedata = planedata

    def __len__(self):
        return T * num_t  # Total number of samples

    def __getitem__(self, index):
        index = np.random.randint(0, num_t)  # Sample any one of them
        t = np.random.randint(0, T - 1)
        x = np.array(self.planedata.getX(index, t))
        x_next = np.array(self.planedata.getX(index, t + 1))
        u = np.copy(self.planedata.U[index, t, :])
        return x, u, x_next
