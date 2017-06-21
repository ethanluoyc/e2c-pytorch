import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torchvision
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
from skimage.transform import resize

from torch.utils.data import Dataset
from pixel2torque.tf_e2c.plane_data2 import T, num_t
from skimage.transform import resize
from skimage.color import rgb2gray


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


class GymPendulumDataset(Dataset):
    width = 40
    height = 40
    """Sample from the OpenAI Gym environment, requires a patched version of gym"""
    def __init__(self, filename):
        _data = np.load(filename)
        self.X0 = np.copy(_data['X0'])  # Copy to memory, otherwise it's slow.
        self.X1 = np.copy(_data['X1'])
        self.U = np.copy(_data['U'])
        _data.close()

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, index):
        return self.X0[index], self.U[index], self.X1[index]

    @classmethod
    def all_states(cls):
        _env = gym.make('Pendulum-v0').env
        width = GymPendulumDataset.width
        height = GymPendulumDataset.height
        X = np.zeros((360, width, height))

        for i in range(360):
            th = i / 360. * 2 * np.pi
            state = _env.render_state(th)
            X[i, :, :] = resize(rgb2gray(state), (width, height), mode='reflect')
        _env.close()
        _env.viewer.close()
        return X

    @classmethod
    def sample_trajectories(self, sample_size, step_size=1, apply_control=True):
        _env = gym.make('Pendulum-v0').env
        X0 = np.zeros((sample_size, 500, 500, 3), dtype=np.uint8)
        U  = np.zeros((sample_size, 1), dtype=np.float32)
        X1 = np.zeros((sample_size, 500, 500, 3), dtype=np.uint8)
        for i in range(sample_size):
            th = np.random.uniform(0, np.pi*2)
            thdot = np.random.uniform(-8, 8)
            state = np.array([th, thdot])
            initial = state
            # apply the same control over a few timesteps
            if apply_control:
                u = np.random.uniform(-2, 2, size=(1,))
            else:
                u = np.zeros((1,))
            for _ in range(step_size):
                state = _env.step_from_state(state, u)

            X0[i, :, :, :] = _env.render_state(initial[0])
            U[i, :] = u
            X1[i, :, :, :] = _env.render_state(state[0])
        _env.viewer.close()
        return X0, U, X1
