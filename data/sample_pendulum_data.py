from pixel2torque.pytorch.datasets import GymPendulumDatasetV2
import numpy as np

np.random.seed(0)

GymPendulumDatasetV2.sample(10000, 'data/pendulum_markov')
dataset = GymPendulumDatasetV2('data/pendulum_markov')
