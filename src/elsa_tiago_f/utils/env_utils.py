import gym
from gym import spaces
import numpy as np

class Environment():
    pass


class DiscretizedBox(spaces.Discrete):
    def __init__(self, box_space, num_bins_per_dim):
        assert isinstance(box_space, spaces.Box), "Input must be a Box space"
        self.box_space = box_space
        self.low = box_space.low
        self.high = box_space.high
        self.shape = (num_bins_per_dim,) * box_space.shape[0]
        self.n = np.prod(self.shape, dtype=int)
        self.delta = (self.high - self.low) / (self.n - 1)
        super(DiscretizedBox, self).__init__(self.n)

    def discretize(self, action):
        indices = ((action - self.low) / self.delta).astype(int)
        indices = np.clip(indices, 0, self.n - 1)
        return tuple(indices)
