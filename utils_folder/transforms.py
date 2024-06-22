import random

import numpy as np
import torch
from scipy.ndimage import rotate


class RandomHorizontalFlipMultiChannel(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {key: torch.flip(value, dims=[-1]) if torch.is_tensor(value) else value for key, value in sample.items()}
        return sample
    
class RandomBrightnessMultiChannel(object):
    def __init__(self, factor_range=(0.99, 1.01), p=0.5):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            brightness_factor = random.uniform(self.factor_range[0], self.factor_range[1])
            return {key: value * brightness_factor if torch.is_tensor(value) else value for key, value in sample.items()}
        return sample