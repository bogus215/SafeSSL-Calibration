"""
    Base superclass `ImageAugment` for defining RGB image-based augmentations.
"""

import numpy as np


class ImageAugment(object):
    
    SUPPORTED_DATASETS = ['cifar10', 'cifar100', 'svhn', 'stl10', 'tiny', 'imagenet', 'imagenet32']

    MEAN = {
        'cifar10':      [0.4914, 0.4822, 0.4465],
        'cifar100':     [0.5071, 0.4867, 0.4408],
        'svhn':         [0.4377, 0.4437, 0.4728],
        'stl10':        [0.485,  0.456,  0.406],
        'tiny':         [0.4807,  0.4485,  0.3979],
        'imagenet':     [0.485,  0.456,  0.406],
        'imagenet32':   [0.485,  0.456,  0.406]
    }

    STD = {
        'cifar10':      [0.2023, 0.1994, 0.2010],
        'cifar100':     [0.2675, 0.2565, 0.2761],
        'svhn':         [0.1980, 0.2010, 0.1970],
        'stl10':        [0.229, 0.224, 0.225],
        'tiny':         [0.2618, 0.2537, 0.2676],
        'imagenet':     [0.229, 0.224, 0.225],
        'imagenet32':   [0.229, 0.224, 0.225]
    }

    WEAK_AUG_LIST = {
        'cifar10':      [True, True, True, True, True, False],
        'cifar100':     [True, True, True, True, True, False],
        'svhn':         [True, False, True, True, True, False],
        'tiny':     [True, True, True, True, True, False],
    }

    def __init__(self,
                 size: int or tuple,
                 data: str,
                 impl: str):

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size

        if data not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: '{data}'.")
        self.data = data

        if impl not in ['torchvision', 'albumentations']:
            raise ValueError
        self.impl = impl

    def __call__(self, img: np.ndarray):
        if self.impl == 'torchvision':
            return self.transform(img)
        elif self.impl == 'albumentations':
            return self.transform(image=img)['image']
        else:
            raise NotImplementedError

    def with_torchvision(self, size: tuple, blur: bool = True):
        raise NotImplementedError

    def with_albumentations(self, size: tuple, blur: bool = True):
        raise NotImplementedError

    @property
    def mean(self):
        return self.MEAN[self.data]

    @property
    def std(self):
        return self.STD[self.data]

    @property
    def weak_aug_list(self):
        return self.WEAK_AUG_LIST[self.data]