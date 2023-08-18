import albumentations as A
from torchvision import transforms
from datasets.transforms.base import ImageAugment
from datasets.transforms.albumentations import NumpyToTensor
import torch
from datasets.transforms.randaugment import RandAugmentMC

class SemiAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super(SemiAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
            self.strong_transform = self.with_strong_torchvision()
            
        elif self.impl == 'albumentations':
            raise NotImplementedError()

    def with_torchvision(self):

        transform = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            GaussianNoise()
        ]

        transform = [item for item, flag in zip(transform, self.weak_aug_list) if flag]

        return transforms.Compose(transform)

    def with_strong_torchvision(self):

        transform = [
            transforms.ToPILImage(),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ]

        return transforms.Compose(transform)

class TestAugment(ImageAugment):
    def __init__(self,
                 size: int or tuple = (224, 224),
                 data: str = 'imagenet',
                 impl: str = 'torchvision',
                 **kwargs):
        super(TestAugment, self).__init__(size, data, impl)

        if self.impl == 'torchvision':
            self.transform = self.with_torchvision()
        elif self.impl == 'albumentations':
            self.transform = self.with_albumentations()

    def with_torchvision(self):
        transform = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return transforms.Compose(transform)

    def with_albumentations(self):
        transform = [
            A.Normalize(self.mean, self.std, always_apply=True),
            NumpyToTensor()
        ]
        return A.Compose(transform)
    
class GaussianNoise(object):
    def __init__(self, mean=0., std=0.15):
        self.std = std
        self.mean = mean

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)