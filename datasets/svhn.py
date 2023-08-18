import cv2
import numpy as np
import torchvision
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
import os
import glob
import torch

def load_only_testCIFAR(root:str,
                       data_name:str,
                       **kwargs):
    
    assert data_name=='cifar10', f"{data_name} is not supported"
    load_func = torchvision.datasets.cifar.CIFAR10
    
    random_state = kwargs.get('seed',2021)
    np.random.seed(random_state)
    
    # load dataset
    test_dataset = load_func(root=root, train=False, download=True)
    
    test_dataset = {'images': np.array(test_dataset.data),
                     'labels': np.array(test_dataset.targets)}

    datasets = edict({
        'test': test_dataset
    })

    return datasets

def load_SVHN(root: str,
               data_name: str,
               n_ratio_valid_per_class: float = 0.1,
               **kwargs):

    assert data_name=='svhn', f"{data_name} is not supported"
    load_func = torchvision.datasets.svhn.SVHN

    random_state = kwargs.get('seed',2021)
    drop_train = kwargs.get('drop_train',None)
    drop_validation = kwargs.get('drop_validation',None)
    np.random.seed(random_state)

    # load dataset
    train_dataset = load_func(root=root, split='train', download=True)
    train_index, validation_index = train_test_split(np.arange(len(train_dataset)),
                                                     train_size = (1-n_ratio_valid_per_class) if drop_train is None else float(drop_train),
                                                     test_size = n_ratio_valid_per_class if drop_validation is None else (n_ratio_valid_per_class * drop_validation),
                                                     stratify = train_dataset.labels,
                                                     shuffle = True,
                                                     random_state = random_state)

    train_dataset.data = train_dataset.data.transpose(0,2,3,1)
    validation_dataset = {'images': np.array(train_dataset.data)[validation_index],
                          'labels': np.array(train_dataset.labels)[validation_index]}
    train_dataset = {'images': np.array(train_dataset.data)[train_index],
                     'labels': np.array(train_dataset.labels)[train_index]}

    test_dataset = load_func(root=root, split='test', download=True)
    test_dataset.data = test_dataset.data.transpose(0,2,3,1)
    test_dataset = {'images': np.array(test_dataset.data),
                    'labels': np.array(test_dataset.labels)}

    features, targets = [], []
    corrupted_paths = {'svhn':'SVHN-C'}
    for npy in glob.glob(os.path.join(root, corrupted_paths[data_name],'feature','**.npy')):
        features.append(np.load(npy))
        targets.append(np.load(os.path.join(root, corrupted_paths[data_name],'labels.npy')))
    
    c_test_dataset = {'images':np.concatenate(features,axis=0),'labels':np.concatenate(targets,axis=0)}

    datasets = edict({
        'l_train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
        'c_test': c_test_dataset
    })

    return datasets

class SVHN(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    @staticmethod
    def load_image_cv2(path: str):
        """Load image with opencv."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.targets)

class SVHN_with_OOD(Dataset):
    def __init__(self,
                 data_name: str,
                 feature: np.array,
                 targets: np.array,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = feature
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, ood_labels=target, idx=idx)

    def __len__(self):
        return len(self.targets)

class SVHN_ADV(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            x1 = self.transform(img)
            x2 = self.transform(img)
            x3 = self.transform(img)

        return dict(x1=x1, x2=x2, x3=x3, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)
    
class SVHN_AUGMIXDROP(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 mixture_width: int,
                 mixture_depth: int,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img_orgin, augmix1, augmix2 = (self.transform.orgin_transform(img), 
                                       self.augmix(img, 
                                                   utils_preprocess=self.transform.utils1_transform, 
                                                   utils2_preprocess=self.transform.utils2_transform,
                                                   utils3_preprocess=self.transform.utils3_transform,
                                                   orgin_preprocess=self.transform.orgin_transform),
                                       self.augmix(img, 
                                                   utils_preprocess=self.transform.utils1_transform, 
                                                   utils2_preprocess=self.transform.utils2_transform,
                                                   utils3_preprocess=self.transform.utils3_transform,
                                                   orgin_preprocess=self.transform.orgin_transform)
                                       )

        return dict(img_orgin=img_orgin, img_augmix1=augmix1, img_augmix2=augmix2, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)

    def augmix(self,image ,utils_preprocess, utils2_preprocess, utils3_preprocess, orgin_preprocess):
        
        # image : Raw image
        # utils_preprocess : from Raw Image to ToPILImage
        # utils2_preprocess : from ToPILImage to ToTensor * Normalize
        # utils3_preprocess : from ToPILImage to RANDOMHORIZIN*CROP
        # origin_transform : from Raw Image to ToPILIMAGE*RANDOMHORIZIN*CROP*ToTensor * Normalize

        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        orgin_preprocessed_image = orgin_preprocess(image)
        mix = torch.zeros_like(orgin_preprocessed_image)
        for i in range(self.mixture_width):
            image_aug = utils_preprocess(image.copy())
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                image_aug = utils3_preprocess()(image_aug)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * utils2_preprocess(image_aug)

        mixed = (1 - m) * orgin_preprocessed_image + m * mix
        return mixed
    

class SVHN_AUGMIX(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 mixture_width: int,
                 mixture_depth: int,
                 aug_severity: int,
                 aug_list: list,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.aug_list = aug_list

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img_orgin, augmix1, augmix2 = (self.transform.orgin_transform(img), 
                                       self.augmix(img, utils_preprocess=self.transform.utils1_transform, utils2_preprocess=self.transform.utils2_transform,orgin_preprocess=self.transform.orgin_transform),
                                       self.augmix(img, utils_preprocess=self.transform.utils1_transform, utils2_preprocess=self.transform.utils2_transform,orgin_preprocess=self.transform.orgin_transform))

        return dict(img_orgin=img_orgin, img_augmix1=augmix1, img_augmix2=augmix2, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)

    def augmix(self,image ,utils_preprocess, utils2_preprocess, orgin_preprocess):
        
        """Perform AugMix augmentations and compute mixture.
        Args:
            image: PIL.Image input image
            preprocess: Preprocessing function which should return a torch tensor.
        Returns:
            mixed: Augmented and mixed image.
        """

        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        orgin_preprocessed_image = orgin_preprocess(image)
        mix = torch.zeros_like(orgin_preprocessed_image)
        for i in range(self.mixture_width):
            image_aug = utils_preprocess(image.copy())
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * utils2_preprocess(image_aug)

        mixed = (1 - m) * orgin_preprocessed_image + m * mix
        return mixed

class SVHN_PROPOSED2(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 aug_severity: int,
                 aug_list: list,
                 aug_depth_min: int,
                 aug_depth_max: int, 
                 aug_depth_width: int,
                 beta_lower: float = 1.0,
                 beta_upper: float = 1.0,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.aug_severity = aug_severity
        self.aug_list = aug_list
        self.aug_depth_min = aug_depth_min
        self.aug_depth_max = aug_depth_max
        self.aug_depth_width = aug_depth_width
        self.beta_lower = beta_lower
        self.beta_upper = beta_upper

        assert self.aug_depth_min <= self.aug_depth_max

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        weak_data = self.transform.weak_transform(img)
        augmix_data = self.hard_augmented(img,
                                   to_pil_transform = self.transform.to_pil_transform,
                                   to_tensor_normalize_transform=self.transform.to_tensor_normalize_transform,
                                   weak_augmented_data = weak_data)

        return dict(weak_data=weak_data,
                    augmix_data=augmix_data,
                    y=target, idx=idx)


    def __len__(self):
        return len(self.targets)

    def hard_augmented(self, image ,to_pil_transform, to_tensor_normalize_transform, weak_augmented_data):
        
        ws = np.float32(np.random.dirichlet([1] * self.aug_depth_width))
        m = np.float32(np.random.beta(self.beta_lower, self.beta_upper))

        mix = torch.zeros_like(weak_augmented_data)
        for i in range(self.aug_depth_width):
            image_aug = to_pil_transform(image.copy())
            depth = np.random.randint(self.aug_depth_min, self.aug_depth_max + 1 )

            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * to_tensor_normalize_transform(image_aug)

        mixed = (1 - m) * weak_augmented_data + m * mix

        return mixed
    
    
class SVHN_PROPOSED5_Temperature_Scaling(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 aug_severity: int,
                 aug_list: list,
                 aug_depth_min: int,
                 aug_depth_max: int, 
                 aug_depth_width: int,
                 transform: object = None,
                 test_transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.test_transform = test_transform
        self.aug_severity = aug_severity
        self.aug_list = aug_list
        self.aug_depth_min = aug_depth_min
        self.aug_depth_max = aug_depth_max
        self.aug_depth_width = aug_depth_width

        assert self.aug_depth_min < self.aug_depth_max

        if self.data_name == 'svhn':
            self.num_classes = 10
        else:
            raise ValueError(f'{self.data_name} is not supported')

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        weak_augmented_data1 , weak_augmented_data2 = self.transform.weak_transform(img), self.transform.weak_transform(img)
        test_data = self.test_transform(img)

        hard1 = self.hard_augmented(img,
                                    to_pil_transform = self.transform.to_pil_transform,
                                    to_tensor_normalize_transform=self.transform.to_tensor_normalize_transform,
                                    weak_augmented_data = weak_augmented_data1)

        hard2 = self.hard_augmented(img,
                                    to_pil_transform = self.transform.to_pil_transform,
                                    to_tensor_normalize_transform=self.transform.to_tensor_normalize_transform,
                                    weak_augmented_data = weak_augmented_data2)

        return dict(weak_augmented_data1=weak_augmented_data1,
                    weak_augmented_data2=weak_augmented_data2,
                    hard_augmented_data1=hard1,
                    hard_augmented_data2=hard2,
                    test_data=test_data,
                    y=target, idx=idx)

    def __len__(self):
        return len(self.targets)

    def hard_augmented(self, image ,to_pil_transform, to_tensor_normalize_transform, weak_augmented_data):
        
        ws = np.float32(np.random.dirichlet([1] * self.aug_depth_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(weak_augmented_data)
        for i in range(self.aug_depth_width):
            image_aug = to_pil_transform(image.copy())
            depth = np.random.randint(self.aug_depth_min, self.aug_depth_max + 1 )

            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * to_tensor_normalize_transform(image_aug)

        mixed = (1 - m) * weak_augmented_data + m * mix

        return mixed