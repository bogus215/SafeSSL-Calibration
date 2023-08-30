from copy import deepcopy

import numpy as np
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# CIFAR-10
CIFAR_MISMATCH_CONFIG = {
    'cifar10': {'target_classes': [2, 3, 4, 5, 6, 7],
                'unknown_classes': [0, 1, 8, 9],
                'n_unlabeled': 20000},

    'cifar100': {'target_classes': list(range(0, 50)),
                 'unknown_classes': list(range(50,100)),
                 'n_unlabeled': 20000},

    'tinyimagenet': {'target_classes': list(range(0, 100)),
                     'unknown_classes': list(range(100,200)),
                     'n_unlabeled': 40000}
                     }

def load_CIFAR(root: str,
               data_name: str,
               n_label_per_class: int,
               n_valid_per_class: int,
               mismatch_ratio,
               **kwargs):

    # data setting
    if data_name == 'cifar10':
        num_classes = 10
        load_func = torchvision.datasets.cifar.CIFAR10
    elif data_name == 'cifar100':
        num_classes = 100
        load_func = torchvision.datasets.cifar.CIFAR100
    else:
        raise ValueError(f'{data_name} is not supported')
    CONFIG = CIFAR_MISMATCH_CONFIG[data_name]

    random_state = kwargs.get('seed',2021)
    np.random.seed(random_state)

    target_classes = CONFIG['target_classes']
    unknown_classes = CONFIG['unknown_classes']
    n_unlabels = CONFIG['n_unlabeled']

    # load dataset
    train_dataset = load_func(root=root, train=True, download=True)
    train_index, validation_index = train_test_split(np.arange(len(train_dataset)),
                                                     test_size=n_valid_per_class * num_classes,
                                                     stratify=train_dataset.targets,
                                                     shuffle=True,
                                                     random_state=random_state)

    validation_dataset = {'images': np.array(train_dataset.data)[validation_index],
                          'labels': np.array(train_dataset.targets)[validation_index]}
    train_dataset = {'images': np.array(train_dataset.data)[train_index],
                     'labels': np.array(train_dataset.targets)[train_index]}

    test_dataset = load_func(root=root, train=False, download=True)
    test_dataset = {'images': np.array(test_dataset.data),
                    'labels': np.array(test_dataset.targets)}

    # 1. train dataset
    # initialize dataset
    train_images = train_dataset['images']
    train_labels = train_dataset['labels']

    l_train_images, l_train_labels = [], []
    u_train_images, u_train_labels = [], []

    # target classes: labeled data, partial unlabeled data
    n_unlabels_per_cls = int(n_unlabels*(1.0-mismatch_ratio)) // len(target_classes)
    for c in target_classes:
        # train
        l_train_images.extend(train_images[train_labels == c][:n_label_per_class])
        l_train_labels.extend(train_labels[train_labels == c][:n_label_per_class])

        # unlabel train
        u_train_images.extend(train_images[train_labels == c][n_label_per_class:n_label_per_class+n_unlabels_per_cls])
        u_train_labels.extend(train_labels[train_labels == c][n_label_per_class:n_label_per_class+n_unlabels_per_cls])

    # unknown_classes: partial unlabeled data
    n_unlabels_shifts = (n_unlabels - n_unlabels_per_cls*len(target_classes)) // len(unknown_classes)
    for c in unknown_classes:
        # unlabel train
        u_train_images.extend(train_images[train_labels == c][:n_unlabels_shifts])
        u_train_labels.extend(train_labels[train_labels == c][:n_unlabels_shifts])

    # to array
    l_train_images, l_train_labels = np.array(l_train_images), np.array(l_train_labels)
    u_train_images, u_train_labels = np.array(u_train_images), np.array(u_train_labels)

    # to dictioanry
    l_train_dataset = {'images': l_train_images, 'labels': l_train_labels}
    u_train_dataset = {'images': u_train_images, 'labels': u_train_labels}

    # check OOD ratio
    n_in = np.isin(u_train_dataset['labels'], target_classes).sum()
    n_ood = len(u_train_dataset['labels']) - n_in
    print(f'[Mismatch={mismatch_ratio}] In: {n_in} vs. OOD : {n_ood}')

    # 2. validation: only target classes
    target_indices = np.isin(validation_dataset['labels'], target_classes)
    validation_dataset = {'images': validation_dataset['images'][target_indices],
                          'labels': validation_dataset['labels'][target_indices]}

    # 3. test: only target classes
    test_total_dataset = deepcopy(test_dataset)

    target_indices = np.isin(test_dataset['labels'], target_classes)
    test_dataset = {'images': test_dataset['images'][target_indices],
                    'labels': test_dataset['labels'][target_indices]}

    # convert to 0 ~ target_classes
    convert_target = dict([(k, i) for i, k in enumerate(target_classes)])
    convert_nontarget = dict([(k, i + len(target_classes)) for i, k in enumerate(unknown_classes)])

    convert_dict = convert_target.copy()
    convert_dict.update(convert_nontarget)

    l_train_dataset['labels'] = [convert_dict[x] for x in l_train_dataset['labels']]
    u_train_dataset['labels'] = [convert_dict[x] for x in u_train_dataset['labels']]
    validation_dataset['labels'] = [convert_dict[x] for x in validation_dataset['labels']]
    test_dataset['labels'] = [convert_dict[x] for x in test_dataset['labels']]
    test_total_dataset['labels'] = [convert_dict[x] for x in test_total_dataset['labels']]

    datasets = edict({
        'l_train': l_train_dataset,
        'u_train': u_train_dataset,
        'validation': validation_dataset,
        'test': test_dataset,
        'test_total': test_total_dataset
    })

    return datasets, convert_dict

class CIFAR(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)

class CIFAR_STRONG(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            weak_img = self.transform(img)
            strong_img = self.transform.strong_transform(img)

        return dict(weak_img=weak_img, strong_img=strong_img, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)
    
class CIFAR_WEAK_AND_RAW(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            weak_img = self.transform(img)
            raw_img = self.transform.raw_transform(img)

        return dict(weak_img=weak_img, raw_img=raw_img, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)
    
class CIFAR_K_AUG(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.K = kwargs.get("K",2)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            weak_img = self.transform(img)
            data = dict(weak_img=weak_img, y=target, idx=idx)
            K_data = dict([(f"weak_img_{i+1}",self.transform(img)) for i in range(1,self.K)])
            data.update(K_data)

        return data

    def __len__(self):
        return len(self.targets)