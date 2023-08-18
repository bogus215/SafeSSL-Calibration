from copy import deepcopy

import numpy as np
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# CIFAR-10
CIFAR_MISMATCH_CONFIG = {
    'cifar10': {'target_classes': [2, 3, 4, 5, 6, 7],
                'unlabeled_classes': {'0.00': [4, 5, 6, 7],
                                      '0.25': [0, 5, 6, 7],
                                      '0.50': [0, 1, 6, 7],
                                      '0.75': [0, 1, 7, 8],
                                      '1.00': [0, 1, 8, 9]}},
    'cifar100': {'target_classes': list(range(0, 50)),
                 'unlabeled_classes': {'0.50': list(range(25, 75))}}
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
    unlabeled_classes = CONFIG['unlabeled_classes'][mismatch_ratio]
    nontarget_classes = list(set(range(num_classes)) - set(target_classes))

    # load dataset
    train_dataset = load_func(root=root, train=True, download=False)
    train_index, validation_index = train_test_split(np.arange(len(train_dataset)),
                                                     test_size=n_valid_per_class * num_classes,
                                                     stratify=train_dataset.targets,
                                                     shuffle=True,
                                                     random_state=random_state)

    validation_dataset = {'images': np.array(train_dataset.data)[validation_index],
                          'labels': np.array(train_dataset.targets)[validation_index]}
    train_dataset = {'images': np.array(train_dataset.data)[train_index],
                     'labels': np.array(train_dataset.targets)[train_index]}

    test_dataset = load_func(root=root, train=False, download=False)
    test_dataset = {'images': np.array(test_dataset.data),
                    'labels': np.array(test_dataset.targets)}

    # 1. train dataset
    # initialize dataset
    train_images = train_dataset['images']
    train_labels = train_dataset['labels']

    l_train_images, l_train_labels = [], []
    u_train_images, u_train_labels = [], []

    # target classes: labeled data
    for c in target_classes:
        # train
        l_train_images.extend(train_images[train_labels == c][:n_label_per_class])
        l_train_labels.extend(train_labels[train_labels == c][:n_label_per_class])

    # unlabel - doesnt matter whether target or not
    for c in unlabeled_classes:
        # unlabel train
        u_train_images.extend(train_images[train_labels == c][n_label_per_class:])
        u_train_labels.extend(train_labels[train_labels == c][n_label_per_class:])

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
    convert_nontarget = dict([(k, i + len(target_classes)) for i, k in enumerate(nontarget_classes)])

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