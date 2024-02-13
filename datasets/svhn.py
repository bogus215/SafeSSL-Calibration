from copy import deepcopy

import numpy as np
import torchvision
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# SVHN
SVHN_MISMATCH_CONFIG = {
    'svhn': {'target_classes': [2, 3, 4, 5, 6, 7],
                 'unknown_classes': [0, 1, 8, 9],
                 'n_unlabeled': 20000}
    }

def load_SVHN(root: str,
              data_name: str,
              n_label_per_class: int,
              mismatch_ratio,
              random_state,
              logger,
              n_ratio_valid_per_class: float = 0.1):

    assert data_name=='svhn', f"{data_name} is not supported"
    load_func = torchvision.datasets.svhn.SVHN

    CONFIG = SVHN_MISMATCH_CONFIG[data_name]

    np.random.seed(random_state)

    target_classes = CONFIG['target_classes']
    unknown_classes = CONFIG['unknown_classes']
    n_unlabels = CONFIG['n_unlabeled']
    
    # load dataset
    train_dataset = load_func(root=root, split='train', download=True)
    train_index, validation_index = train_test_split(np.arange(len(train_dataset)),
                                                     train_size = (1-n_ratio_valid_per_class),
                                                     test_size = n_ratio_valid_per_class,
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
    logger.info(f'[Mismatch={mismatch_ratio}] In: {n_in} vs. OOD : {n_ood}')

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

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return dict(x=img, y=target, idx=idx)

    def __len__(self):
        return len(self.targets)
    
class SVHN_STRONG(Dataset):
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


class Selcted_DATA(Dataset):
    def __init__(self,
                 dataset: dict,
                 name: str,
                 transform: object = None,
                 **kwargs):

        self.data = deepcopy(dataset['images'])
        self.targets = deepcopy(dataset['labels'])
        self.transform = transform
        self.name = name
        
        self.idx_data = np.arange(len(self.data))
        
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices.cpu()]
            self.targets_index = np.array(self.targets)[indices.cpu()].tolist()
        else:
            self.data_index = self.data
            self.targets_index = self.targets
            
    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        if self.targets is None:
            target = None
        else:
            target = self.targets_index[idx]
        img = self.data_index[idx]
        data_idx = self.idx_data[idx]

        return img, target, data_idx

    def __getitem__(self, idx):
        
        img, target, data_idx = self.__sample__(idx)
        if self.transform is not None:
            weak_img = self.transform(img)

        if self.name == 'train_lb':
            return {'idx_lb': data_idx, 'x_lb': weak_img, 'x_lb_w_0': weak_img, 'x_lb_w_1': self.transform(img),'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': data_idx, 'x_ulb_w_0': weak_img, 'x_ulb_w_1': self.transform(img), 'y_ulb': target}
        elif self.name == 'train_ulb_selected':
            return {'x_ulb_w': weak_img, 'x_ulb_w_1': self.transform(img), 'x_ulb_s': self.transform.strong_transform(img), 'unlabel_y': target}
        else:
            raise ValueError
        
class Selcted_DATA_Proposed(Dataset):
    def __init__(self,
                 dataset: dict,
                 name: str,
                 transform: object = None,
                 **kwargs):

        self.data = deepcopy(dataset['images'])
        self.targets = deepcopy(dataset['labels'])
        self.transform = transform
        self.name = name
        
        self.data_index = None
        self.targets_index = None
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices.cpu()]
            self.targets_index = np.array(self.targets)[indices.cpu()].tolist()
        else:
            self.data_index = self.data
            self.targets_index = self.targets
            
    def __len__(self):
        return len(self.data_index)

    def __sample__(self, idx):
        if self.targets is None:
            target = None
        else:
            target = self.targets_index[idx]
        img = self.data_index[idx]

        return img, target

    def __getitem__(self, idx):
        
        img, target = self.__sample__(idx)
        if self.transform is not None:
            weak_img = self.transform(img)

        if self.name == 'train_lb':
            return {'idx_lb': idx, 'x_lb': weak_img, 'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w': weak_img, 'x_ulb_s': self.transform.strong_transform(img) ,'y_ulb': target}
        elif self.name == 'train_ulb_selected':
            return {'x_ulb_w': weak_img, 'x_ulb_s': self.transform.strong_transform(img), 'unlabel_y': target}
        else:
            raise ValueError
        
        
class MTC_DATASET(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 len_label,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform

        # Labeled data : 1 / Unlabeled data : 0 <--> match data : 1 / mismatch data : 0
        self.len_label = len_label
        self.soft_labels = np.zeros((len(self.data)))
        self.soft_labels[:len_label] = 1

    def label_update(self, results):

        self.soft_labels[self.len_label:] = results[self.len_label:].cpu().numpy()

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        soft_labels = self.soft_labels[idx]

        if self.transform is not None:
            weak_img = self.transform(img)

        return dict(weak_img=weak_img, target=target, soft_label=soft_labels, idx=idx)

    def __len__(self):
        return len(self.data)