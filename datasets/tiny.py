import os
import glob
import cv2
from copy import deepcopy
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict
from torch.utils.data import Dataset

TINY_MISMATCH_CONFIG = {
    'tiny': {'target_classes': list(range(0, 100)),
             'unknown_classes': list(range(100,200)),
             'n_unlabeled': 40000}
    }

def load_image_cv2(path: str):
    """Load image with OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dsize=(32,32), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 2:
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), dsize=(32,32), interpolation=cv2.INTER_LINEAR)
    else:
        raise NotImplementedError

def load_tiny(root: str, 
              n_label_per_class: int,
              n_valid_per_class: int,
              mismatch_ratio,
              random_state,
              logger,
              **kwargs):

    data_dir = os.path.join(root, 'tiny-imagenet-200')
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CONFIG = TINY_MISMATCH_CONFIG['tiny']
    num_classes = 200

    target_classes = CONFIG['target_classes']
    unknown_classes = CONFIG['unknown_classes']
    n_unlabels = CONFIG['n_unlabeled']

    with open(os.path.join(data_dir, CLASS_LIST_FILE), 'r') as fp:
        label_texts = [text.strip() for text in fp.readlines()]
    label_text_to_number = {text: i for i, text in enumerate(label_texts)}

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # train: shuffled
    image_paths = sorted(glob.glob(os.path.join(train_dir, "**/*.JPEG"), recursive=True))
    np.random.seed(random_state)
    np.random.shuffle(image_paths)

    train_images = np.array([load_image_cv2(path) for path in tqdm(image_paths)])
    train_labels = np.array([label_text_to_number[path.split("/")[-3]] for path in image_paths])
    
    train_index, validation_index = train_test_split(np.arange(len(train_images)),
                                                     test_size=n_valid_per_class * num_classes,
                                                     stratify=train_labels,
                                                     shuffle=True,
                                                     random_state=random_state)

    validation_dataset = {'images': np.array(train_images)[validation_index],
                          'labels': np.array(train_labels)[validation_index]}
    train_dataset = {'images': np.array(train_images)[train_index],
                     'labels': np.array(train_labels)[train_index]}

    # 1. train dataset
    # initialize dataset
    train_images = train_dataset['images']
    train_labels = train_dataset['labels']

    # 1. train dataset
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

    del train_images, train_labels

    # check OOD ratio
    n_in = np.isin(u_train_dataset['labels'], target_classes).sum()
    n_ood = len(u_train_dataset['labels']) - n_in
    if logger is not None:
        logger.info(f'[Mismatch={mismatch_ratio}] In: {n_in} vs. OOD : {n_ood}')

    # 2. validation: only target classes
    target_indices = np.isin(validation_dataset['labels'], target_classes)
    validation_dataset = {'images': validation_dataset['images'][target_indices],
                          'labels': validation_dataset['labels'][target_indices]}

    # 3. test: only target classes
    test_images = []
    test_labels = []

    with open(os.path.join(val_dir, VAL_ANNOTATION_FILE), 'r') as fp:
        for line in fp.readlines():
            filename, label_text, *_ = line.split('\t')
            label = label_text_to_number[label_text]
            img = load_image_cv2(os.path.join(val_dir, 'images', filename))
            test_images.append(img)
            test_labels.append(label)

    test_dataset = {'images': np.array(test_images),
                    'labels': np.array(test_labels)}

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


class TinyImageNet(Dataset):
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

class TinyImageNet_STRONG(Dataset):
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
    
class TINY_TWO_AUG(Dataset):
    def __init__(self,
                 data_name: str,
                 dataset: dict,
                 name: str,
                 transform: object = None,
                 **kwargs):

        self.data_name = data_name
        self.data = dataset['images']
        self.targets = dataset['labels']
        self.transform = transform
        self.name = name
        self.set_index()

    def set_index(self, indices=None):
        if indices is not None:
            self.data_index = self.data[indices]
            self.targets_index = self.targets[indices]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

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
            return {'idx_lb': idx, 'x_lb': weak_img, 'x_lb_w_0': weak_img, 'x_lb_w_1': self.transform(img),'y_lb': target}
        elif self.name == 'train_ulb':
            return {'idx_ulb': idx, 'x_ulb_w_0': weak_img, 'x_ulb_w_1': self.transform(img), 'y_ulb': target}
        elif self.name == 'train_ulb_selected':
            return {'x_ulb_w': weak_img, 'x_ulb_s': self.transform.strong_transform(img), 'unlabel_y': target}

    def __len__(self):
        return len(self.data_index)
    
if __name__ == '__main__':

    datasets, _ = load_tiny(root='/mnthdd/Dropbox/D/personal_study/SafeSSL+Calibration/datasets',
                            n_label_per_class=100,
                            n_valid_per_class=50,
                            mismatch_ratio=0.3,
                            random_state=99999,
                            logger=None
                            )
    
'''
TinyImageNet contains 200 categories which includes 500 training images and 50 testing images in each category.
We resize all images to 32 × 32. We use the first 100 categories as seen classes, 
and the remaining classes as unseen classes. 
We select 100 images from each seen category to construct the labeled data set DL, i.e., 10000 labeled instances. 
Meanwhile, 40,000 images in total are randomly selected as the unlabeled data set DU from all the 200 categories with different ratios of unseen classes.
'''