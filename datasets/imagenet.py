import os
import cv2
from copy import deepcopy
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict

from torchvision.datasets import ImageFolder
    
MISMATCH_CONFIG = {
    'imagenet': {'target_classes': list(range(0, 500)),
                 'unknown_classes': list(range(500,1000)),
                 'n_unlabeled': 500000,
                 }
}

def load_image_cv2(path: str):
    """Load image with OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dsize=(256,256), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 2:
        return cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), dsize=(256,256), interpolation=cv2.INTER_LINEAR)
    else:
        raise NotImplementedError

def load_imagenet(root: str, 
                  n_label_per_class: int,
                  mismatch_ratio,
                  random_state,
                  logger,
                  **kwargs):
    
    data_dir = os.path.join(root, 'full-imagenet')
    CONFIG = MISMATCH_CONFIG['imagenet']
    
    target_classes = CONFIG['target_classes']
    unknown_classes = CONFIG['unknown_classes']
    n_unlabels = CONFIG['n_unlabeled']

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    trainset = ImageFolder(train_dir)
    testset = ImageFolder(val_dir)
    
    images_inds_train, images_inds_valid, train_labels, val_labels = train_test_split(np.arange(len(trainset)),np.array(trainset.targets),test_size=0.1,stratify=trainset.targets,shuffle=True,random_state=random_state)

    train_dataset = {'images': images_inds_train, 'labels': train_labels}
    validation_dataset = {'images': images_inds_valid, 'labels': val_labels}

    # ✅ Directly access dataset without extra variable assignment
    train_images = train_dataset['images']
    train_labels = train_dataset['labels']

    # ✅ Preallocate NumPy arrays instead of using Python lists
    l_train_images = []
    l_train_labels = []
    u_train_images = []
    u_train_labels = []

    # ✅ Faster vectorized slicing for labeled/unlabeled data
    n_unlabels_per_cls = int(n_unlabels*(1.0-mismatch_ratio)) // len(target_classes)
    for c in tqdm(target_classes, desc='target_classes loading'):
        mask = train_labels == c
        indices = np.where(mask)[0]

        # ✅ Use NumPy slicing instead of `.extend()`
        l_train_images.append(train_images[indices[:n_label_per_class]])
        l_train_labels.append(train_labels[indices[:n_label_per_class]])

        u_train_images.append(train_images[indices[n_label_per_class:n_label_per_class + n_unlabels_per_cls]])
        u_train_labels.append(train_labels[indices[n_label_per_class:n_label_per_class + n_unlabels_per_cls]])

    # ✅ Faster processing of unknown classes
    n_unlabels_shifts = (n_unlabels - n_unlabels_per_cls*len(target_classes)) // len(unknown_classes)
    for c in tqdm(unknown_classes, desc='unknown classes loading'):
        mask = train_labels == c
        indices = np.where(mask)[0]

        u_train_images.append(train_images[indices[:n_unlabels_shifts]])
        u_train_labels.append(train_labels[indices[:n_unlabels_shifts]])

    # ✅ Use NumPy concatenation instead of converting lists later
    l_train_images = np.concatenate(l_train_images, axis=0)
    l_train_labels = np.concatenate(l_train_labels, axis=0)
    u_train_images = np.concatenate(u_train_images, axis=0)
    u_train_labels = np.concatenate(u_train_labels, axis=0)

    # ✅ Final datasets
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

    test_dataset = {'images': np.arange(len(testset)), 'labels': np.array(testset.targets)}

    test_total_dataset = deepcopy(test_dataset)

    target_indices = np.isin(test_dataset['labels'], target_classes)
    test_dataset = {'images': test_dataset['images'][target_indices], 'labels': test_dataset['labels'][target_indices]}

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
    
    return datasets, convert_dict, trainset, testset