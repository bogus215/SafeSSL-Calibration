import os
import sys
import time

import numpy as np
import rich
import torch
import torch.nn as nn

sys.path.append("./")
from configs import TestingConfig
from datasets.cifar import CIFAR, load_CIFAR
from datasets.transforms import SemiAugment, TestAugment
from datasets.tiny import TinyImageNet ,load_tiny
from datasets.svhn import SVHN, load_SVHN
from models import WRN, LULClassifier
from tasks.testing import Testing
from utils.gpu import set_gpu
from utils.logging import get_rich_logger

NUM_CLASSES = {
    'cifar10': 6,
    'cifar100': 50,
    'svhn': 6,
    'tiny': 100,
}

AUGMENTS = {
    'semi': SemiAugment,
    'test': TestAugment
}

def main():

    """Main function for single/distributed linear classification."""

    config = TestingConfig.parse_arguments()
    set_gpu(config)

    rich.print(config.__dict__)
    config.save(os.path.join(config.checkpoint_dir,"configs.json"))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    rich.print("Single GPU training.")
    main_worker(0, config=config)  # single machine, single gpu

def main_worker(local_rank: int, config: object):
    """Single process."""

    torch.cuda.set_device(local_rank)
    num_classes = NUM_CLASSES[config.data]

    # Networks
    if config.backbone_type in ['wide28_10',"wide28_2"]:
        model = WRN(width=int(config.backbone_type.split("_")[-1]), num_classes=num_classes, normalize=config.normalize)
    else:
        raise NotImplementedError

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'testing.log')
        logger = get_rich_logger(logfile=logfile)
    else:
        logger = None

    # Data (transforms & datasets)
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    test_trans = AUGMENTS[config.test_augment](**trans_kwargs)

    if config.for_what=="Proposed":
        setattr(model,'mlp', LULClassifier(model.output.in_features, size=config.layer_size, lambda_weight=0))
        setattr(model,'cali_scaler', nn.Parameter(torch.ones(1) * 1.5))
    elif config.for_what=='IOMATCH':
        setattr(model,'mlp_proj', nn.Sequential(nn.Linear(model.output.in_features,model.output.in_features),nn.ReLU(),nn.Linear(model.output.in_features,model.output.in_features)))
        setattr(model,'mb_classifiers', nn.Linear(model.output.in_features,int(model.class_num*2)))
        setattr(model,'openset_classifier', nn.Linear(model.output.in_features,int(model.class_num+1)))
    elif config.for_what=='OPENMATCH':
        setattr(model,'ova_classifiers', nn.Linear(model.output.in_features,int(model.class_num*2), bias=False))
    else:
        pass

    if config.data in ['cifar10', 'cifar100']:

        datasets, _ = load_CIFAR(root=config.root,data_name=config.data,n_valid_per_class=config.n_valid_per_class,seed=config.seed, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio, logger=logger)
        open_test_set = CIFAR(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)

    elif config.data == 'tiny':
        
        datasets, _ = load_tiny(root=config.root, n_label_per_class=config.n_label_per_class,n_valid_per_class=config.n_valid_per_class,mismatch_ratio=config.mismatch_ratio,random_state=config.seed,logger=logger)
        open_test_set = TinyImageNet(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)

    elif config.data == 'svhn':
        datasets, _ = load_SVHN(root=config.root, data_name=config.data, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio,random_state=config.seed,logger=logger)
        open_test_set = SVHN(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)

    else:
        raise NotImplementedError

    # Model (Task)
    model = Testing(backbone=model)
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        model_ckpt_dir=config.checkpoint_dir.replace(f"{config.for_what}/","").replace("Testing",config.for_what),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        local_rank=local_rank
    )

    # Evaluate
    start = time.time()
    model.run(
        for_what=config.for_what,
        open_test_set=open_test_set,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)