import os
import sys
import time

import numpy as np
import rich
import torch

sys.path.append("./")
from configs import CaliMATCHConfig
from datasets.cifar import CIFAR, load_CIFAR, CIFAR_STRONG
from datasets.transforms import SemiAugment, TestAugment
from models import WRN, densenet121, vgg16_bn, inceptionv4
from tasks.classification_CaliMATCH import Classification
from utils.logging import get_rich_logger
from utils.wandb import configure_wandb
from utils.gpu import set_gpu

NUM_CLASSES = {
    'cifar10': 6,
    'cifar100': 50,
    'svhn': 6,
}

AUGMENTS = {
    'semi': SemiAugment,
    'test': TestAugment
}


def main():
    """Main function for single/distributed linear classification."""

    config = CaliMATCHConfig.parse_arguments()
    set_gpu(config)

    rich.print(config.__dict__)
    config.save()

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
        model = WRN(width=int(config.backbone_type.split("_")[-1]), num_classes=num_classes)
    elif config.backbone_type == 'densenet121':
        model = densenet121(num_class=num_classes) 
    elif config.backbone_type == 'vgg16_bn':
        model = vgg16_bn(num_class=num_classes)
    elif config.backbone_type == 'inceptionv4':
        model = inceptionv4(class_nums=num_classes)
    else:
        raise NotImplementedError
    
    # Network + Temperature (trainable)
    import torch.nn as nn
    setattr(model,'temperature', nn.Parameter(torch.ones(1) * 1.5))

    # Data (transforms & datasets)
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    train_trans = AUGMENTS[config.train_augment](**trans_kwargs)
    test_trans = AUGMENTS[config.test_augment](**trans_kwargs)

    if config.data in ['cifar10', 'cifar100']:

        datasets, _ = load_CIFAR(root=config.root,data_name=config.data,n_valid_per_class=config.n_valid_per_class,seed=config.seed, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio)

        labeled_set = CIFAR(data_name=config.data, dataset=datasets['l_train'], transform=train_trans)
        unlabeled_set = CIFAR_STRONG(data_name=config.data, dataset=datasets['u_train'], transform=train_trans)
        eval_set = CIFAR(data_name=config.data, dataset=datasets['validation'], transform=test_trans)
        test_set = CIFAR(data_name=config.data, dataset=datasets['test'], transform=test_trans)
        open_test_set = CIFAR(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)
    else:
        raise NotImplementedError

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, 'main.log')
        logger = get_rich_logger(logfile=logfile)
        if config.enable_wandb:
            configure_wandb(
                name=f'{config.task} : {config.hash}',
                project=f'SafeSSL-Calibration-{config.data}-{config.task}',
                config=config
            )
    else:
        logger = None

    if local_rank == 0:
        logger.info(f'Data: {config.data}')
        logger.info(f'Labeled Data Observations: {len(labeled_set):,}')
        logger.info(f'Unlabeled Data Observations: {len(unlabeled_set):,}')
        logger.info(f'Backbone: {config.backbone_type}')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')

    # Model (Task)
    model = Classification(backbone=model)
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        iterations=config.iterations,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        local_rank=local_rank,
        mixed_precision=config.mixed_precision,
        gamma = config.gamma,
        milestones= config.milestones,
        weight_decay=config.weight_decay
    )

    # Train & evaluate
    start = time.time()
    model.run(
        train_set=[labeled_set,unlabeled_set],
        eval_set=eval_set,
        test_set=test_set,
        open_test_set=open_test_set,
        save_every=config.save_every,
        tau=config.tau,
        consis_coef=config.consis_coef,
        warm_up_end=config.warm_up,
        n_bins=config.n_bins,
        train_n_bins=config.train_n_bins,
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
