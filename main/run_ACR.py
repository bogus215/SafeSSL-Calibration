import os
import sys
import time

import numpy as np
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append("./")
from configs import ACRConfig
from datasets.cifar import CIFAR, load_CIFAR
from datasets.svhn import ACR, SVHN, load_SVHN
from datasets.tiny import TinyImageNet, load_tiny
from datasets.transforms import SemiAugment, TestAugment
from models import WRN, ResNet50, ViT
from tasks.classification_ACR import Classification
from utils.gpu import set_gpu
from utils.initialization import initialize_weights
from utils.logging import get_rich_logger
from utils.wandb import configure_wandb

NUM_CLASSES = {
    "cifar10": 6,
    "cifar100": 50,
    "svhn": 6,
    "tiny": 100,
    "imagenet": 500,
}

AUGMENTS = {"semi": SemiAugment, "test": TestAugment}


def main():
    """Main function for single/distributed linear classification."""

    config = ACRConfig.parse_arguments()
    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, "num_gpus_per_node", num_gpus_per_node)
    setattr(config, "world_size", world_size)
    setattr(config, "distributed", distributed)

    rich.print(config.__dict__)
    config.save()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if config.distributed:
        rich.print(f"Distributed training on {world_size} GPUs.")
        mp.spawn(main_worker, nprocs=config.num_gpus_per_node, args=(config,))
    else:
        rich.print("Single GPU training.")
        main_worker(0, config=config)  # single machine, single gpu


def main_worker(local_rank: int, config: object):
    """Single process."""

    torch.cuda.set_device(local_rank)
    if config.distributed:
        dist_rank = config.node_rank * config.num_gpus_per_node + local_rank
        dist.init_process_group(
            backend=config.dist_backend,
            init_method=config.dist_url,
            world_size=config.world_size,
            rank=dist_rank,
        )

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

    num_classes = NUM_CLASSES[config.data]

    # Networks
    if config.backbone_type in ["wide28_10", "wide28_2"]:
        model = WRN(
            width=int(config.backbone_type.split("_")[-1]),
            num_classes=num_classes,
            normalize=config.normalize,
        )
    elif config.backbone_type == "resnet50":
        model = ResNet50(num_classes=num_classes, normalize=config.normalize)
    elif config.backbone_type == "vit":
        model = ViT(num_classes=num_classes, normalize=config.normalize)
    else:
        raise NotImplementedError

    # create logger
    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, "main.log")
        logger = get_rich_logger(logfile=logfile)
        if config.enable_wandb:
            configure_wandb(
                name=f"{config.task} : {config.hash}",
                project=f"SafeSSL-Calibration-{config.data}-{config.task}",
                config=config,
            )
    else:
        logger = None

    # Sub-Network Plus
    import torch.nn as nn
    setattr(
        model, "fc1", nn.Linear(model.in_features, model.class_num, bias=False)
    )
    initialize_weights(model)

    # Data (transforms & datasets)
    trans_kwargs = dict(
        size=config.input_size, data=config.data, impl=config.augmentation
    )
    train_trans = AUGMENTS[config.train_augment](**trans_kwargs)
    test_trans = AUGMENTS[config.test_augment](**trans_kwargs)

    if config.data in ["cifar10", "cifar100"]:

        datasets, _ = load_CIFAR(
            root=config.root,
            data_name=config.data,
            n_valid_per_class=config.n_valid_per_class,
            seed=config.seed,
            n_label_per_class=config.n_label_per_class,
            mismatch_ratio=config.mismatch_ratio,
            input_size=config.input_size,
            logger=logger,
        )

        labeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=None,
            transform=train_trans,
            name="lb",
        )
        unlabeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=datasets["u_train"],
            name="ulb",
            transform=train_trans,
        )

        eval_set = CIFAR(
            data_name=config.data, dataset=datasets["validation"], transform=test_trans
        )
        test_set = CIFAR(
            data_name=config.data, dataset=datasets["test"], transform=test_trans
        )

    elif config.data == "tiny":

        datasets, _ = load_tiny(
            root=config.root,
            n_label_per_class=config.n_label_per_class,
            n_valid_per_class=config.n_valid_per_class,
            mismatch_ratio=config.mismatch_ratio,
            random_state=config.seed,
            logger=logger,
        )

        labeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=None,
            transform=train_trans,
            name="lb",
        )
        unlabeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=datasets["u_train"],
            name="ulb",
            transform=train_trans,
        )

        eval_set = TinyImageNet(
            data_name=config.data, dataset=datasets["validation"], transform=test_trans
        )
        test_set = TinyImageNet(
            data_name=config.data, dataset=datasets["test"], transform=test_trans
        )

    elif config.data == "svhn":

        datasets, _ = load_SVHN(
            root=config.root,
            data_name=config.data,
            n_label_per_class=config.n_label_per_class,
            mismatch_ratio=config.mismatch_ratio,
            random_state=config.seed,
            logger=logger,
        )

        labeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=None,
            transform=train_trans,
            name="lb",
        )
        unlabeled_set = ACR(
            labelset=datasets["l_train"],
            unlabelset=datasets["u_train"],
            name="ulb",
            transform=train_trans,
        )

        eval_set = SVHN(
            data_name=config.data, dataset=datasets["validation"], transform=test_trans
        )
        test_set = SVHN(
            data_name=config.data, dataset=datasets["test"], transform=test_trans
        )

    elif config.data == "imagenet":
        raise NotImplementedError
    else:
        raise NotImplementedError

    if local_rank == 0:
        logger.info(f"Data: {config.data}")
        logger.info(
            f"Labeled Data Observations: {len(datasets['l_train']['labels']):,}"
        )
        logger.info(
            f"Unlabeled Data Observations: {len(datasets['u_train']['labels']):,}"
        )
        logger.info(f"Backbone: {config.backbone_type}")
        logger.info(f"Checkpoint directory: {config.checkpoint_dir}")

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
        gamma=config.gamma,
        milestones=config.milestones,
        weight_decay=config.weight_decay,
    )

    # Train & evaluate
    start = time.time()
    model.run(
        train_set=[labeled_set, unlabeled_set],
        eval_set=eval_set,
        test_set=test_set,
        n_bins=config.n_bins,
        save_every=config.save_every,
        start_fix=config.start_fix,
        threshold=config.threshold,
        T=config.T,
        tau1=config.tau1,
        tau12=config.tau12,
        tau2=config.tau2,
        ema_u=config.ema_u,
        est_epoch=config.est_epoch,
        enable_plot=config.enable_plot,
        distributed=config.distributed,
        logger=logger,
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f"Total training time: {elapsed_mins:,.2f} minutes.")
        logger.handlers.clear()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
