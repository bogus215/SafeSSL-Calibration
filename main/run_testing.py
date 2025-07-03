import os
import sys
import time

import numpy as np
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.append("./")
from configs import TestingConfig
from datasets.cifar import CIFAR, load_CIFAR
from datasets.imagenet import load_imagenet
from datasets.svhn import SVHN, load_SVHN
from datasets.tiny import TinyImageNet, load_tiny
from datasets.transforms import SemiAugment, TestAugment
from models import WRN, ResNet50, densenet121, inceptionv4, vgg16_bn
from tasks.testing import Testing
from utils.gpu import set_gpu
from utils.logging import get_rich_logger

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

    config = TestingConfig.parse_arguments()
    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, "num_gpus_per_node", num_gpus_per_node)
    setattr(config, "world_size", world_size)
    setattr(config, "distributed", distributed)

    rich.print(config.__dict__)
    config.save(os.path.join(config.checkpoint_dir, "configs.json"))

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
    else:
        raise NotImplementedError

    if local_rank == 0:
        logfile = os.path.join(config.checkpoint_dir, "testing.log")
        logger = get_rich_logger(logfile=logfile)
    else:
        logger = None

    # Data (transforms & datasets)
    trans_kwargs = dict(
        size=config.input_size, data=config.data, impl=config.augmentation
    )
    test_trans = AUGMENTS[config.test_augment](**trans_kwargs)

    if config.for_what == "Proposed":
        setattr(model, "cali_scaler", nn.Parameter(torch.ones(1) * 1.5))
        setattr(model, "ova_cali_scaler", nn.Parameter(torch.ones(1) * 1.5))
        setattr(
            model,
            "ova_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2), bias=False),
        )
    elif config.for_what == "IOMATCH":
        setattr(
            model,
            "mlp_proj",
            nn.Sequential(
                nn.Linear(model.output.in_features, model.output.in_features),
                nn.ReLU(),
                nn.Linear(model.output.in_features, model.output.in_features),
            ),
        )
        setattr(
            model,
            "mb_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2)),
        )
        setattr(
            model,
            "openset_classifier",
            nn.Linear(model.output.in_features, int(model.class_num + 1)),
        )
    elif "OPENMATCH" in config.for_what:
        setattr(
            model,
            "ova_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2), bias=False),
        )
    elif config.for_what == "MTC":
        setattr(model, "domain_classifier", nn.Linear(model.output.in_features, 1))
    elif config.for_what in ["Ablation1", "Ablation4"]:
        setattr(model, "ova_cali_scaler", nn.Parameter(torch.ones(1) * 1.5))
        setattr(
            model,
            "ova_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2), bias=False),
        )
    elif config.for_what in ["Ablation2", "Ablation5"]:
        setattr(model, "cali_scaler", nn.Parameter(torch.ones(1) * 1.5))
        setattr(
            model,
            "ova_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2), bias=False),
        )
    elif config.for_what == "Ablation3":
        setattr(
            model,
            "ova_classifiers",
            nn.Linear(model.output.in_features, int(model.class_num * 2), bias=False),
        )
    else:
        pass

    if config.data in ["cifar10", "cifar100"]:

        datasets, _ = load_CIFAR(
            root=config.root,
            data_name=config.data,
            n_valid_per_class=config.n_valid_per_class,
            seed=config.seed,
            n_label_per_class=config.n_label_per_class,
            mismatch_ratio=config.mismatch_ratio,
            logger=logger,
        )
        open_test_set = CIFAR(
            data_name=config.data, dataset=datasets["test_total"], transform=test_trans
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
        open_test_set = TinyImageNet(
            data_name=config.data, dataset=datasets["test_total"], transform=test_trans
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
        open_test_set = SVHN(
            data_name=config.data, dataset=datasets["test_total"], transform=test_trans
        )

    elif config.data == "imagenet":
        datasets, _, _, testset = load_imagenet(
            root=config.root,
            n_label_per_class=config.n_label_per_class,
            mismatch_ratio=config.mismatch_ratio,
            random_state=config.seed,
            logger=logger,
        )

        from ffcv.fields import IntField, RGBImageField
        from ffcv.writer import DatasetWriter
        from torch.utils.data import Subset

        ffcv_file_path = os.path.join(
            config.root, "full-imagenet", "val", "test_total.ffcv"
        )
        if not os.path.exists(ffcv_file_path):
            writer = DatasetWriter(
                ffcv_file_path,
                {
                    "image": RGBImageField(
                        write_mode="proportion",
                        max_resolution=500,
                        compress_probability=0.5,
                        jpeg_quality=90,
                    ),
                    "label": IntField(),
                },
                num_workers=config.num_workers,
            )
            writer.from_indexed_dataset(
                Subset(testset, datasets["test_total"]["images"]), chunksize=100
            )
        open_test_set = os.path.join(
            config.root, "full-imagenet", "val", "test_total.ffcv"
        )
        from tasks.testing import ImageNetTesting
    else:
        raise NotImplementedError

    # Model (Task)
    model = (
        Testing(backbone=model)
        if config.data != "imagenet"
        else ImageNetTesting(backbone=model)
    )
    model.prepare(
        ckpt_dir=config.checkpoint_dir,
        model_ckpt_dir=config.checkpoint_dir.replace(f"{config.for_what}/", "").replace(
            "Testing", config.for_what
        ),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        local_rank=local_rank,
    )

    # Evaluate
    start = time.time()
    model.run(
        for_what=config.for_what,
        open_test_set=open_test_set,
        mismatch_ratio=config.mismatch_ratio,
        safe_student_T=config.safe_student_T,
        ova_pi=config.ova_pi,
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
