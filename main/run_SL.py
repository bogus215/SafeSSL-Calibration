import os
import sys
import time

import numpy as np
import rich
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append("./")
from configs import SLConfig

from datasets.cifar import CIFAR, load_CIFAR
from datasets.tiny import TinyImageNet ,load_tiny
from datasets.svhn import SVHN, load_SVHN
from datasets.imagenet import load_imagenet
from datasets.transforms import SemiAugment, TestAugment

from models import WRN, densenet121, vgg16_bn, inceptionv4, ResNet50
from tasks.classification import Classification
from utils.logging import get_rich_logger
from utils.wandb import configure_wandb
from utils.gpu import set_gpu

NUM_CLASSES = {
    'cifar10': 6,
    'cifar100': 50,
    'svhn': 6,
    'tiny': 100,
    'imagenet': 500,
}

AUGMENTS = {
    'semi': SemiAugment,
    'test': TestAugment
}


def main():
    """Main function for single/distributed linear classification."""

    config = SLConfig.parse_arguments()
    set_gpu(config)
    num_gpus_per_node = len(config.gpus)
    world_size = config.num_nodes * num_gpus_per_node
    distributed = world_size > 1
    setattr(config, 'num_gpus_per_node', num_gpus_per_node)
    setattr(config, 'world_size', world_size)
    setattr(config, 'distributed', distributed)

    rich.print(config.__dict__)
    config.save()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if config.distributed:
        rich.print(f"Distributed training on {world_size} GPUs.")
        mp.spawn(
            main_worker,
            nprocs=config.num_gpus_per_node,
            args=(config, )
        )
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
            rank=dist_rank
        )

    config.batch_size = config.batch_size // config.world_size
    config.num_workers = config.num_workers // config.num_gpus_per_node

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
    elif config.backbone_type == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    else:
        raise NotImplementedError

    # create logger
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

    # Data (transforms & datasets)
    trans_kwargs = dict(size=config.input_size, data=config.data, impl=config.augmentation)
    train_trans = AUGMENTS[config.train_augment](**trans_kwargs)
    test_trans = AUGMENTS[config.test_augment](**trans_kwargs)

    if config.data in ['cifar10', 'cifar100']:

        datasets, _ = load_CIFAR(root=config.root,data_name=config.data,n_valid_per_class=config.n_valid_per_class,seed=config.seed, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio, logger=logger)

        labeled_set = CIFAR(data_name=config.data, dataset=datasets['l_train'], transform=train_trans)
        unlabeled_set = CIFAR(data_name=config.data, dataset=datasets['u_train'], transform=train_trans)
        eval_set = CIFAR(data_name=config.data, dataset=datasets['validation'], transform=test_trans)
        test_set = CIFAR(data_name=config.data, dataset=datasets['test'], transform=test_trans)
        open_test_set = CIFAR(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)
    
    elif config.data == 'tiny':
        
        datasets, _ = load_tiny(root=config.root, n_label_per_class=config.n_label_per_class,n_valid_per_class=config.n_valid_per_class,mismatch_ratio=config.mismatch_ratio,random_state=config.seed,logger=logger)

        labeled_set = TinyImageNet(data_name=config.data, dataset=datasets['l_train'], transform=train_trans)
        unlabeled_set = TinyImageNet(data_name=config.data, dataset=datasets['u_train'], transform=train_trans)
        eval_set = TinyImageNet(data_name=config.data, dataset=datasets['validation'], transform=test_trans)
        test_set = TinyImageNet(data_name=config.data, dataset=datasets['test'], transform=test_trans)
        open_test_set = TinyImageNet(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)
        
    elif config.data == 'svhn':
        
        datasets, _ = load_SVHN(root=config.root, data_name=config.data, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio,random_state=config.seed,logger=logger)

        labeled_set = SVHN(data_name=config.data, dataset=datasets['l_train'], transform=train_trans)
        unlabeled_set = SVHN(data_name=config.data, dataset=datasets['u_train'], transform=train_trans)
        eval_set = SVHN(data_name=config.data, dataset=datasets['validation'], transform=test_trans)
        test_set = SVHN(data_name=config.data, dataset=datasets['test'], transform=test_trans)
        open_test_set = SVHN(data_name=config.data, dataset=datasets['test_total'], transform=test_trans)
    
    elif config.data == 'imagenet':
        
        datasets, _, trainset, testset = load_imagenet(root=config.root, n_label_per_class=config.n_label_per_class, mismatch_ratio=config.mismatch_ratio, random_state=config.seed, logger=logger)

        from ffcv.writer import DatasetWriter
        from ffcv.fields import IntField, RGBImageField
        from torch.utils.data import Subset
        
        for mode in ['l_train','u_train','validation']:
            ffcv_file_path = os.path.join(config.root,'full-imagenet','train',f'{mode}.ffcv')
            if not os.path.exists(ffcv_file_path):
                writer = DatasetWriter(ffcv_file_path,{'image': RGBImageField(write_mode="proportion",max_resolution=500,compress_probability=0.5,jpeg_quality=90),'label': IntField()}, num_workers=config.num_workers)
                writer.from_indexed_dataset(Subset(trainset, datasets[mode]['images']), chunksize=100)

        for mode in ['test','test_total']:
            ffcv_file_path = os.path.join(config.root,'full-imagenet','val',f'{mode}.ffcv')
            if not os.path.exists(ffcv_file_path):
                writer = DatasetWriter(ffcv_file_path,{'image': RGBImageField(write_mode="proportion",max_resolution=500,compress_probability=0.5,jpeg_quality=90),'label': IntField()}, num_workers=config.num_workers)
                writer.from_indexed_dataset(Subset(testset, datasets[mode]['images']), chunksize=100)

        labeled_set = os.path.join(config.root,'full-imagenet','train','l_train.ffcv')
        unlabeled_set = os.path.join(config.root,'full-imagenet','train','u_train.ffcv')
        eval_set = os.path.join(config.root,'full-imagenet','train','validation.ffcv')
        test_set = os.path.join(config.root,'full-imagenet','val','test.ffcv')
        open_test_set = os.path.join(config.root,'full-imagenet','val','test_total.ffcv')
        
        from tasks.classification import ImageNetClassification

    else:
        raise NotImplementedError

    if local_rank == 0:
        logger.info(f'Data: {config.data}')
        logger.info(f"Labeled Data Observations: {len(datasets['l_train']['labels']):,}")
        logger.info(f"Unlabeled Data Observations: {len(datasets['u_train']['labels']):,}")
        logger.info(f'Backbone: {config.backbone_type}')
        logger.info(f'Checkpoint directory: {config.checkpoint_dir}')

    # Model (Task)
    model = Classification(backbone=model) if config.data != 'imagenet' else ImageNetClassification(backbone=model)
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
        n_bins=config.n_bins,
        enable_plot=config.enable_plot,
        distributed=config.distributed,
        warm_up=config.warm_up,
        logger=logger
    )
    elapsed_sec = time.time() - start

    if logger is not None:
        elapsed_mins = elapsed_sec / 60
        logger.info(f'Total training time: {elapsed_mins:,.2f} minutes.')
        logger.handlers.clear()

    if config.distributed:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"[WARNING] dist.barrier() timed out: {e}")

            if dist.is_initialized():
                dist.destroy_process_group()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
