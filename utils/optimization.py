# -*- coding: utf-8 -*-

import torch.optim as optim
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts

def get_optimizer(params, name: str, lr: float, weight_decay: float = 0.00, **kwargs):
    """Returns an `Optimizer` object given proper arguments."""

    if name == 'adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif name == 'lookahead':
        raise NotImplementedError
    else:
        raise NotImplementedError

def get_multi_step_scheduler(optimizer: optim.Optimizer, milestones: list, gamma: float = 0.1):
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

def get_cosine_anneal_scheduler(optimizer: optim.Optimizer, milestones: int ,gamma: float = 0.1):
    return CosineAnnealingWarmRestarts(optimizer, T_0=milestones,eta_min=optimizer.param_groups[0]['lr']*gamma)