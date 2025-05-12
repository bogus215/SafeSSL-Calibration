import collections
import os, math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from rich.progress import Progress
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from tasks.classification import Classification as Task
from utils import TopKAccuracy
from utils.logging import make_epoch_description


class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            n_bins,
            save_every,
            start_fix,
            lambda_ova_u,
            ova_unlabeled_threshold,
            lambda_x,
            lambda_ova,
            lambda_oem,
            lambda_socr,
            lambda_u,
            T,
            threshold,
            **kwargs):  # pylint: disable=unused-argument

        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader
        epochs = self.iterations // save_every
        per_epoch_steps = self.iterations // epochs
        num_samples = per_epoch_steps * self.batch_size // 2 

        l_sampler = DistributedSampler(dataset=train_set[0], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        l_loader = DataLoader(train_set[0],batch_size=self.batch_size//2, sampler=l_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)

        u_sampler = DistributedSampler(dataset=train_set[1], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        unlabel_loader = DataLoader(train_set[1],batch_size=self.batch_size//2, sampler=u_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)

        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

        # Logging
        logger = kwargs.get('logger', None)
        enable_plot = kwargs.get('enable_plot',False)

        # Supervised training
        best_eval_acc = -float('inf')
        best_epoch    = 0

        for epoch in range(1, epochs + 1):

            train_history = self.train(label_loader=l_loader,
                                       unlabel_loader=unlabel_loader,
                                       current_epoch=epoch,
                                       start_fix=start_fix,
                                       n_bins=n_bins,
                                       lambda_ova_u=lambda_ova_u,
                                       ova_unlabeled_threshold=ova_unlabeled_threshold,
                                       lambda_x=lambda_x,
                                       lambda_ova=lambda_ova,
                                       lambda_oem=lambda_oem,
                                       lambda_socr=lambda_socr,
                                       lambda_u=lambda_u,
                                       T=T,
                                       threshold=threshold,
                                       )
            eval_history = self.evaluate(eval_loader, n_bins)
            if enable_plot:
                raise NotImplementedError

            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]['train'] = v1
                try:
                    v2 = eval_history[k]
                    epoch_history[k]['eval'] = v2
                except KeyError:
                    continue

            # Write TensorBoard summary
            if self.writer is not None:
                for k, v in epoch_history.items():
                    for k_, v_ in v.items():
                        self.writer.add_scalar(f'{k}_{k_}', v_, global_step=epoch)
                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('lr', lr, global_step=epoch)

            # Save best model checkpoint and Logging
            eval_acc = eval_history['top@1']

            if logger is not None and eval_acc==1:
                logger.info("Eval acc == 1 --> Stop training")
                break

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader,n_bins)
                for k, v1 in test_history.items():
                    epoch_history[k]['test'] = v1

                if self.writer is not None:
                    self.writer.add_scalar('Best_Test_top@1', test_history['top@1'], global_step=epoch)

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

    def train(self, label_loader, unlabel_loader, current_epoch, start_fix, n_bins,
                    lambda_ova_u,
                    ova_unlabeled_threshold,
                    lambda_x,
                    lambda_ova,
                    lambda_oem,
                    lambda_socr,
                    lambda_u,
                    T,
                    threshold
                    ):

        """Training defined for a single epoch."""

        iteration = len(unlabel_loader)
        
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
        }
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb) in enumerate(zip(label_loader, unlabel_loader)):

                with torch.cuda.amp.autocast(self.mixed_precision):

                    inputs_x_w = data_lb["inputs_x_w"]
                    inputs_x_s = data_lb["inputs_x_s"]
                    inputs_x_s2 = data_lb["inputs_x_s2"]
                    inputs_x = data_lb["inputs_x"]
                    targets_x = data_lb["targets_x"]
                    
                    inputs_u_w  = data_ulb["inputs_u_w"]
                    inputs_u_s = data_ulb["inputs_u_s"]
                    targets_u_gt = data_ulb["targets_u_gt"]

                    inputs_all_w  = data_ulb["inputs_all_w"]
                    inputs_all_s = data_ulb["inputs_all_s"]
                    inputs_all_s2 = data_ulb["inputs_all_s2"]
                    inputs_all = data_ulb["inputs_all"]
                    targets_all_u = data_ulb["targets_all_u"]
                    targets_all_u[targets_all_u >= self.backbone.class_num] = self.backbone.class_num

                    b_size = inputs_x.shape[0]  # 64
                    inputs = torch.cat([inputs_x_w, inputs_x, inputs_x_s, inputs_x_s2,
                                        inputs_all_w, inputs_all, inputs_all_s, inputs_all_s2], 0).to(self.local_rank)
                    targets_x = targets_x.to(self.local_rank)

                    logits, logits_open = self.ssb_predict(inputs)
                    logits_open_u1, logits_open_u2, logits_open_s1, logits_open_s2 = logits_open[4*b_size:].chunk(4)

                    Lx = F.cross_entropy(logits[:2*b_size], targets_x.repeat(2), reduction='mean')
                    Lo = ova_loss(logits_open[:2*b_size], logits_open[2*b_size:4*b_size], targets_x.repeat(2), targets_x.repeat(2))
                                     
                    # unlabeled OVA loss starts
                    if current_epoch >= start_fix and lambda_ova_u != 0:
                        with torch.no_grad():
                            logits_open_w = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                            logits_open_w = F.softmax(logits_open_w, 1)
                            know_score_w = logits_open_w[:, 1, :]  # [bs, num_class]

                            neg_mask = (know_score_w <= ova_unlabeled_threshold).float()  # [bs, num_class]
                            tmp = torch.zeros((neg_mask.size(0), neg_mask.size(1) + 1))  # [bs, num_class]
                            tmp.scatter_(1, targets_all_u.view(-1, 1), 1)
                            gt_mask = (1 - tmp).float()
                            gt_mask = gt_mask[:, :-1]

                            if neg_mask.cpu().view(-1).sum() != 0:
                                prec = ((neg_mask.cpu() == gt_mask) * neg_mask.cpu()).view(-1).sum() / neg_mask.cpu().view(
                                    -1).sum()
                        Lo_u = unlabeled_ova_neg_loss(logits_open_u1, logits_open_u2, logits_open_s1, logits_open_s2, neg_mask)
                    else:
                        Lo_u = torch.zeros(1).to(self.local_rank).mean()
                    # unlabeled OVA loss ends

                    ## Open-set entropy minimization
                    L_oem = ova_ent(logits_open_u1) / 2.
                    L_oem += ova_ent(logits_open_u2) / 2.

                    ## Soft consistenty regularization
                    logits_open_u1_ = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                    logits_open_u2_ = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
                    logits_open_u1_ = F.softmax(logits_open_u1_, 1)
                    logits_open_u2_ = F.softmax(logits_open_u2_, 1)
                    L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
                        logits_open_u1_ - logits_open_u2_)**2, 1), 1))

                    if current_epoch >= start_fix:
                        inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(self.local_rank)
                        # logits, logits_open_fix, features_fix = model(inputs_ws, feature=True)  # [256, 55], [256, 110]
                        logits, logits_open_fix = self.ssb_predict(inputs_ws)  # [256, 55], [256, 110]
                        logits_u_w, logits_u_s = logits.chunk(2)
                        # features_u_w, features_u_s = features_fix.chunk(2)
                        pseudo_label = torch.softmax(logits_u_w.detach()/T, dim=-1)
                        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(threshold).float()
                        L_fix = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                        total_acc = targets_u.cpu().eq(targets_u_gt).float().view(-1)
                        if mask.sum() != 0:
                            tmp = (targets_u_gt[mask.cpu() != 0] == self.backbone.class_num).float()

                        pred_id = mask.cpu().numpy()  # 1 for ID data
                        targets_id = (targets_u_gt < self.backbone.class_num).float().cpu().numpy()  # ID index
                    else:
                        L_fix = torch.zeros(1).to(self.local_rank).mean()

                    loss = lambda_x * Lx + lambda_ova * Lo + lambda_oem * L_oem  \
                        + lambda_socr * L_socr + lambda_u * L_fix + lambda_ova_u * Lo_u

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits[:2*b_size], targets_x.repeat(2)).detach()
                result['ece'][i] = self.get_ece(preds=logits[:2*b_size].softmax(dim=1).detach().cpu().numpy(), targets=targets_x.repeat(2).cpu().numpy(), n_bins=n_bins, plot=False)[0]

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}
    
    def ssb_predict(self, x: torch.FloatTensor):

        logits, feat = self.get_feature(x)
        
        return logits, self.backbone.fc_open(feat)

    def get_feature(self, x: torch.FloatTensor):
        """Make a prediction provided a batch of samples."""
        return self.backbone(x, return_feature=True)

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, num_samples=None):


        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = num_samples
        assert num_samples % self.num_replicas == 0, f'{num_samples} samples cant' \
                                                     f'be evenly distributed among {num_replicas} devices.'
        self.num_samples = int(num_samples // self.num_replicas)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n
        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
class ImageNetClassification(Classification):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)
        
    def run(self,
            train_set,
            eval_set,
            test_set,
            open_test_set,
            p_cutoff,
            pi,
            warm_up_end,
            save_every,
            n_bins,
            start_fix,
            lambda_em,
            lambda_socr,
            train_trans,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        distributed = kwargs.get('distributed')
        
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Squeeze, RandomHorizontalFlip, RandomColorJitter, RandomGrayscale, RandomSolarization, Cutout
        from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
        
        label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True)]
        img_pipeline_weak = [RandomResizedCropRGBImageDecoder((192, 192)), RandomHorizontalFlip(), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD, np.float16)]
        img_pipeline_strong = [RandomResizedCropRGBImageDecoder((192, 192)), RandomHorizontalFlip(), RandomColorJitter(0.8,0.4,0.4,0.2,0.1), RandomGrayscale(0.2), RandomSolarization(0.2,128), Cutout(crop_size=50), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD, np.float16)]
        img_pipeline_eval = [CenterCropRGBImageDecoder((224, 224),DEFAULT_CROP_RATIO), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD, np.float16)]

        # DataLoader (train, val, test)
        label_loader = Loader(train_set[0],batch_size=batch_size//4,order=OrderOption.RANDOM,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_weak,'label':label_pipeline})
        unlabel_loader = Loader(train_set[1],batch_size=batch_size,order=OrderOption.RANDOM,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_weak,'label':label_pipeline,'image_0':img_pipeline_weak,'image_1':img_pipeline_strong},custom_field_mapper={'image_0':'image','image_1':'image'})
        eval_loader = Loader(eval_set,batch_size=batch_size,order=OrderOption.SEQUENTIAL,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_eval,'label':label_pipeline})
        test_loader = Loader(test_set,batch_size=batch_size,order=OrderOption.SEQUENTIAL,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_eval,'label':label_pipeline})

        # Logging
        logger = kwargs.get('logger', None)

        # Supervised training
        best_eval_acc = -float('inf')
        best_epoch    = 0

        epochs = self.iterations // save_every
        self.warm_up_end = (warm_up_end // save_every) * len(unlabel_loader)
        self.trained_iteration = 0

        import torchlars
        self.optimizer = torchlars.LARS(self.optimizer)
        del self.scheduler

        for epoch in range(1, epochs + 1):

            # Train & evaluate
            train_history, cls_wise_results = self.train(label_loader, unlabel_loader, current_epoch=epoch, start_fix=start_fix, pi=pi, p_cutoff=p_cutoff, n_bins=n_bins, lambda_em=lambda_em,lambda_socr=lambda_socr)
            eval_history = self.evaluate(eval_loader, n_bins)

            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]['train'] = v1
                try:
                    v2 = eval_history[k]
                    epoch_history[k]['eval'] = v2
                except KeyError:
                    continue

            # Write TensorBoard summary
            if self.writer is not None:
                for k, v in epoch_history.items():
                    for k_, v_ in v.items():
                        self.writer.add_scalar(f'{k}_{k_}', v_, global_step=epoch)
                if cls_wise_results is not None:
                    self.writer.add_scalar("trained_unlabeled_data_in", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key<500]).item() , global_step=epoch)
                    self.writer.add_scalar("trained_unlabeled_data_ood", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key>=500]).item() , global_step=epoch)

            # Save best model checkpoint and Logging
            eval_acc = eval_history['top@1']
            if eval_acc==1:
                if logger is not None:
                    logger.info("Eval acc == 1 --> Stop training")
                break

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader,n_bins)
                for k, v1 in test_history.items():
                    epoch_history[k]['test'] = v1

                if self.writer is not None:
                    self.writer.add_scalar('Best_Test_top@1', test_history['top@1'], global_step=epoch)

            if epoch in [60, 120, 160]:
                self.learning_rate *= 0.1

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

    def train(self, label_loader, unlabel_loader, current_epoch, start_fix, pi, p_cutoff, n_bins, lambda_em,lambda_socr):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        iteration=len(unlabel_loader)
        result = {
            'loss': torch.zeros(iteration),
            'top@1': torch.zeros(iteration),
            'top@5': torch.zeros(iteration),
            'ece': torch.zeros(iteration),
            'unlabeled_top@1': torch.zeros(iteration),
            'unlabeled_top@5': torch.zeros(iteration),
            'unlabeled_ece': torch.zeros(iteration),
            'warm_up_lr': torch.zeros(iteration),
            'N_used_unlabeled': torch.zeros(iteration)
        }
        
        label_iterator = iter(label_loader)
        cls_wise_results = {i:torch.zeros(iteration) for i in range(1000)}
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (x_ulb_w, unlabel_y, x_ulb_w_1, x_ulb_s) in enumerate(unlabel_loader):

                warm_up_lr = self.learning_rate*math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warm_up_lr
                try:
                    l_batch = next(label_iterator)
                except:
                    label_iterator = iter(label_loader)
                    l_batch = next(label_iterator)

                x_lb_w_0, y_lb = l_batch[0], l_batch[1]
                num_lb = y_lb.shape[0]
                with torch.autocast('cuda', enabled = self.mixed_precision):

                    inputs = torch.cat((x_lb_w_0, x_ulb_w, x_ulb_w_1))
                    outputs = self.openmatch_predict(inputs)

                    logits_x_lb = outputs['logits'][:num_lb]
                    sup_loss = self.loss_function(logits_x_lb, y_lb)

                    logits_open_lb = outputs['logits_open'][:num_lb]
                    ova_loss = ova_loss_func(logits_open_lb, y_lb)

                    logits_open_ulb_0, logits_open_ulb_1 = outputs['logits_open'][num_lb:].chunk(2)

                    em_loss = em_loss_func(logits_open_ulb_0, logits_open_ulb_1)
                    socr_loss = socr_loss_func(logits_open_ulb_0, logits_open_ulb_1)

                    fix_loss = torch.tensor(0).cuda(self.local_rank)
                    if current_epoch >= start_fix:

                        logits_x_ulb_s = self.predict(x_ulb_s)
                        with torch.no_grad():
                            logits_x_ulb_w, _ = outputs['logits'][num_lb:].chunk(2)
                            unlabel_confidence, unlabel_pseudo_y = logits_x_ulb_w.softmax(1).max(1)
                            logits_open = nn.functional.softmax(logits_open_ulb_0.view(logits_open_ulb_0.size(0), 2, -1), 1)
                            tmp_range = torch.arange(0, logits_open.size(0)).long().cuda(self.local_rank)
                            unk_score = logits_open[tmp_range, :, unlabel_pseudo_y]
                            s_us_confidence, s_us_result = unk_score.max(1)
                            used_unlabeled_index = (unlabel_confidence>p_cutoff) & (s_us_confidence>=pi) & (s_us_result==1)
                        fix_loss = self.loss_function(logits_x_ulb_s[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())

                    loss = sup_loss + ova_loss + lambda_em * em_loss + lambda_socr * socr_loss + fix_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.trained_iteration+=1

                result['loss'][i] = loss.item()
                result['top@1'][i] = TopKAccuracy(k=1)(logits_x_lb, y_lb).item()
                result['top@5'][i] = TopKAccuracy(k=5)(logits_x_lb, y_lb).item()
                result['ece'][i] = self.get_ece(preds=logits_x_lb.softmax(dim=1).detach().cpu().numpy(), targets=y_lb.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                result['warm_up_lr'][i] = warm_up_lr
                if current_epoch >= start_fix:
                    if used_unlabeled_index.sum().item() != 0:
                        result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(logits_x_ulb_w[used_unlabeled_index], unlabel_y[used_unlabeled_index]).item()
                        result['unlabeled_top@5'][i] = TopKAccuracy(k=5)(logits_x_ulb_w[used_unlabeled_index], unlabel_y[used_unlabeled_index]).item()
                        result['unlabeled_ece'][i] = self.get_ece(preds=logits_x_ulb_w[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                                targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                    result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()
                    
                    unique, counts = np.unique(unlabel_y[used_unlabeled_index].cpu().numpy(), return_counts = True)
                    uniq_cnt_dict = dict(zip(unique, counts))
                    
                    for key,value in uniq_cnt_dict.items():
                        cls_wise_results[key][i] = value

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        if current_epoch >= start_fix:
            return {k: v.mean().item() for k, v in result.items()}, cls_wise_results
        else:
            return {k: v.mean().item() for k, v in result.items()}, None

    @torch.no_grad()
    def evaluate(self, data_loader, n_bins):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps),
            'top@1': torch.zeros(1),
            'top@5': torch.zeros(1),
            'ece': torch.zeros(1)
        }

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            pred,true = [],[]
            for i, batch in enumerate(data_loader):

                x, y = batch[0], batch[1]

                with torch.autocast('cuda', enabled = self.mixed_precision):
                    logits = self.predict(x)
                    loss = self.loss_function(logits, y.long())

                result['loss'][i] = loss.item()
                true.append(y.cpu())
                pred.append(logits.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: " + f" loss : {result['loss'][:i+1].mean():.4f} |" + f" top@1 : {TopKAccuracy(k=1)(logits, y).detach():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # preds, pred are logit vectors
        preds, trues = torch.cat(pred,axis=0), torch.cat(true,axis=0)
        result['top@1'][0] = TopKAccuracy(k=1)(preds, trues).item()
        result['top@5'][0] = TopKAccuracy(k=5)(preds, trues).item()

        ece_results = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=n_bins, plot=False)
        result['ece'][0] = ece_results[0]

        return {k: v.mean().item() for k, v in result.items()}

        
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


import torch
import torch.nn.functional as F

__all__ = ['ova_loss', 'unlabeled_ova_neg_loss']

def ova_loss(logits_open_w, logits_open_s, label, label_s):
    """
        label_sp_neg =
        [[1, 1, 0, 1],
         [0, 1, 1, 1],
         [1, 1, 1, 0]]

    OpenMatch hard classifier mining. For each training sample, we find a hard classifier
    (previously was each training sample trains all classifiers)
    In fact, we want, for each classifier, find a hard negative sample

    How to train a OVA classifier?
    1. Consider all negatives, which will lead to imbalance problem.
    2. Always select the top K negatives for each classifier, where K = bs / C, so that the number of positives
       and negatives the classifier receives per batch (averagely speaking) are equal.
       To do so, compute a topK_mask.
    3. Always select the top 1 negatives for each classifier.
       To do so, compute a topK_mask.
    4. A classifier receives bs / C positive samples per batch, but (bs - bs / C) negative samples per batch.
       So to balance out, we need to divide the negative loss by some value so that they are of the same scale.
       But maybe this is not necessary? Because if you take the mean, they are already of the same scale.
       So maybe start with 1. and 2.

    """
    logits_open_w = logits_open_w.view(logits_open_w.size(0), 2, -1)  # [bs, 2, num_class]
    logits_open_s = logits_open_s.view(logits_open_s.size(0), 2, -1)  # [bs, 2, num_class]

    # negative loss with one-hot label
    label_s_sp = torch.zeros((logits_open_w.size(0), logits_open_w.size(2))).long().to(label.device)  # [bs, num_class]
    label_s_sp.scatter_(1, label.view(-1, 1), 1)
    label_sp_neg = 1 - label_s_sp
    loss_values_w = -F.log_softmax(logits_open_w, dim=1)[:, 0, :] * label_sp_neg  # [bs, num_class]

    label_s_sp_s = torch.zeros((logits_open_s.size(0), logits_open_s.size(2))).long().to(label_s.device)  # [bs, num_class]
    label_s_sp_s.scatter_(1, label_s.view(-1, 1), 1)

    loss_values = loss_values_w
    open_loss_neg = torch.mean(loss_values.mean(dim=0))

    # positive loss
    open_loss_pos = torch.mean((-F.log_softmax(logits_open_w, dim=1)[:, 1, :] * label_s_sp).sum(dim=1))

    return open_loss_pos + open_loss_neg

def unlabeled_ova_neg_loss(logits_w1, logits_w2, logits_s1, logits_s2, mask):
    logits_w2 = logits_w2.view(logits_w2.size(0), 2, -1)  # [bs, 2, num_class]
    logits_s1 = logits_s1.view(logits_s1.size(0), 2, -1)  # [bs, 2, num_class]
    open_loss_neg = torch.mean((-F.log_softmax(logits_s1, dim=1)[:, 0, :] * mask).sum(dim=1))
    return open_loss_neg


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le