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
            threshold,
            T,
            tau1,
            tau12,
            tau2,
            ema_u,
            est_epoch,
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

        self.est_step = 0
        self.py_con = compute_py(l_loader, device=self.local_rank)
        self.py_uni = torch.ones(self.backbone.class_num) / self.backbone.class_num
        self.u_py = torch.ones(self.backbone.class_num) / self.backbone.class_num
        self.u_py = self.u_py.to(self.local_rank)
        self.py_rev = torch.flip(self.py_con, dims=[0])
        self.py_uni = self.py_uni.to(self.local_rank)
        self.adjustment_l1 = compute_adjustment_by_py(self.py_con, tau1, device=self.local_rank)
        self.adjustment_l12 = compute_adjustment_by_py(self.py_con, tau12, device=self.local_rank)
        self.adjustment_l2 = compute_adjustment_by_py(self.py_con, tau2, device=self.local_rank)
        self.taumin = 0
        self.taumax = tau1
        self.ema_u = ema_u

        self.threshold = threshold
        self.T = T
        self.est_epoch = est_epoch

        for epoch in range(1, epochs + 1):

            if epoch > self.est_epoch:
                count_KL = count_KL / len(l_loader)
                KL_softmax = (torch.exp(count_KL[0])) / (torch.exp(count_KL[0])+torch.exp(count_KL[1])+torch.exp(count_KL[2]))
                tau = self.taumin + (self.taumax - self.taumin) * KL_softmax
                if math.isnan(tau)==False:
                    self.adjustment_l1 = compute_adjustment_by_py(self.py_con, tau, device=self.local_rank)
            count_KL = torch.zeros(3).to(self.local_rank)

            train_history, count_KL = self.train(label_loader=l_loader,
                                       unlabel_loader=unlabel_loader,
                                       current_epoch=epoch,
                                       start_fix=start_fix,
                                       n_bins=n_bins,
                                       count_KL=count_KL,
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

    def train(self, label_loader, unlabel_loader, current_epoch, start_fix, n_bins, count_KL):
        """Training defined for a single epoch."""

        iteration = len(unlabel_loader)
        
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
        }
        
        KL_div = nn.KLDivLoss(reduction='sum')
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb) in enumerate(zip(label_loader, unlabel_loader)):

                with torch.cuda.amp.autocast(self.mixed_precision):

                    inputs_x = data_lb["inputs_x"]
                    targets_x = data_lb["targets_x"] 

                    inputs_u_w = data_ulb["inputs_u_w"]
                    inputs_u_s = data_ulb["inputs_u_s"]
                    inputs_u_s1 = data_ulb["inputs_u_s1"]
                    u_real = data_ulb["u_real"]

                    u_real = u_real.cuda()
                    mask_l = (u_real != -2)
                    mask_l = mask_l.cuda()                    
                    
                    batch_size = inputs_x.shape[0]
                    inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)).to(self.local_rank)
                    targets_x = targets_x.to(self.local_rank)

                    logits, logits_b = self.acr_predict(inputs)

                    logits_x = logits[:batch_size]
                    logits_u_w, logits_u_s, logits_u_s1 = logits[batch_size:].chunk(3)
                    del logits
                    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                    logits_x_b = logits_b[:batch_size]
                    logits_u_w_b, logits_u_s_b, logits_u_s1_b = logits_b[batch_size:].chunk(3)
                    del logits_b
                    Lx_b = F.cross_entropy(logits_x_b + self.adjustment_l2, targets_x, reduction='mean')

                    pseudo_label = torch.softmax((logits_u_w.detach() - self.adjustment_l1) / self.T, dim=-1)
                    pseudo_label_h2 = torch.softmax((logits_u_w.detach() - self.adjustment_l12) / self.T, dim=-1)
                    pseudo_label_b = torch.softmax(logits_u_w_b.detach() / self.T, dim=-1)
                    pseudo_label_t = torch.softmax(logits_u_w.detach() / self.T, dim=-1)

                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    max_probs_h2, targets_u_h2 = torch.max(pseudo_label_h2, dim=-1)
                    max_probs_b, targets_u_b = torch.max(pseudo_label_b, dim=-1)
                    max_probs_t, targets_u_t = torch.max(pseudo_label_t, dim=-1)

                    mask = max_probs.ge(self.threshold)
                    mask_h2 = max_probs_h2.ge(self.threshold)
                    mask_b = max_probs_b.ge(self.threshold)
                    mask_t = max_probs_t.ge(self.threshold)

                    mask_ss_b_h2 = mask_b + mask_h2
                    mask_ss_t = mask + mask_t

                    mask = mask.float()
                    mask_b = mask_b.float()

                    mask_ss_b_h2 = mask_ss_b_h2.float()
                    mask_ss_t = mask_ss_t.float()

                    mask_twice_ss_b_h2 = torch.cat([mask_ss_b_h2, mask_ss_b_h2], dim=0).cuda()
                    mask_twice_ss_t = torch.cat([mask_ss_t, mask_ss_t], dim=0).cuda()

                    logits_u_s_twice = torch.cat([logits_u_s, logits_u_s1], dim=0).cuda()
                    targets_u_twice = torch.cat([targets_u, targets_u], dim=0).cuda()
                    targets_u_h2_twice = torch.cat([targets_u_h2, targets_u_h2], dim=0).cuda()

                    logits_u_s_b_twice = torch.cat([logits_u_s_b, logits_u_s1_b], dim=0).cuda()

                    now_mask = torch.zeros(self.backbone.class_num)
                    now_mask = now_mask.to(self.local_rank)
                    u_real[u_real==-2] = 0

                    if current_epoch > self.est_epoch:
                        now_mask[targets_u_b] += mask_l*mask_b
                        self.est_step += 1

                        if now_mask.sum() > 0:
                            now_mask = now_mask / now_mask.sum()
                            self.u_py = self.ema_u * self.u_py + (1-self.ema_u) * now_mask
                            KL_con = 0.5 * KL_div(self.py_con.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_con)
                            KL_uni = 0.5 * KL_div(self.py_uni.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_uni)
                            KL_rev = 0.5 * KL_div(self.py_rev.log(), self.u_py) + 0.5 * KL_div(self.u_py.log(), self.py_rev)
                            count_KL[0] = count_KL[0] + KL_con
                            count_KL[1] = count_KL[1] + KL_uni
                            count_KL[2] = count_KL[2] + KL_rev

                    Lu = (F.cross_entropy(logits_u_s_twice, targets_u_twice,
                                        reduction='none') * mask_twice_ss_t).mean()
                    Lu_b = (F.cross_entropy(logits_u_s_b_twice, targets_u_h2_twice,
                                            reduction='none') * mask_twice_ss_b_h2).mean()

                    if current_epoch < start_fix:

                        Lu = torch.zeros(1).to(self.local_rank).mean()
                        Lu_b = torch.zeros(1).to(self.local_rank).mean()

                    loss = Lx + Lu + Lx_b + Lu_b

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits_x, targets_x).detach()
                result['ece'][i] = self.get_ece(preds=logits_x.softmax(dim=1).detach().cpu().numpy(), targets=targets_x.cpu().numpy(), n_bins=n_bins, plot=False)[0]

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}, count_KL
    
    def acr_predict(self, x: torch.FloatTensor):

        logits, feat = self.get_feature(x)
        
        return logits, self.backbone.fc1(feat)

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


from tqdm import tqdm
def compute_py(train_loader, device):
    """compute the base probabilities"""
    label_freq = {}
    for data_lb in tqdm(train_loader):
        labell = data_lb['targets_x'].to(device)
        for j in labell:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    label_freq_array = torch.from_numpy(label_freq_array)
    label_freq_array = label_freq_array.to(device)
    return label_freq_array

def compute_adjustment_by_py(py, tro, device):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(device)
    return adjustments