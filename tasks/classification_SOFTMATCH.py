import collections
import math
import os
import numpy as np

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader

from tasks.classification import Classification as Task
from utils import RandomSampler, TopKAccuracy
from utils.logging import make_epoch_description

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            open_test_set,
            save_every,
            n_sigma,
            ema_p,
            warm_up_end,
            n_bins,
            dist_da_len,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (train, val, test)
        
        ## labeled 
        sampler = RandomSampler(len(train_set[0]), self.iterations * self.batch_size // 2)
        train_l_loader = DataLoader(train_set[0],batch_size=batch_size//2, sampler=sampler,num_workers=num_workers,drop_last=False,pin_memory=False)
        train_l_iterator = iter(train_l_loader)
        
        ## unlabeled
        sampler = RandomSampler(len(train_set[1]), self.iterations * self.batch_size // 2)
        train_u_loader = DataLoader(train_set[1],batch_size=batch_size//2,sampler=sampler,num_workers=num_workers,drop_last=False,pin_memory=False)
        train_u_iterator = iter(train_u_loader)
        
        label_loader = DataLoader(train_set[0],batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        unlabel_loader = DataLoader(train_set[1],batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        open_test_loader = DataLoader(open_test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

        # Logging
        logger = kwargs.get('logger', None)
        enable_plot = kwargs.get('enable_plot',False)

        # Supervised training
        best_eval_acc = -float('inf')
        best_epoch    = 0

        epochs = self.iterations // save_every
        self.warm_up_end = warm_up_end
        self.trained_iteration = 0

        # Distribution Alignment
        self.da_module = DistAlignQueueHook(num_classes=self.backbone.class_num, queue_length=dist_da_len, p_target_type='uniform')
        self.softmatch_weight = SoftMatchWeightingHook(num_classes=self.backbone.class_num,n_sigma=n_sigma,momentum=ema_p)

        for epoch in range(1, epochs + 1):

            self.logging_unlabeled_dataset(unlabeled_dataset=train_set[1],current_epoch=epoch)
            
            # Train & evaluate
            train_history, cls_wise_results, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, n_bins=n_bins)
            eval_history = self.evaluate(eval_loader, n_bins)
            try:
                if self.ckpt_dir.split("/")[2]=='cifar10' and enable_plot:
                    label_preds, label_trues, label_FEATURE = self.log_plot_history(data_loader=label_loader, time=self.trained_iteration, name="label", return_results=True)
                    self.log_plot_history(data_loader=unlabel_loader, time=self.trained_iteration, name="unlabel", get_results=[label_preds, label_trues, label_FEATURE])
                    self.log_plot_history(data_loader=open_test_loader, time=self.trained_iteration, name="open+test")
            except:
                pass

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
                self.writer.add_scalar("trained_unlabeled_data_in", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key<self.backbone.class_num]).item() , global_step=epoch)
                self.writer.add_scalar("trained_unlabeled_data_ood", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key>=self.backbone.class_num]).item() , global_step=epoch)

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

    def train(self, label_iterator, unlabel_iterator, iteration, n_bins):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
            'warm_up_coef': torch.zeros(iteration, device=self.local_rank),
            'soft_mask_avg': torch.zeros(iteration, device=self.local_rank),
            'N_used_unlabeled': torch.zeros(iteration, device=self.local_rank)
        }
        
        if self.backbone.class_num==6:
            cls_wise_results = {i:torch.zeros(iteration) for i in range(10)}
        elif self.backbone.class_num==50:
            cls_wise_results = {i:torch.zeros(iteration) for i in range(100)}
        else:
            cls_wise_results = {i:torch.zeros(iteration) for i in range(200)}
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i in range(iteration):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    l_batch = next(label_iterator)
                    u_batch = next(unlabel_iterator)

                    label_x = l_batch['x'].to(self.local_rank)
                    label_y = l_batch['y'].to(self.local_rank)

                    unlabel_weak_x, unlabel_strong_x = u_batch['weak_img'].to(self.local_rank), u_batch['strong_img'].to(self.local_rank)
                    unlabel_y = u_batch['y'].to(self.local_rank)

                    outputs = self.predict(torch.cat([label_x, unlabel_weak_x, unlabel_strong_x],axis=0))

                    logits_x_lb = outputs[:label_x.size(0)]
                    logits_x_ulb_w, logits_x_ulb_s = outputs[label_x.size(0):].chunk(2)

                    # supervised losses
                    sup_loss = self.loss_function(logits_x_lb, label_y.long())

                    # generator closed-set and open-set targets (pseudo-labels)
                    with torch.no_grad():
                        p = nn.functional.softmax(logits_x_ulb_w, dim=-1)
                        targets_p = p.detach()
                        probs_x_ulb_w = self.da_module.dist_align(algorithm=None, probs_x_ulb=targets_p)
                        mask = self.softmatch_weight.masking(algorithm=None, logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
                        pseudo_label = targets_p.argmax(-1)

                    unsup_loss = (nn.functional.cross_entropy(logits_x_ulb_s, pseudo_label, reduction='none') * (mask)).mean()

                    warm_up_coef = math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)
                    loss = sup_loss + warm_up_coef*unsup_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.trained_iteration+=1

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(logits_x_lb, label_y).detach()
                result['ece'][i] = self.get_ece(preds=logits_x_lb.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(logits_x_ulb_w, unlabel_y).detach()
                result['unlabeled_ece'][i] = self.get_ece(preds=logits_x_ulb_w.softmax(dim=1).detach().cpu().numpy(),
                                                          targets=unlabel_y.cpu().numpy(),n_bins=n_bins, plot=False)[0]
                result['warm_up_coef'][i] = warm_up_coef
                result["N_used_unlabeled"][i] = logits_x_ulb_w.size(0)
                result["soft_mask_avg"][i] = mask.mean().item()

                unique, counts = np.unique(unlabel_y.cpu().numpy(), return_counts = True)
                uniq_cnt_dict = dict(zip(unique, counts))
                
                for key,value in uniq_cnt_dict.items():
                    cls_wise_results[key][i] = value

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}, cls_wise_results, label_iterator, unlabel_iterator
    
    def get_feature(self, x: torch.FloatTensor):
        """Make a prediction provided a batch of samples."""
        return self.backbone(x, True)
    
    def logging_unlabeled_dataset(self,unlabeled_dataset,current_epoch):

        loader = DataLoader(dataset=unlabeled_dataset,
                            batch_size=128,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

        self._set_learning_phase(train=False)
        
        with torch.no_grad():
            with Progress(transient=True, auto_refresh=False) as pg:
                if self.local_rank == 0:
                    task = pg.add_task(f"[bold red] Extracting...", total=len(loader))
                for batch_idx, data in enumerate(loader):

                    x = data['weak_img'].cuda(self.local_rank)
                    y = data['y'].cuda(self.local_rank)

                    logits_x_ulb_w = self.predict(x)

                    p = nn.functional.softmax(logits_x_ulb_w, dim=-1)
                    select_idx = p.max(1)[0] > 0.95

                    targets_p = p.detach()
                    targets_p = self.da_module.dist_align(algorithm=None, probs_x_ulb=targets_p)

                    gt_idx = y < self.backbone.class_num

                    if batch_idx == 0:
                        select_all = select_idx
                        gt_all = gt_idx
                        logits_all = logits_x_ulb_w
                        labels_all = y
                    else:
                        select_all = torch.cat([select_all, select_idx], 0)
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        logits_all = torch.cat([logits_all, logits_x_ulb_w], 0)
                        labels_all = torch.cat([labels_all, y], 0)
                        
                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(loader)}] "
                        pg.update(task, advance=1., description=desc)
                        pg.refresh()

        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy()) # positive : inlier, negative : out of distribution
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        selected_idx = torch.arange(0, len(select_all),device=self.local_rank)[select_all]

        probs_all = logits_all.softmax(-1)
        
        # Write TensorBoard summary
        if self.writer is not None:
            self.writer.add_scalar('Selected ratio', len(selected_idx) / len(select_all), global_step=current_epoch)
            self.writer.add_scalar('Selected accuracy', select_accuracy, global_step=current_epoch)
            self.writer.add_scalar('Selected precision', select_precision, global_step=current_epoch)
            self.writer.add_scalar('Selected recall', select_recall, global_step=current_epoch)
            self.writer.add_scalar('In distribution: ECE', self.get_ece(probs_all[gt_all].cpu().numpy(), labels_all[gt_all].cpu().numpy())[0], global_step=current_epoch)
            self.writer.add_scalar('In distribution: ACC', TopKAccuracy(k=1)(logits_all[gt_all],labels_all[gt_all]).item(), global_step=current_epoch)

            if ((gt_all) & (probs_all.max(1)[0]>=0.95)).sum()>0:
                self.writer.add_scalar('In distribution over conf 0.95: ECE', self.get_ece(probs_all[(gt_all) & (probs_all.max(1)[0]>=0.95)].cpu().numpy(), labels_all[(gt_all) & (probs_all.max(1)[0]>=0.95)].cpu().numpy())[0], global_step=current_epoch)
                self.writer.add_scalar('In distribution over conf 0.95: ACC', TopKAccuracy(k=1)(logits_all[(gt_all) & (probs_all.max(1)[0]>=0.95)], labels_all[(gt_all) & (probs_all.max(1)[0]>=0.95)]).item(), global_step=current_epoch)
                self.writer.add_scalar('Selected ratio of i.d over conf 0.95', ((gt_all) & (probs_all.max(1)[0]>=0.95)).sum() / gt_all.sum() , global_step=current_epoch)

            if (probs_all.max(1)[0]>=0.95).sum()>0:
                self.writer.add_scalar('Seen-class ratio over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]<self.backbone.class_num).sum() / (probs_all.max(1)[0]>=0.95).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class ratio over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]>=self.backbone.class_num).sum() / (probs_all.max(1)[0]>=0.95).sum(), global_step=current_epoch)

                self.writer.add_scalar('Seen-class over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]<self.backbone.class_num).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]>=self.backbone.class_num).sum(), global_step=current_epoch)

class DistAlignQueueHook:
    """
    Distribution Alignment Hook for conducting distribution alignment
    
    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook")
    """
    def __init__(self, num_classes, queue_length=128, p_target_type='uniform', p_target=None):
        self.num_classes = num_classes
        self.queue_length = queue_length

        # p_target
        self.p_target_ptr, self.p_target = self.set_p_target(p_target_type, p_target)    

        # p_model
        self.p_model = torch.zeros(self.queue_length, self.num_classes, dtype=torch.float)
        self.p_model_ptr = torch.zeros(1, dtype=torch.long)

    @torch.no_grad()
    def dist_align(self, algorithm, probs_x_ulb, probs_x_lb=None):

        # update queue
        self.update_p(algorithm, probs_x_ulb, probs_x_lb)

        # dist align
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target.mean(dim=0) + 1e-6) / (self.p_model.mean(dim=0) + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned
    
    @torch.no_grad()
    def update_p(self, algorithm, probs_x_ulb, probs_x_lb):
        # TODO: think better way?
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)
            if self.p_target_ptr is not None:
                self.p_target_ptr = self.p_target_ptr.to(probs_x_ulb.device)
        
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(probs_x_ulb.device)
            self.p_model_ptr = self.p_model_ptr.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        p_model_ptr = int(self.p_model_ptr)
        self.p_model[p_model_ptr] = probs_x_ulb.mean(dim=0)
        self.p_model_ptr[0] = (p_model_ptr + 1) % self.queue_length

        if self.p_target_ptr is not None:
            assert probs_x_lb is not None
            p_target_ptr = int(self.p_target_ptr)
            self.p_target[p_target_ptr] = probs_x_lb.mean(dim=0)
            self.p_target_ptr[0] = (p_target_ptr + 1) % self.queue_length
    
    def set_p_target(self, p_target_type='uniform', p_target=None):
        assert p_target_type in ['uniform', 'gt', 'model']

        # p_target
        p_target_ptr = None
        if p_target_type == 'uniform':
            p_target = torch.ones(self.queue_length, self.num_classes, dtype=torch.float) / self.num_classes
        elif p_target_type == 'model':
            p_target = torch.zeros((self.queue_length, self.num_classes), dtype=torch.float)
            p_target_ptr = torch.zeros(1, dtype=torch.long)
        else:
            assert p_target is not None
            if isinstance(p_target, np.ndarray):
                p_target = torch.from_numpy(p_target)
            p_target = p_target.unsqueeze(0).repeat((self.queue_length, 1))
        
        return p_target_ptr, p_target
    
class SoftMatchWeightingHook:
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False):

        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.args.num_classes
            self.prob_max_var_t =  torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
            self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))

        return mask