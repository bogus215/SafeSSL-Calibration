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


class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            save_every,
            tau,
            consis_coef,
            warm_up_end,
            n_bins: int = 15,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers
        self.n_bins = n_bins

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (train, val, test)
        
        ## labeled 
        sampler = RandomSampler(len(train_set[0]), self.iterations * self.batch_size // 2)
        train_l_loader = DataLoader(train_set[0],batch_size=batch_size//2, sampler=sampler,num_workers=num_workers,drop_last=False,pin_memory=True)
        train_l_iterator = iter(train_l_loader)
        
        ## unlabeled
        sampler = RandomSampler(len(train_set[1]), self.iterations * self.batch_size // 2)
        train_u_loader = DataLoader(train_set[1],batch_size=batch_size//2,sampler=sampler,num_workers=num_workers,drop_last=False,pin_memory=True)
        train_u_iterator = iter(train_u_loader)
        
        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

        # Logging
        logger = kwargs.get('logger', None)

        # Supervised training
        best_eval_acc = -float('inf')
        best_epoch    = 0

        epochs = self.iterations // save_every
        self.warm_up_end = warm_up_end
        self.trained_iteration = 0

        for epoch in range(1, epochs + 1):

            # Train & evaluate
            train_history, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, tau=tau, consis_coef=consis_coef, smoothing_proposed=None if epoch==1 else ece_results)
            eval_history, ece_results = self.evaluate(eval_loader, n_bins)

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
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader, n_bins)
                for k, v1 in test_history[0].items():
                    epoch_history[k]['test'] = v1

                if self.writer is not None:
                    self.writer.add_scalar('Best_Test_top@1', test_history[0]['top@1'], global_step=epoch)

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

    def train(self, label_iterator, unlabel_iterator, iteration, tau, consis_coef, smoothing_proposed):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'warm_up_coef': torch.zeros(iteration, device=self.local_rank),
            'N_used_unlabeled': torch.zeros(iteration, device=self.local_rank)
        }
        
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

                    full_logits = self.predict(torch.cat([label_x, unlabel_weak_x, unlabel_strong_x],axis=0))
                    label_logit, unlabel_weak_logit, unlabel_strong_logit = full_logits.split(label_y.size(0))

                    full_scaled_logits = self.backbone.scaling_logits(full_logits)
                    label_scaled_logit, unlabel_weak_scaled_logit, unlabel_strong_scaled_logit = full_scaled_logits.split(label_y.size(0))
                    
                    unlabel_confidence, unlabel_pseudo_y = unlabel_weak_scaled_logit.softmax(1).max(1)

                    label_loss = self.loss_function(label_logit, label_y.long())
                    used_unlabeled_index = (unlabel_confidence>tau)

                    if smoothing_proposed is not None:

                        smoothing_proposed_surgery = dict()
                        for index_, (key_, value_) in enumerate(smoothing_proposed.items()):
                            if value_ is None:
                                smoothing_proposed_surgery[key_] = None
                            else:
                                if index_ != (len(smoothing_proposed)-1):
                                    if key_<=value_<=list(smoothing_proposed.keys())[index_+1]:
                                        smoothing_proposed_surgery[key_] = value_
                                    elif value_<key_:
                                        smoothing_proposed_surgery[key_] = key_
                                    else:
                                        smoothing_proposed_surgery[key_] = list(smoothing_proposed.keys())[index_+1]
                                else:
                                    if key_<=value_<=1:
                                        smoothing_proposed_surgery[key_] = value_
                                    elif value_<key_:
                                        smoothing_proposed_surgery[key_] = key_
                                    else:
                                        smoothing_proposed_surgery[key_] = 1

                        unlabeled_confidence = unlabel_weak_logit.softmax(dim=-1).max(1)[0].detach()
                        unlabel_confidence_surgery = unlabeled_confidence.clone()
                        
                        labeled_confidence = label_logit.softmax(dim=-1).max(1)[0].detach()
                        label_confidence_surgery = labeled_confidence.clone()

                        for index_, (key_, value_) in enumerate(smoothing_proposed_surgery.items()):
                            if index_ != (len(smoothing_proposed_surgery)-1):
                                mask = ((unlabel_confidence > key_) & (unlabel_confidence <= list(smoothing_proposed_surgery.keys())[index_+1]))
                                mask_ = ((labeled_confidence > key_) & (labeled_confidence <= list(smoothing_proposed_surgery.keys())[index_+1]))
                            else:
                                mask = ((unlabel_confidence > key_) & (unlabel_confidence <= 1))
                                mask_ = ((labeled_confidence > key_) & (labeled_confidence <= 1))
                                
                            if value_ is not None:
                                unlabel_confidence_surgery[mask] = value_
                                label_confidence_surgery[mask_] = value_

                        for_one_hot_label = nn.functional.one_hot(label_y,num_classes=self.backbone.output.out_features)
                        for_one_hot_unlabel = nn.functional.one_hot(unlabel_pseudo_y,num_classes=self.backbone.output.out_features)

                        for_smoothoed_target_label = (label_confidence_surgery.view(-1,1)*(for_one_hot_label==1) + ((1-label_confidence_surgery)/(self.backbone.output.out_features-1)).view(-1,1)*(for_one_hot_label!=1))
                        for_smoothoed_target_unlabel = (unlabel_confidence_surgery.view(-1,1)*(for_one_hot_unlabel==1) + ((1-unlabel_confidence_surgery)/(self.backbone.output.out_features-1)).view(-1,1)*(for_one_hot_unlabel!=1))

                        label_loss+= (-torch.mean(torch.sum(torch.log(label_scaled_logit.softmax(1)+1e-5)*for_smoothoed_target_label,axis=1)))
                        if used_unlabeled_index.sum().item() != 0:
                            unlabel_loss = self.loss_function(unlabel_strong_logit[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())
                            unlabel_loss += (-torch.mean(torch.sum(torch.log(unlabel_strong_scaled_logit[used_unlabeled_index].softmax(1)+1e-5)*for_smoothoed_target_unlabel.detach()[used_unlabeled_index],axis=1)))
                        else:
                            unlabel_loss = torch.zeros(1).to(self.local_rank)
                    else:
                        if used_unlabeled_index.sum().item() == 0:
                            unlabel_loss = torch.zeros(1).to(self.local_rank)
                        else:
                            unlabel_loss = self.loss_function(unlabel_strong_logit[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())
                 
                    warm_up_coef = consis_coef*math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)
                    loss = label_loss + warm_up_coef*unlabel_loss

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
                result['top@1'][i] = TopKAccuracy(k=1)(label_logit, label_y).detach()
                result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(unlabel_weak_logit, unlabel_y).detach()
                result['warm_up_coef'][i] = warm_up_coef
                result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}, label_iterator, unlabel_iterator

    @torch.no_grad()
    def evaluate(self, data_loader, n_bins):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(1, device=self.local_rank),
            'ece': torch.zeros(1, device=self.local_rank)
        }

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            pred,true,IDX=[],[],[]
            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)
                idx = batch['idx'].to(self.local_rank)

                logits = self.predict(x)
                logits = self.backbone.scaling_logits(logits)
                loss = self.loss_function(logits, y.long())

                result['loss'][i] = loss
                true.append(y.cpu())
                pred.append(logits.cpu())
                IDX += [idx]
                
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: " + f" loss : {result['loss'][:i+1].mean():.4f} |" + f" top@1 : {TopKAccuracy(k=1)(logits, y).detach():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # preds, pred are logit vectors
        preds, trues = torch.cat(pred,axis=0), torch.cat(true,axis=0)
        result['top@1'][0] = TopKAccuracy(k=1)(preds, trues)

        ece_results = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=n_bins, plot=False)
        result['ece'][0] = ece_results[0]
        
        return {k: v.mean().item() for k, v in result.items()}, ece_results[1]

    @staticmethod
    def get_ece(preds: np.array, targets: np.array, n_bins: int=15, **kwargs):

        bin_boundaries = np.linspace(0,1,n_bins+1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = np.max(preds, 1), np.argmax(preds,1)
        accuracies = (predictions == targets)

        ece = 0.0
        avg_confs_in_bins , x_ticks = [], []
        acc_ticks, confs_ticks = [] , []
        y_ticks_second_ticks=[]
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower , confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                delta = avg_confidence_in_bin - accuracy_in_bin

                avg_confs_in_bins.append(delta)
                acc_ticks.append(accuracy_in_bin)
                confs_ticks.append(avg_confidence_in_bin)
                x_ticks.append((bin_lower+bin_upper)/2)
                y_ticks_second_ticks.append(prop_in_bin)

                ece += np.abs(delta) * prop_in_bin
            else:
                avg_confs_in_bins.append(None)
                acc_ticks.append(None)
                confs_ticks.append(None)
                x_ticks.append(None)
                y_ticks_second_ticks.append(None)

        return ece, {tick:accuracy for tick, accuracy in zip(bin_boundaries.round(2)[:-1], acc_ticks)}