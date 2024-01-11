import collections
import os

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress
from sklearn.metrics import accuracy_score, precision_score, recall_score
from skimage.filters import threshold_otsu

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tasks.classification import Classification as Task
from tasks.classification_OPENMATCH import DistributedSampler
from utils import TopKAccuracy

from utils.logging import make_epoch_description
import matplotlib.pyplot as plt
plt.style.use('bmh')

class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            save_every,
            tau,
            tau_two,
            alpha_kl,
            n_bins: int = 15,
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
        u_loader = DataLoader(train_set[1],batch_size=self.batch_size//2, sampler=u_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)

        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

        # Logging
        logger = kwargs.get('logger', None)
        enable_plot = kwargs.get('enable_plot',False)

        best_eval_acc = -float('inf')
        best_epoch    = 0

        self.trained_iteration = 0
        
        for epoch in range(1, epochs//2 + 1):
            print(epoch)
            # train_history = self.pretrain(l_loader,u_loader)
            # eval_history = self.evaluate(eval_loader, n_bins)

        #     epoch_history = collections.defaultdict(dict)
        #     for k, v1 in train_history.items():
        #         epoch_history[k]['train'] = v1
        #         try:
        #             v2 = eval_history[k]
        #             epoch_history[k]['eval'] = v2
        #         except KeyError:
        #             continue

        #     # Write TensorBoard summary
        #     if self.writer is not None:
        #         for k, v in epoch_history.items():
        #             for k_, v_ in v.items():
        #                 self.writer.add_scalar(f'{k}_{k_}', v_, global_step=epoch)
        #         if self.scheduler is not None:
        #             lr = self.scheduler.get_last_lr()[0]
        #             self.writer.add_scalar('lr', lr, global_step=epoch)

        #     # Save best model checkpoint and Logging
        #     eval_acc = eval_history['top@1']
        #     if eval_acc > best_eval_acc:
        #         best_eval_acc = eval_acc
        #         best_epoch = epoch
        #         if self.local_rank == 0:
        #             ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
        #             self.save_checkpoint(ckpt, epoch=epoch)

        #         test_history = self.evaluate(test_loader, n_bins=n_bins)
        #         for k, v1 in test_history.items():
        #             epoch_history[k]['test'] = v1

        #         if self.writer is not None:
        #             self.writer.add_scalar('Best_Test_top@1', test_history['top@1'], global_step=epoch)

        #     # Write logs
        #     log = make_epoch_description(
        #         history=epoch_history,
        #         current=epoch,
        #         total=epochs,
        #         best=best_epoch,
        #     )
        #     if logger is not None:
        #         logger.info(log)
            
        prototypes = self.backbone.output.state_dict()['linear.weight']
        prototypes = self.exclude_dataset(unlabeled_dataset=train_set[1],selected_dataset=train_set[-1],prototypes=prototypes,current_epoch=epoch)

        pre_weight = self.backbone.novel_classifier.state_dict()
        pre_weight['weight'] = prototypes
        self.backbone.novel_classifier.load_state_dict(pre_weight)

        u_sel_sampler = DistributedSampler(dataset=train_set[-1], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        selected_u_loader = DataLoader(train_set[-1], sampler=u_sel_sampler,batch_size=self.batch_size//2,num_workers=num_workers,drop_last=False,pin_memory=False,shuffle=False)

        self.backbone.register_buffer('memory_queue',torch.zeros((len(train_set[-1]),self.backbone.class_num+1),device=self.local_rank))
        
        for epoch in range(epochs//2 + 1, epochs + 1):

            train_history, cls_wise_results = self.k_plus_train(l_loader,selected_u_loader,tau=tau,tau_two=tau_two,alpha_kl=alpha_kl,n_bins=n_bins)
            eval_history = self.k_plus_evaluate(eval_loader, n_bins)

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
                if cls_wise_results is not None:
                    self.writer.add_scalar("trained_unlabeled_data_in", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key<self.backbone.class_num]).item() , global_step=epoch)
                    self.writer.add_scalar("trained_unlabeled_data_ood", sum([cls_wise_results[key].mean() for key in cls_wise_results.keys() if key>=self.backbone.class_num]).item() , global_step=epoch)

            # Save best model checkpoint and Logging
            eval_acc = eval_history['top@1']
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader, n_bins=n_bins)
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

    def pretrain(self, label_loader, unlabel_loader):
        """Training defined for a single epoch."""

        iteration = len(label_loader)

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'rot_top@1': torch.zeros(iteration, device=self.local_rank),
        }
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb) in enumerate(zip(label_loader, unlabel_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    label_x = data_lb['x_lb'].to(self.local_rank)
                    label_y = data_lb['y_lb'].to(self.local_rank)

                    x_ulb_w  = data_ulb["x_ulb_w"].to(self.local_rank)

                    label_logit = self.predict(label_x)

                    x_ulb_r = torch.cat([torch.rot90(x_ulb_w, rot , [2, 3]) for rot in range(4)], dim=0)
                    y_ulb_r = torch.cat([torch.empty(x_ulb_w.size(0)).fill_(rot).long() for rot in range(4)], dim=0).to(self.local_rank)

                    logits_rot = self.backbone.rot_classifier(self.get_feature(x_ulb_r)[-1])
                    loss = self.loss_function(label_logit, label_y.long()) + self.loss_function(logits_rot, y_ulb_r)
                  
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
                result['rot_top@1'][i] = TopKAccuracy(k=1)(logits_rot, y_ulb_r).detach()

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

    def exclude_dataset(self,unlabeled_dataset,selected_dataset,prototypes,current_epoch):

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

                    x = data['x_ulb_w'].cuda(self.local_rank)
                    y = data['y_ulb'].cuda(self.local_rank)

                    _, feature = self.get_feature(x)
                    
                    d_i = (feature.unsqueeze(1)-prototypes.unsqueeze(0)).square().sum(-1)
                    ood_score = d_i.min(-1)[0]

                    gt_idx = y < self.backbone.class_num
                    if batch_idx == 0:
                        score_all = ood_score
                        gt_all = gt_idx
                        features = feature
                    else:
                        score_all = torch.cat([score_all, ood_score], 0)
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        features = torch.cat([features, feature], 0)
                        
                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(loader)}] "
                        pg.update(task, advance=1., description=desc)
                        pg.refresh()

        lambda_ = threshold_otsu(score_all.cpu().numpy())
        select_all = score_all <= lambda_

        # positive : inlier, negative : out of distribution
        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        selected_idx = torch.arange(0, len(select_all),device=self.local_rank)[select_all]

        # Write TensorBoard summary
        if self.writer is not None:
            self.writer.add_scalar('Selected ratio', len(selected_idx) / len(select_all), global_step=current_epoch)
            self.writer.add_scalar('Selected accuracy', select_accuracy, global_step=current_epoch)
            self.writer.add_scalar('Selected precision', select_precision, global_step=current_epoch)
            self.writer.add_scalar('Selected recall', select_recall, global_step=current_epoch)

        self._set_learning_phase(train=True)
        if len(selected_idx) > 0:
            selected_dataset.set_index(selected_idx)
            
        return torch.cat([prototypes,features[~select_all].mean(0,keepdims=True)],axis=0)
            
    @torch.no_grad()
    def k_plus_evaluate(self, data_loader, n_bins):
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

                feat = self.backbone.get_only_feat(x)
                logits = self.backbone.novel_classifier(feat)

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

        return {k: v.mean().item() for k, v in result.items()}

    def k_plus_train(self, label_loader, selected_unlabel_loader, tau, tau_two, alpha_kl, n_bins):
        """Training defined for a single epoch."""
        
        iteration = len(selected_unlabel_loader)

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'negative_loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
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

            for i, (data_lb, data_ulb_selected) in enumerate(zip(label_loader, selected_unlabel_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    label_x = data_lb['x_lb'].to(self.local_rank)
                    label_y = data_lb['y_lb'].to(self.local_rank)

                    x_ulb_w = data_ulb_selected["x_ulb_w"].to(self.local_rank)
                    x_ulb_s = data_ulb_selected["x_ulb_s"].to(self.local_rank)
                    ulb_idx = data_ulb_selected["idx_ulb"].to(self.local_rank)

                    unlabel_y = data_ulb_selected["unlabel_y"].to(self.local_rank)

                    features = self.backbone.get_only_feat(torch.cat((label_x, x_ulb_w, x_ulb_s), 0))
                    full_logits = self.backbone.novel_classifier(features)
                    
                    label_logit, u_weak_logit, u_strong_logit = full_logits.chunk(3)
                    label_loss = self.loss_function(label_logit, label_y.long())
                    
                    unlabel_confidence, unlabel_pseudo_y = u_weak_logit.softmax(1).max(1)
                    used_unlabeled_index = unlabel_confidence>tau

                    unlabel_loss = alpha_kl* torch.nn.functional.kl_div(input=u_strong_logit.log_softmax(dim=1),target=u_weak_logit,reduction='batchmean')
                    if used_unlabeled_index.sum().item() != 0:
                        unlabel_loss += self.loss_function(u_strong_logit[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())

                    complementary_confidence, l_in_equ_nine = u_weak_logit.softmax(1).min(1)
                    negative_loss = torch.tensor(0).to(self.local_rank)

                    negative_idx = (complementary_confidence<=tau_two) & (~used_unlabeled_index)
                    ulb_negative_idx = (negative_idx) & (self.backbone.memory_queue[ulb_idx].sum(1)<self.backbone.class_num)
                    self.backbone.memory_queue[ulb_idx[ulb_negative_idx],l_in_equ_nine[ulb_negative_idx]] = 1

                    if (~used_unlabeled_index).sum().item()!=0:
                        
                        complementary_label = self.backbone.memory_queue[ulb_idx[~used_unlabeled_index]]
                        negative_loss += ((1-u_weak_logit[~used_unlabeled_index].softmax(1)+1e-8).log()*complementary_label).sum(1).mean().neg()
                
                    loss = label_loss + unlabel_loss + negative_loss
                  
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
                result['negative_loss'][i] = negative_loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(label_logit, label_y).detach()
                result['ece'][i] = self.get_ece(preds=label_logit.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()

                if used_unlabeled_index.sum().item()!=0:
                    result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(u_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_ece'][i] = self.get_ece(preds=u_weak_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                                targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
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

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}, cls_wise_results

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

        return {k: v.mean().item() for k, v in result.items()}

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