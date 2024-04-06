import collections
import os

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tasks.classification import Classification as Task
from tasks.classification_OPENMATCH import DistributedSampler
from utils import TopKAccuracy
from utils.logging import make_epoch_description

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns

class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            open_test_set,

            save_every,
            tau,

            lambda_cali,
            lambda_ova_soft,
            lambda_ova,
            lambda_fix,

            start_fix: int =5,

            n_bins: int = 15,
            train_n_bins: int = 30,
            **kwargs):  # pylint: disable=unused-argument

        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (train, val, test)
        epochs = self.iterations // save_every
        per_epoch_steps = self.iterations // epochs
        num_samples = per_epoch_steps * self.batch_size // 2 

        l_sampler = DistributedSampler(dataset=train_set[0], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        l_loader = DataLoader(train_set[0],batch_size=self.batch_size//2, sampler=l_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)

        u_sampler = DistributedSampler(dataset=train_set[1], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        u_loader = DataLoader(train_set[1],batch_size=self.batch_size//2, sampler=u_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)
        
        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

        # Logging
        logger = kwargs.get('logger', None)
        enable_plot = kwargs.get('enable_plot',False)

        best_eval_acc = -float('inf')
        best_epoch    = 0

        self.trained_iteration = 0
        
        for epoch in range(1, epochs + 1):

            # Selection related to unlabeled data
            self.logging_unlabeled_dataset(unlabeled_dataset=train_set[1],current_epoch=epoch)

            train_history, cls_wise_results = self.train(label_loader=l_loader,
                                                         unlabel_loader=u_loader,
                                                         current_epoch=epoch,
                                                         start_fix=start_fix,
                                                         tau=tau,

                                                         lambda_cali=lambda_cali,
                                                         lambda_ova_soft=lambda_ova_soft,
                                                         lambda_ova=lambda_ova,
                                                         lambda_fix=lambda_fix,

                                                         smoothing_linear=None if epoch==1 else ece_linear_results,

                                                         n_bins=n_bins)

            eval_history, ece_linear_results, ece_ova_results = self.evaluate(eval_loader, n_bins=n_bins, train_n_bins=train_n_bins)

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

            if logger is not None and eval_acc==1:
                logger.info("Eval acc == 1 --> Stop training")
                break

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader, n_bins=n_bins, train_n_bins=train_n_bins)
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

    def train(self, label_loader, unlabel_loader, current_epoch, start_fix, tau, lambda_cali, lambda_ova_soft, lambda_ova, lambda_fix, smoothing_linear, n_bins):
        """Training defined for a single epoch."""

        iteration = len(label_loader)

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),

            'label_sup': torch.zeros(iteration, device=self.local_rank),
            'label_cali': torch.zeros(iteration, device=self.local_rank),
            'label_ova': torch.zeros(iteration, device=self.local_rank),

            'unlabel_fix': torch.zeros(iteration, device=self.local_rank),
            'unlabel_ova_socr': torch.zeros(iteration, device=self.local_rank),

            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ova-top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
            'N_used_unlabeled': torch.zeros(iteration, device=self.local_rank),
            "cali_temp": torch.zeros(iteration, device=self.local_rank),
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

            for i, (data_lb, data_ulb) in enumerate(zip(label_loader, unlabel_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    label_x = data_lb['x_lb'].to(self.local_rank)
                    label_y = data_lb['y_lb'].to(self.local_rank)

                    unlabel_weak_x = data_ulb['x_ulb_w'].to(self.local_rank)
                    unlabel_weak_t_x = data_ulb['x_ulb_w_t'].to(self.local_rank)
                    unlabel_y = data_ulb['y_ulb'].to(self.local_rank)
                    x_ulb_s = data_ulb["x_ulb_s"].to(self.local_rank)

                    full_logits, full_features = self.get_feature(torch.cat([label_x, unlabel_weak_x, unlabel_weak_t_x, x_ulb_s],axis=0))
                    label_logit, weak_logit, _, strong_logit = full_logits.chunk(4)

                    ova_full_logits = self.backbone.ova_classifiers(full_features)
                    label_ova_logit, weak_ova_logit, weak_ova_t_logit, _ = ova_full_logits.chunk(4)

                    label_sup_loss = self.loss_function(label_logit, label_y)

                    label_ova_loss = ova_loss_func(label_ova_logit,label_y)
                    ova_soft_loss = socr_loss_func(weak_ova_logit, weak_ova_t_logit)

                    cali_loss = torch.tensor(0).cuda(self.local_rank)

                    if smoothing_linear is not None:

                        smoothing_proposed_surgery = self.clamp(smoothing_linear)

                        labeled_confidence = label_logit.softmax(dim=-1).max(1)[0].detach()
                        label_confidence_surgery = self.adaptive_smoothing(confidence=labeled_confidence,acc_distribution=smoothing_proposed_surgery)

                        for_one_hot_label = nn.functional.one_hot(label_y,num_classes=self.backbone.output.out_features)
                        for_smoothoed_target_label = (label_confidence_surgery.view(-1,1)*(for_one_hot_label==1) + ((1-label_confidence_surgery)/(self.backbone.output.out_features-1)).view(-1,1)*(for_one_hot_label!=1))

                        cali_loss = (-torch.mean(torch.sum(torch.log(self.backbone.scaling_logits(label_logit).softmax(1)+1e-5)*for_smoothoed_target_label,axis=1)))

                    fix_loss = torch.tensor(0).cuda(self.local_rank)

                    if current_epoch >= start_fix:

                        with torch.no_grad():
                            unlabel_weak_scaled_logit = self.backbone.scaling_logits(weak_logit)
                            unlabel_weak_scaled_softmax = unlabel_weak_scaled_logit.softmax(1)
                            unlabel_confidence, unlabel_pseudo_y = unlabel_weak_scaled_softmax.max(1)
                            
                            ood_score = (((weak_ova_logit.detach()).view(len(unlabel_weak_x),2,-1).softmax(1))[:,0,:] * unlabel_weak_scaled_softmax).sum(1)
                            used_unlabeled_index = (ood_score < 0.5) & (unlabel_confidence > tau)

                        if used_unlabeled_index.sum().item() != 0:
                            fix_loss = self.loss_function(strong_logit[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())
                  
                    loss = label_sup_loss + lambda_cali*cali_loss + lambda_ova*label_ova_loss + lambda_ova_soft*ova_soft_loss +lambda_fix*fix_loss

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

                result['label_sup'][i] = label_sup_loss.detach()
                result['label_cali'][i] = cali_loss.detach()
                result['label_ova'][i] = label_ova_loss.detach()

                result['unlabel_ova_socr'][i] = ova_soft_loss.detach()
                result['unlabel_fix'][i] = fix_loss.detach()

                result['top@1'][i] = TopKAccuracy(k=1)(label_logit, label_y).detach()
                result['ova-top@1'][i] = TopKAccuracy(k=1)(label_ova_logit.view(len(label_y),2,-1).softmax(1)[:,1,:], label_y).detach()
                result['ece'][i] = self.get_ece(preds=self.backbone.scaling_logits(label_logit).softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                result["cali_temp"][i] = self.backbone.cali_scaler.item()

                if current_epoch>=start_fix:
                    if used_unlabeled_index.sum().item()!=0:
                        result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                        result['unlabeled_ece'][i] = self.get_ece(preds=unlabel_weak_scaled_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                                  targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                    result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()

                    unique, counts = np.unique(unlabel_y[used_unlabeled_index].cpu().numpy(), return_counts = True)
                    uniq_cnt_dict = dict(zip(unique, counts))
                    
                    for key,value in uniq_cnt_dict.items():
                        cls_wise_results[key][i] = value
                else:
                    cls_wise_results = None
                    
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
    def evaluate(self, data_loader, n_bins, train_n_bins, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(1, device=self.local_rank),
            'ova-top@1': torch.zeros(1, device=self.local_rank),
            'ece': torch.zeros(1, device=self.local_rank)
        }

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            unscaled_pred, pred, ova_pred, true = [], [], [], []
            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                unscaled_logit, features = self.get_feature(x)
                logits = self.backbone.scaling_logits(unscaled_logit)

                loss = self.loss_function(logits, y.long())

                ova_logits = ((self.backbone.ova_classifiers(features))).view(features.size(0),2,-1)
                ova_logits = ova_logits.softmax(1)[:,1,:]

                result['loss'][i] = loss
                true.append(y.cpu())

                pred.append(logits.cpu())
                ova_pred.append(ova_logits.cpu())

                unscaled_pred.append(unscaled_logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: " + f" loss : {result['loss'][:i+1].mean():.4f} |" + f" top@1 : {TopKAccuracy(k=1)(logits, y).detach():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # preds, pred are logit vectors
        preds, trues, ova_preds = torch.cat(pred,axis=0), torch.cat(true,axis=0), torch.cat(ova_pred)
        unscaled_preds = torch.cat(unscaled_pred,axis=0)

        result['top@1'][0] = TopKAccuracy(k=1)(preds, trues)
        result['ova-top@1'][0] = TopKAccuracy(k=1)(ova_preds, trues)

        ece_results = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=n_bins, plot=False)
        result['ece'][0] = ece_results[0]

        train_ece_results = self.get_ece(preds=unscaled_preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=train_n_bins, plot=False)
        train_ece_results_ova = self.get_ece(preds=ova_preds.numpy(), targets=trues.numpy(), n_bins=train_n_bins, plot=False)
        
        return {k: v.mean().item() for k, v in result.items()}, train_ece_results[1], train_ece_results_ova[1]

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

                    x = data['x_ulb_w'].cuda(self.local_rank)
                    y = data['y_ulb'].cuda(self.local_rank)

                    logits, features = self.get_feature(x)
                    logits = self.backbone.scaling_logits(logits)
                    probs = nn.functional.softmax(logits, 1)

                    ova_logits = (self.backbone.ova_classifiers(features)).view(features.size(0),2,-1)
                    outlier_score = (ova_logits.softmax(1)[:,0,:] * probs).sum(1)
                    ova_in_logits = ova_logits.softmax(1)[:,1,:]

                    gt_idx = y < self.backbone.class_num

                    if batch_idx == 0:
                        gt_all = gt_idx
                        probs_all, logits_all = probs, logits
                        labels_all = y
                        ova_in_all = ova_in_logits
                        outlier_score_all = outlier_score
                    else:
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        probs_all, logits_all = torch.cat([probs_all, probs], 0), torch.cat([logits_all, logits], 0)
                        labels_all = torch.cat([labels_all, y], 0)
                        outlier_score_all = torch.cat([outlier_score_all, outlier_score], 0)
                        ova_in_all = torch.cat([ova_in_all, ova_in_logits],0)
                        
                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(loader)}] "
                        pg.update(task, advance=1., description=desc)
                        pg.refresh()

        select_all = outlier_score_all < 0.5

        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy()) # positive : inlier, negative : out of distribution
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_f1 = f1_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        selected_idx = torch.arange(0, len(select_all),device=self.local_rank)[select_all]

        # Write TensorBoard summary
        if self.writer is not None:
            self.writer.add_scalar('Selected accuracy', select_accuracy, global_step=current_epoch)
            self.writer.add_scalar('Selected precision', select_precision, global_step=current_epoch)
            self.writer.add_scalar('Selected recall', select_recall, global_step=current_epoch)
            self.writer.add_scalar('Selected f1', select_f1, global_step=current_epoch)
            self.writer.add_scalar('Selected ratio', len(selected_idx) / len(select_all), global_step=current_epoch)

            self.writer.add_scalar('In distribution: ECE', self.get_ece(probs_all[gt_all].cpu().numpy(), labels_all[gt_all].cpu().numpy())[0], global_step=current_epoch)
            self.writer.add_scalar('In distribution: ACC', TopKAccuracy(k=1)(logits_all[gt_all],labels_all[gt_all]).item(), global_step=current_epoch)
            self.writer.add_scalar('In distribution: ACC(OVA)', TopKAccuracy(k=1)(ova_in_all[gt_all],labels_all[gt_all]).item(), global_step=current_epoch)

            if ((gt_all) & (probs_all.max(1)[0]>=0.95)).sum()>0:
                self.writer.add_scalar('In distribution over conf 0.95: ECE', self.get_ece(probs_all[(gt_all) & (probs_all.max(1)[0]>=0.95)].cpu().numpy(), labels_all[(gt_all) & (probs_all.max(1)[0]>=0.95)].cpu().numpy())[0], global_step=current_epoch)
                self.writer.add_scalar('In distribution over conf 0.95: ACC', TopKAccuracy(k=1)(logits_all[(gt_all) & (probs_all.max(1)[0]>=0.95)], labels_all[(gt_all) & (probs_all.max(1)[0]>=0.95)]).item(), global_step=current_epoch)
                self.writer.add_scalar('In distribution over conf 0.95: ACC(OVA)', TopKAccuracy(k=1)(ova_in_all[(gt_all) & (probs_all.max(1)[0]>=0.95)], labels_all[(gt_all) & (probs_all.max(1)[0]>=0.95)]).item(), global_step=current_epoch)
                self.writer.add_scalar('Selected ratio of i.d over conf 0.95', ((gt_all) & (probs_all.max(1)[0]>=0.95)).sum() / gt_all.sum() , global_step=current_epoch)

            if ((gt_all) & (select_all)).sum()>0:
                self.writer.add_scalar('In distribution under ood score 0.5: ECE', self.get_ece(probs_all[(gt_all) & (select_all)].cpu().numpy(), labels_all[(gt_all) & (select_all)].cpu().numpy())[0], global_step=current_epoch)
                self.writer.add_scalar('In distribution under ood score 0.5: ACC', TopKAccuracy(k=1)(logits_all[(gt_all) & (select_all)], labels_all[(gt_all) & (select_all)]).item(), global_step=current_epoch)
                self.writer.add_scalar('In distribution under ood score 0.5: ACC(OVA)', TopKAccuracy(k=1)(ova_in_all[(gt_all) & (select_all)], labels_all[(gt_all) & (select_all)]).item(), global_step=current_epoch)
                self.writer.add_scalar('Selected ratio of i.d under ood score 0.5', ((gt_all) & (select_all)).sum() / gt_all.sum() , global_step=current_epoch)

            if (probs_all.max(1)[0]>=0.95).sum()>0:
                self.writer.add_scalar('Seen-class ratio over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]<self.backbone.class_num).sum() / (probs_all.max(1)[0]>=0.95).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class ratio over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]>=self.backbone.class_num).sum() / (probs_all.max(1)[0]>=0.95).sum(), global_step=current_epoch)

                self.writer.add_scalar('Seen-class over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]<self.backbone.class_num).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class over conf 0.95', (labels_all[(probs_all.max(1)[0]>=0.95)]>=self.backbone.class_num).sum(), global_step=current_epoch)
                
            if select_all.sum()>0:
                self.writer.add_scalar('Seen-class ratio under ood score 0.5', (labels_all[select_all]<self.backbone.class_num).sum() / select_all.sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class ratio under ood score 0.5', (labels_all[select_all]>=self.backbone.class_num).sum() / select_all.sum(), global_step=current_epoch)

                self.writer.add_scalar('Seen-class under ood score 0.5', (labels_all[select_all]<self.backbone.class_num).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class under ood score 0.5', (labels_all[select_all]>=self.backbone.class_num).sum(), global_step=current_epoch)

            if ((select_all) & (probs_all.max(1)[0]>=0.95)).sum()>0:
                self.writer.add_scalar('Seen-class ratio both under ood score 0.5 and over conf 0.95', (labels_all[((select_all) & (probs_all.max(1)[0]>=0.95))]<self.backbone.class_num).sum() / ((select_all) & (probs_all.max(1)[0]>=0.95)).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class ratio both under ood score 0.5 and over conf 0.95', (labels_all[((select_all) & (probs_all.max(1)[0]>=0.95))]>=self.backbone.class_num).sum() / ((select_all) & (probs_all.max(1)[0]>=0.95)).sum(), global_step=current_epoch)

                self.writer.add_scalar('Seen-class both under ood score 0.5 and over conf 0.95', (labels_all[((select_all) & (probs_all.max(1)[0]>=0.95))]<self.backbone.class_num).sum(), global_step=current_epoch)
                self.writer.add_scalar('Unseen-class both under ood score 0.5 and over conf 0.95', (labels_all[((select_all) & (probs_all.max(1)[0]>=0.95))]>=self.backbone.class_num).sum(), global_step=current_epoch)
                    
    @staticmethod
    def clamp(smoothing_proposed):
        
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
                        
        return smoothing_proposed_surgery
    
    @staticmethod
    def adaptive_smoothing(confidence, acc_distribution):
        
        confidence_surgery = confidence.clone()
        
        for index_, (key_, value_) in enumerate(acc_distribution.items()):
            if index_ != (len(acc_distribution)-1):
                mask_ = ((confidence > key_) & (confidence <= list(acc_distribution.keys())[index_+1]))
            else:
                mask_ = ((confidence > key_) & (confidence <= 1))

            if value_ is not None:
                confidence_surgery[mask_] = value_
                
        return confidence_surgery
    
    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'backbone': self.backbone.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)
        
def ova_loss_func(logits_open, label):
    # Eq.(1) in the paper
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = nn.functional.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = 1  # one-hot labels, in the shape of (bsz, num_classes)
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova = open_loss_neg + open_loss
    return l_ova

def ova_soft_loss_func(logits_open, confidence, label):
    # Eq.(1) in the paper
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = nn.functional.softmax(logits_open, 1)

    label_s_sp = torch.zeros((logits_open.size(0),logits_open.size(2))).to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_s_sp[label_range, label] = confidence  # one-hot labels, in the shape of (bsz, num_classes)

    non_label_index = torch.ones((logits_open.size(0),logits_open.size(2))).to(label.device)
    non_label_index[label_range, label] = 0
    label_s_sp[non_label_index.bool()] = (1-confidence).repeat_interleave(logits_open.size(2)-1)

    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova = open_loss_neg + open_loss

    return l_ova

def socr_loss_func(logits_open_u1, logits_open_u2):
    # Eq.(3) in the paper
    logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
    logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
    logits_open_u1 = nn.functional.softmax(logits_open_u1, 1)
    logits_open_u2 = nn.functional.softmax(logits_open_u2, 1)
    l_socr = torch.mean(torch.sum(torch.sum(torch.abs(
        logits_open_u1 - logits_open_u2) ** 2, 1), 1))
    return l_socr