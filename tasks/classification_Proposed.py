import collections
import math
import os

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tasks.classification import Classification as Task
from utils import RandomSampler, TopKAccuracy
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
            pi,
            consis_coef,
            cali_coef,
            entropy_coef,
            warm_up_end,
            n_bins: int = 15,
            train_n_bins: int = 30,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

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
        
        label_loader = DataLoader(train_set[0],batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        unlabel_loader = DataLoader(train_set[1],batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
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

        for epoch in range(1, epochs + 1):

            # Train & evaluate
            train_history, cls_wise_results, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, tau=tau, pi=pi, consis_coef=consis_coef, cali_coef=cali_coef, entropy_coef=entropy_coef, smoothing_proposed=None if epoch==1 else ece_results, n_bins=n_bins)
            eval_history, ece_results = self.evaluate(eval_loader, n_bins=n_bins, train_n_bins=train_n_bins)
            try:
                if self.ckpt_dir.split("/")[2] in ['cifar10','svhn'] and enable_plot:
                    label_preds, label_trues, label_FEATURE, label_CLS_LOSS, label_IDX = self.log_plot_history(data_loader=label_loader, time=self.trained_iteration, name="label", return_results=True)
                    self.log_plot_history(data_loader=unlabel_loader, time=self.trained_iteration, name="unlabel", get_results=[label_preds, label_trues, label_FEATURE, label_CLS_LOSS, label_IDX])
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

    def train(self, label_iterator, unlabel_iterator, iteration, tau, pi, consis_coef, cali_coef, entropy_coef, smoothing_proposed, n_bins):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
            'warm_up_coef': torch.zeros(iteration, device=self.local_rank),
            'N_used_unlabeled': torch.zeros(iteration, device=self.local_rank),
            "Temperature": torch.zeros(iteration, device=self.local_rank),
            'cali_loss': torch.zeros(iteration, device=self.local_rank),
            'l_ul_cls_loss' : torch.zeros(iteration, device=self.local_rank),
            'entropy_loss' : torch.zeros(iteration, device=self.local_rank)
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
                    warm_up_coef = math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)

                    l_batch = next(label_iterator)
                    u_batch = next(unlabel_iterator)

                    label_x = l_batch['x'].to(self.local_rank)
                    label_y = l_batch['y'].to(self.local_rank)

                    unlabel_weak_x, unlabel_strong_x = u_batch['weak_img'].to(self.local_rank), u_batch['strong_img'].to(self.local_rank)
                    unlabel_y = u_batch['y'].to(self.local_rank)

                    full_logits, full_features = self.get_feature(torch.cat([label_x, unlabel_weak_x, unlabel_strong_x],axis=0))
                    label_logit, unlabel_weak_logit, unlabel_strong_logit = full_logits.split(label_y.size(0))

                    full_scaled_logits = self.backbone.scaling_logits(full_logits)
                    label_scaled_logit, unlabel_weak_scaled_logit, _ = full_scaled_logits.split(label_y.size(0))
                    
                    unlabel_confidence, unlabel_pseudo_y = unlabel_weak_scaled_logit.softmax(1).max(1)

                    label_loss = self.loss_function(label_logit, label_y.long())
                    
                    l_ul_cls_losses = -(self.backbone.mlp(full_features).log_softmax(1)*nn.functional.one_hot(torch.cat([torch.zeros_like(label_y.long()),
                                                                                                                                   torch.ones_like(label_y.long()),
                                                                                                                                   torch.ones_like(label_y.long())]),2)).sum(1)
                    l_ul_cls_loss = l_ul_cls_losses.mean()
                    used_unlabeled_index = (unlabel_confidence>tau) & (l_ul_cls_losses[label_y.size(0):-label_y.size(0)].detach()>(pi*warm_up_coef))
                    
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

                        labeled_confidence = label_logit.softmax(dim=-1).max(1)[0].detach()
                        label_confidence_surgery = labeled_confidence.clone()

                        for index_, (key_, value_) in enumerate(smoothing_proposed_surgery.items()):
                            if index_ != (len(smoothing_proposed_surgery)-1):
                                mask_ = ((labeled_confidence > key_) & (labeled_confidence <= list(smoothing_proposed_surgery.keys())[index_+1]))
                            else:
                                mask_ = ((labeled_confidence > key_) & (labeled_confidence <= 1))

                            if value_ is not None:
                                label_confidence_surgery[mask_] = value_

                        for_one_hot_label = nn.functional.one_hot(label_y,num_classes=self.backbone.output.out_features)
                        for_smoothoed_target_label = (label_confidence_surgery.view(-1,1)*(for_one_hot_label==1) + ((1-label_confidence_surgery)/(self.backbone.output.out_features-1)).view(-1,1)*(for_one_hot_label!=1))

                        cali_loss = (-torch.mean(torch.sum(torch.log(label_scaled_logit.softmax(1)+1e-5)*for_smoothoed_target_label,axis=1)))
                        label_loss += cali_coef*cali_loss
                    
                    if used_unlabeled_index.sum().item() != 0:
                        unlabel_loss = self.loss_function(unlabel_strong_logit[used_unlabeled_index], unlabel_pseudo_y[used_unlabeled_index].long().detach())
                    else:
                        unlabel_loss = torch.zeros(1).to(self.local_rank)
                
                    loss = label_loss + unlabel_loss + l_ul_cls_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward(retain_graph=True) if used_unlabeled_index.sum().item() != 0 else self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward(retain_graph=True) if used_unlabeled_index.sum().item() != 0 else loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.trained_iteration+=1

                if used_unlabeled_index.sum().item() != 0:
                    with torch.cuda.amp.autocast(self.mixed_precision):
                        self.backbone.update_batch_stats(False)
                        reverse_logits = self.backbone.scaling_logits(self.backbone(unlabel_weak_x[used_unlabeled_index],reverse=True), name="entrop_scaler")
                        entropy_loss = (reverse_logits.softmax(1)*reverse_logits.log_softmax(1)).sum(1).mean()*entropy_coef
                        self.backbone.update_batch_stats(True)

                    if self.scaler is not None:
                        self.scaler.scale(entropy_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        entropy_loss.backward()
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(label_logit, label_y).detach()
                result['ece'][i] = self.get_ece(preds=label_logit.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                if smoothing_proposed is not None:
                    result['cali_loss'][i] = cali_loss.detach()
                if used_unlabeled_index.sum().item() != 0:
                    result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(unlabel_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_ece'][i] = self.get_ece(preds=unlabel_weak_scaled_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                              targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                    result['entropy_loss'][i] = entropy_loss.detach()
                result['warm_up_coef'][i] = warm_up_coef
                result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()
                result["Temperature"][i] = self.backbone.temperature.item()
                result['l_ul_cls_loss'][i] = l_ul_cls_loss.detach()

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

        return {k: v.mean().item() for k, v in result.items()}, cls_wise_results, label_iterator, unlabel_iterator

    @torch.no_grad()
    def evaluate(self, data_loader, n_bins, train_n_bins, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(1, device=self.local_rank),
            'ece': torch.zeros(1, device=self.local_rank)
        }

        swa_on = kwargs.get('swa_on',False)
        swa_start = kwargs.get('swa_start',100000)

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Evaluating...", total=steps)

            pred,true,IDX=[],[],[]
            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)
                idx = batch['idx'].to(self.local_rank)

                if swa_on and self.trained_iteration>=swa_start:
                    logits = self.swa_model(x)
                    logits = self.swa_model.scaling_logits(logits)
                else:
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

        train_ece_results = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=train_n_bins, plot=False)
        
        return {k: v.mean().item() for k, v in result.items()}, train_ece_results[1]

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
    
    @torch.no_grad()
    def log_plot_history(self, data_loader, time, name, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)

        return_results = kwargs.get("return_results",False)
        get_results = kwargs.get("get_results",None)

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Plotting...", total=steps)

            pred,true,IDX,FEATURE, CLS_LOSSES =[],[],[],[],[]
            for i, batch in enumerate(data_loader):

                try:
                    x = batch['x'].to(self.local_rank)
                except:
                    x = batch['weak_img'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)
                idx = batch['idx'].to(self.local_rank)

                logits, feature = self.get_feature(x)
                logits = self.backbone.scaling_logits(logits)
                true.append(y.cpu())
                pred.append(logits.cpu())
                FEATURE.append(feature.squeeze().cpu())
                IDX += [idx]

                if name == 'unlabel':
                    cls_losses = -(self.backbone.mlp(feature.squeeze()).log_softmax(1) * torch.nn.functional.one_hot(torch.ones_like(y), 2)).sum(1)
                    CLS_LOSSES.append(cls_losses)
                elif name == 'label':
                    cls_losses = -(self.backbone.mlp(feature.squeeze()).log_softmax(1) * torch.nn.functional.one_hot(torch.zeros_like(y), 2)).sum(1)
                    CLS_LOSSES.append(cls_losses)
                else:
                    pass

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: Having feature vector..."
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # preds, pred are logit vectors
        preds, trues = torch.cat(pred,axis=0), torch.cat(true,axis=0)
        FEATURE = torch.cat(FEATURE)
        IDX = torch.cat(IDX)
        if name in ['label','unlabel']:
            CLS_LOSSES = torch.cat(CLS_LOSSES)
            CLS_LOSS_IN = CLS_LOSSES[trues<self.backbone.class_num]
            CLS_LOSS_OOD = CLS_LOSSES[trues>=self.backbone.class_num]
            
        if get_results is not None:
            
            # get_results=[label_preds, label_trues, label_FEATURE, label_CLS_LOSS]
            
            labels_unlabels = torch.cat([torch.ones_like(get_results[1]),torch.zeros_like(trues)])
            preds = torch.cat([get_results[0], preds],axis=0)
            trues = torch.cat([get_results[1],trues],axis=0)
            FEATURE = torch.cat([get_results[2], FEATURE],axis=0)
            CLS_LOSSES = torch.cat([get_results[3],CLS_LOSSES],axis=0)
        snd_feature = TSNE(learning_rate=20).fit_transform(FEATURE)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

        if len(trues.unique()) != preds.shape[1]:
            plt.figure(figsize=(24, 24))
            plt.subplot(2, 2, 1)
            if get_results is not None:
                for c in trues.unique()[:preds.shape[1]]:
                    plt.scatter(snd_feature[(labels_unlabels == 1) & (trues==c),0], snd_feature[(labels_unlabels == 1) & (trues==c),1],label=f"{c.item()}-label",c=colors[c], marker="o")
                    plt.scatter(snd_feature[(labels_unlabels == 0) & (trues==c),0], snd_feature[(labels_unlabels == 0) & (trues==c),1],label=f"{c.item()}-unlabel",c=colors[c], marker="*")
            else:
                for c in trues.unique()[:preds.shape[1]]:
                    plt.scatter(snd_feature[trues==c,0], snd_feature[trues==c,1],label=c.item(),c=colors[c])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via true labels - IN')

            plt.subplot(2, 2, 2)
            for c in trues.unique()[preds.shape[1]:]:
                plt.scatter(snd_feature[trues==c,0], snd_feature[trues==c,1],label=c.item(),c=colors[c])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via true labels - OOD')

            plt.subplot(2, 2, 3)
            if get_results is not None:
                for idx,c in enumerate(range(preds.shape[1])):
                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 1),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 1),1],
                                label=f"{c}-label",c=colors[idx], marker='o')

                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 0),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 0),1],
                                label=f"{c}-unlabel",c=colors[idx], marker='*')
            else:
                for idx,c in enumerate(range(preds.shape[1])):
                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c),1],
                                label=c,c=colors[idx])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via predicted label - IN(but, this is true)')

            plt.subplot(2, 2, 4)
            for idx,c in enumerate(range(preds.shape[1])):
                plt.scatter(snd_feature[(trues>=preds.shape[1]) & (preds.argmax(1)==c),0],
                            snd_feature[(trues>=preds.shape[1]) & (preds.argmax(1)==c),1],
                            label=c,c=colors[idx])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via predicted label - OOD(but, this is true)')

            plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type={name}.png"))
            plt.close('all')
            
            if get_results is not None:
                plt.scatter(snd_feature[(labels_unlabels == 0) & (trues>=preds.shape[1]),0], snd_feature[(labels_unlabels == 0) & (trues>=preds.shape[1]),1],label="unlabel-ood",c='black', marker="*",s=5,alpha=.5)
                plt.scatter(snd_feature[(labels_unlabels == 0) & (trues<preds.shape[1]),0], snd_feature[(labels_unlabels == 0) & (trues<preds.shape[1]),1],label="unlabel-In",c='blue', marker="*",s=5,alpha=.5)
                plt.scatter(snd_feature[(labels_unlabels == 1),0], snd_feature[(labels_unlabels == 1),1],label="label",c='red', marker="o",s=5,alpha=.5)
                plt.legend()
                plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
                plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
                plt.title('Label or Unlabel')
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=label-or-unlabel.png"))
                plt.close('all')

                plt.scatter(snd_feature[(labels_unlabels == 1),0], snd_feature[(labels_unlabels == 1),1],
                            c=CLS_LOSSES[labels_unlabels==1].cpu().numpy(),
                            marker="o",s=5,
                            cmap='viridis',
                            alpha=.5)
                plt.colorbar()
                plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
                plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
                plt.title('Label Feature with cls loss')
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=only-label-with-cls-loss.png"))
                plt.close('all')

                plt.scatter(snd_feature[(labels_unlabels==0),0], snd_feature[(labels_unlabels==0),1],
                            c=CLS_LOSSES[labels_unlabels==0].cpu().numpy(),
                            marker="o",s=5,
                            cmap='viridis',
                            alpha=.5)
                plt.colorbar()
                plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
                plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
                plt.title('Unlabel Feature with cls loss')
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=only-unlabel-with-cls-loss.png"))
                plt.close('all')

                plt.scatter(snd_feature[(labels_unlabels == 1),0], snd_feature[(labels_unlabels == 1),1],
                            c=preds.softmax(1).max(1)[0][labels_unlabels==1].cpu().numpy(),
                            marker="o",s=5,
                            cmap='viridis',
                            alpha=.5)
                plt.colorbar()
                plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
                plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
                plt.title('Label Feature with conf')
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=only-label-with-conf.png"))
                plt.close('all')

                plt.scatter(snd_feature[(labels_unlabels==0),0], snd_feature[(labels_unlabels==0),1],
                            c=preds.softmax(1).max(1)[0][labels_unlabels==0].cpu().numpy(),
                            marker="o",s=5,
                            cmap='viridis',
                            alpha=.5)
                plt.colorbar()
                plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
                plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
                plt.title('Unlabel Feature with conf')
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=only-unlabel-with-conf.png"))
                plt.close('all')
                
                sns.jointplot(x=preds.softmax(1).max(1)[0][labels_unlabels==0].cpu().numpy(), y=CLS_LOSSES[labels_unlabels==0].cpu().numpy(), marginal_kws=dict(bins=25, fill=False))
                plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type=only-label-with-conf-cls-loss.png"))
                plt.close('all')
                
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            if get_results is not None:
                for c in trues.unique()[:preds.shape[1]]:
                    plt.scatter(snd_feature[(labels_unlabels == 1) & (trues==c),0], snd_feature[(labels_unlabels == 1) & (trues==c),1],label=c.item(),c=colors[c], marker='o')
                    plt.scatter(snd_feature[(labels_unlabels == 0) & (trues==c),0], snd_feature[(labels_unlabels == 0) & (trues==c),1],label=c.item(),c=colors[c], marker='*')
            else:
                for c in trues.unique()[:preds.shape[1]]:
                    plt.scatter(snd_feature[trues==c,0], snd_feature[trues==c,1],label=c.item(),c=colors[c])

            plt.legend()
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via true labels - IN')

            plt.subplot(1, 2, 2)
            if get_results is not None:
                for idx,c in enumerate(range(preds.shape[1])):
                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 1),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 1),1],
                                label=c,c=colors[idx], marker="o")
                    
                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 0),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c) & (labels_unlabels == 0),1],
                                label=c,c=colors[idx], marker="*")
            else:
                for idx,c in enumerate(range(preds.shape[1])):
                    plt.scatter(snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c),0],
                                snd_feature[(trues<preds.shape[1]) & (preds.argmax(1)==c),1],
                                label=c,c=colors[idx])

            plt.legend()
            plt.xlim(snd_feature[:,0].min()*1.05, snd_feature[:,0].max()*1.05)
            plt.ylim(snd_feature[:,1].min()*1.05, snd_feature[:,1].max()*1.05)
            plt.title('Via prediction - IN')

            plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type={name}.png"))
            plt.close('all')

        if name == 'unlabel':
            plt.hist((CLS_LOSS_IN).cpu().numpy(), label='Unlabel-In', alpha=.5, bins=100)
            plt.hist((CLS_LOSS_OOD).cpu().numpy(), label='Unlabel-Ood', alpha=.5, bins=100)
            plt.xlim(0,3)
            plt.legend()
            plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type={name}+Unlabel+Cls+loss.png"))
            plt.close('all')

            plt.hist(CLS_LOSSES[labels_unlabels==1].cpu().numpy(), label='Label', alpha=.5, bins=100)
            plt.hist((CLS_LOSS_IN).cpu().numpy(), label='Unlabel-In', alpha=.5, bins=100)
            plt.hist((CLS_LOSS_OOD).cpu().numpy(), label='Unlabel-Ood', alpha=.5, bins=100)
            plt.xlim(0,3)
            plt.legend()
            plt.savefig(os.path.join(self.ckpt_dir, f"timestamp={time}+type={name}+Label+Unlabel+Cls+loss.png"))
            plt.close('all')
            
        if return_results:
            return preds, trues, FEATURE, CLS_LOSS_IN, IDX