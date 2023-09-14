import collections
import math
import os

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader

from tasks.classification_CaliMATCH import Classification as Task
from utils import RandomSampler, TopKAccuracy
from utils.logging import make_epoch_description


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
            consis_coef,
            consis_coef2,
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
            train_history, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, tau=tau, consis_coef=consis_coef, consis_coef2=consis_coef2, smoothing_proposed=None if epoch==1 else ece_results, n_bins=n_bins)
            eval_history, ece_results = self.evaluate(eval_loader, n_bins=n_bins, train_n_bins=train_n_bins)
            if self.ckpt_dir.split("/")[2]=='cifar10' and enable_plot:
                label_preds, label_trues, label_FEATURE = self.log_plot_history(data_loader=label_loader, time=self.trained_iteration, name="label", return_results=True)
                self.log_plot_history(data_loader=unlabel_loader, time=self.trained_iteration, name="unlabel", get_results=[label_preds, label_trues, label_FEATURE])
                self.log_plot_history(data_loader=open_test_loader, time=self.trained_iteration, name="open+test")

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

    def train(self, label_iterator, unlabel_iterator, iteration, tau, consis_coef, consis_coef2, smoothing_proposed, n_bins):
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
            'label_cross_entropy' : torch.zeros(iteration, device=self.local_rank),
            "unlabeled_cali_wrong_loss": torch.zeros(iteration, device=self.local_rank),
            "unlabeled_cali_in_loss": torch.zeros(iteration, device=self.local_rank),
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
                    label_scaled_logit, unlabel_weak_scaled_logit, _ = full_scaled_logits.split(label_y.size(0))
                    
                    unlabel_confidence, unlabel_pseudo_y = unlabel_weak_scaled_logit.softmax(1).max(1)

                    label_loss = self.loss_function(label_logit, label_y.long())
                    used_unlabeled_index = (unlabel_confidence>tau)

                    result["label_cross_entropy"][i] = label_loss.detach().item()

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
                        label_loss += cali_loss

                    warm_up_coef = consis_coef*math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)

                    if used_unlabeled_index.sum().item() != 0:
                        unlabel_train_logits = unlabel_strong_logit[used_unlabeled_index]
                        unlabel_train_y = nn.functional.one_hot(unlabel_pseudo_y[used_unlabeled_index], num_classes=unlabel_strong_logit.size(1)).float()

                        unlabel_train_noisy_index = torch.rand(unlabel_train_y.size(0)) <= (1-tau)
                        unlabel_train_noisy_index = unlabel_train_noisy_index.to(self.local_rank)
                        
                        if unlabel_train_noisy_index.sum()!=0:
                            unlabel_train_y[unlabel_train_noisy_index] = (torch.ones(unlabel_train_noisy_index.sum(), self.backbone.output.out_features, device=self.local_rank) / self.backbone.output.out_features)
                            unlabeled_losses = -(unlabel_train_logits.log_softmax(1)*unlabel_train_y).sum(1)
                            
                            warm_up_coefs = torch.where(unlabel_train_noisy_index, consis_coef2, warm_up_coef)
                            unlabel_loss = (unlabeled_losses*warm_up_coefs).mean()
                            
                            # just for logging
                            with torch.no_grad():
                                wrong_idx = unlabel_y[used_unlabeled_index]!=unlabel_pseudo_y[used_unlabeled_index]
                                unlabeled_cali_wrong_loss = unlabeled_losses[(wrong_idx) & (unlabel_train_noisy_index)].mean()
                                unlabeled_cali_in_loss = unlabeled_losses[(~wrong_idx) & (unlabel_train_noisy_index)].mean()
                        else:
                            unlabel_loss = self.loss_function(unlabel_train_logits, unlabel_pseudo_y[used_unlabeled_index].long()) * warm_up_coef
                            unlabeled_cali_wrong_loss = torch.zeros(1).to(self.local_rank)
                            unlabeled_cali_in_loss = torch.zeros(1).to(self.local_rank)
                    else:
                        unlabel_loss = torch.zeros(1).to(self.local_rank)
                    loss = label_loss + unlabel_loss

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
                result['ece'][i] = self.get_ece(preds=label_logit.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                if smoothing_proposed is not None:
                    result['cali_loss'][i] = cali_loss.detach()
                if used_unlabeled_index.sum().item() != 0:
                    result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(unlabel_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_ece'][i] = self.get_ece(preds=unlabel_weak_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                              targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                    result["unlabeled_cali_wrong_loss"][i] = unlabeled_cali_wrong_loss.detach()
                    result["unlabeled_cali_in_loss"][i] = unlabeled_cali_in_loss.detach()
                result['warm_up_coef'][i] = warm_up_coef
                result["N_used_unlabeled"][i] = used_unlabeled_index.sum().item()
                result["Temperature"][i] = self.backbone.temperature.item()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {torch.nanmean(v[:i+1][v[:i+1]!=0]):.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
        
        return {k: torch.nanmean(v[v!=0]).item() for k, v in result.items()}, label_iterator, unlabel_iterator