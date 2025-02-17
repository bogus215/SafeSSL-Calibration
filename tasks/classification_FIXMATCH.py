import collections
import math
import os
import numpy as np

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
            open_test_set,
            save_every,
            tau,
            consis_coef,
            warm_up_end,
            n_bins,
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

        for epoch in range(1, epochs + 1):

            # training unlabeled data logging
            self.log_unlabeled_data(unlabel_loader=unlabel_loader,current_epoch=epoch)

            # Train & evaluate
            train_history, cls_wise_results, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, tau=tau, consis_coef=consis_coef, n_bins=n_bins)
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

    def train(self, label_iterator, unlabel_iterator, iteration, tau, consis_coef, n_bins):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
            'warm_up_coef': torch.zeros(iteration, device=self.local_rank),
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
                with torch.autocast('cuda', enabled = self.mixed_precision):
                    l_batch = next(label_iterator)
                    u_batch = next(unlabel_iterator)

                    label_x = l_batch['x'].to(self.local_rank)
                    label_y = l_batch['y'].to(self.local_rank)

                    unlabel_weak_x, unlabel_strong_x = u_batch['weak_img'].to(self.local_rank), u_batch['strong_img'].to(self.local_rank)
                    unlabel_y = u_batch['y'].to(self.local_rank)

                    full_logits = self.predict(torch.cat([label_x, unlabel_weak_x, unlabel_strong_x],axis=0))
                    label_logit, unlabel_weak_logit, unlabel_strong_logit = full_logits.split(label_y.size(0))
                    
                    unlabel_confidence, unlabel_pseudo_y = unlabel_weak_logit.softmax(1).max(1)

                    label_loss = self.loss_function(label_logit, label_y.long())
                    used_unlabeled_index = (unlabel_confidence>tau)
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
                result['ece'][i] = self.get_ece(preds=label_logit.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                if used_unlabeled_index.sum().item() != 0:
                    result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(unlabel_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_ece'][i] = self.get_ece(preds=unlabel_weak_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                              targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                result['warm_up_coef'][i] = warm_up_coef
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

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        return {k: v.mean().item() for k, v in result.items()}, cls_wise_results, label_iterator, unlabel_iterator
    
    def log_unlabeled_data(self,unlabel_loader,current_epoch):

        self._set_learning_phase(train=False)
        
        with torch.no_grad():
            with Progress(transient=True, auto_refresh=False) as pg:
                if self.local_rank == 0:
                    task = pg.add_task(f"[bold red] Extracting...", total=len(unlabel_loader))
                for batch_idx, data in enumerate(unlabel_loader):

                    x = data['weak_img'].cuda(self.local_rank)
                    y = data['y'].cuda(self.local_rank)

                    logits = self.predict(x)
                    probs = nn.functional.softmax(logits, 1)
                    select_idx = logits.softmax(1).max(1)[0] > 0.95
                    gt_idx = y < self.backbone.class_num

                    if batch_idx == 0:
                        select_all = select_idx
                        gt_all = gt_idx
                        probs_all, logits_all = probs, logits
                        labels_all = y
                    else:
                        select_all = torch.cat([select_all, select_idx], 0)
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        probs_all, logits_all = torch.cat([probs_all, probs], 0), torch.cat([logits_all, logits], 0)
                        labels_all = torch.cat([labels_all, y], 0)
                        
                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(unlabel_loader)}] "
                        pg.update(task, advance=1., description=desc)
                        pg.refresh()

        select_accuracy = accuracy_score(gt_all.cpu().numpy(), select_all.cpu().numpy()) # positive : inlier, negative : out of distribution
        select_precision = precision_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        selected_idx = torch.arange(0, len(select_all),device=self.local_rank)[select_all]

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
                
class ImageNetClassification(Classification):
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
            warm_up_end,
            n_bins,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        distributed = kwargs.get('distributed')
        
        from ffcv.loader import Loader, OrderOption
        from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Squeeze, RandomHorizontalFlip, RandomBrightness, RandomContrast, RandomSaturation, RandomTranslate, Cutout
        from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
        
        label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True)]
        img_pipeline_weak = [RandomResizedCropRGBImageDecoder((192, 192)), RandomHorizontalFlip(), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD, np.float16)]
        img_pipeline_strong = [RandomResizedCropRGBImageDecoder((192, 192)), RandomHorizontalFlip(), RandomBrightness(magnitude=0.3,p=.25), RandomContrast(magnitude=0.3,p=.25), RandomSaturation(magnitude=0.3,p=.25), RandomTranslate(int(224*0.3)), Cutout(crop_size=50,fill=127), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD,np.float16)]
        img_pipeline_eval = [CenterCropRGBImageDecoder((224, 224),DEFAULT_CROP_RATIO), ToTensor(), ToDevice(torch.device(f"cuda:{self.local_rank}"),non_blocking=True), ToTorchImage(), NormalizeImage(IMAGENET_MEAN,IMAGENET_STD, np.float16)]

        # DataLoader (train, val, test)
        label_loader = Loader(train_set[0],batch_size=batch_size//4,order=OrderOption.RANDOM,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_weak,'label':label_pipeline})
        unlabel_loader = Loader(train_set[1],batch_size=batch_size,order=OrderOption.RANDOM,num_workers=num_workers,drop_last=False,distributed=distributed,pipelines={'image':img_pipeline_weak,'label':label_pipeline, 'image_0':img_pipeline_strong},custom_field_mapper={'image_0':'image'})
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
            train_history, cls_wise_results = self.train(label_loader, unlabel_loader, tau=tau, consis_coef=consis_coef, n_bins=n_bins)
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

    def train(self, label_loader, unlabel_loader, tau, consis_coef, n_bins):
        """Training defined for a single epoch."""

        self._set_learning_phase(train=True)
        iteration=len(unlabel_loader)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'top@5': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@1': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_top@5': torch.zeros(iteration, device=self.local_rank),
            'unlabeled_ece': torch.zeros(iteration, device=self.local_rank),
            'warm_up_coef': torch.zeros(iteration, device=self.local_rank),
            'warm_up_lr': torch.zeros(iteration, device=self.local_rank),
            'N_used_unlabeled': torch.zeros(iteration, device=self.local_rank)
        }
        
        label_iterator = iter(label_loader)
        cls_wise_results = {i:torch.zeros(iteration) for i in range(1000)}

        with Progress(transient=True, auto_refresh=False) as pg:
            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (unlabel_weak_x, unlabel_y, unlabel_strong_x) in enumerate(unlabel_loader):

                warm_up_lr = self.learning_rate*math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warm_up_lr
                try:
                    l_batch = next(label_iterator)
                except:
                    label_iterator = iter(label_loader)
                    l_batch = next(label_iterator)

                label_x, label_y = l_batch[0], l_batch[1]
                with torch.autocast('cuda', enabled = self.mixed_precision):

                    full_logits = self.predict(torch.cat([label_x, unlabel_weak_x, unlabel_strong_x],axis=0))
                    label_logit, unlabel_logits = full_logits[:label_x.size(0)], full_logits[label_x.size(0):]
                    unlabel_weak_logit, unlabel_strong_logit = unlabel_logits.split(unlabel_weak_x.size(0))
                    
                    unlabel_confidence, unlabel_pseudo_y = unlabel_weak_logit.softmax(1).max(1)

                    label_loss = self.loss_function(label_logit, label_y)
                    used_unlabeled_index = (unlabel_confidence>tau)
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
                result['top@5'][i] = TopKAccuracy(k=5)(label_logit, label_y).detach()
                result['ece'][i] = self.get_ece(preds=label_logit.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                if used_unlabeled_index.sum().item() != 0:
                    result['unlabeled_top@1'][i] = TopKAccuracy(k=1)(unlabel_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_top@5'][i] = TopKAccuracy(k=5)(unlabel_weak_logit[used_unlabeled_index], unlabel_y[used_unlabeled_index]).detach()
                    result['unlabeled_ece'][i] = self.get_ece(preds=unlabel_weak_logit[used_unlabeled_index].softmax(dim=1).detach().cpu().numpy(),
                                                              targets=unlabel_y[used_unlabeled_index].cpu().numpy(),n_bins=n_bins, plot=False)[0]
                result['warm_up_coef'][i] = warm_up_coef
                result['warm_up_lr'][i] = warm_up_lr
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

        return {k: v.mean().item() for k, v in result.items()}, cls_wise_results

    @torch.no_grad()
    def evaluate(self, data_loader, n_bins):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'loss': torch.zeros(steps, device=self.local_rank),
            'top@1': torch.zeros(1, device=self.local_rank),
            'top@5': torch.zeros(1, device=self.local_rank),
            'ece': torch.zeros(1, device=self.local_rank)
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

                result['loss'][i] = loss
                true.append(y.cpu())
                pred.append(logits.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{steps}]: " + f" loss : {result['loss'][:i+1].mean():.4f} |" + f" top@1 : {TopKAccuracy(k=1)(logits, y).detach():.4f} |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        # preds, pred are logit vectors
        preds, trues = torch.cat(pred,axis=0), torch.cat(true,axis=0)
        result['top@1'][0] = TopKAccuracy(k=1)(preds, trues)
        result['top@5'][0] = TopKAccuracy(k=5)(preds, trues)

        ece_results = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets=trues.numpy(), n_bins=n_bins, plot=False)
        result['ece'][0] = ece_results[0]

        return {k: v.mean().item() for k, v in result.items()}
                
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256