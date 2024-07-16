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
            ema_p,
            ent_loss_ratio,
            use_quantile,
            clip_thresh,
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
        
        # FreeMatch
        self.freematch_mask = FreeMatchThresholingHook(num_classes=self.backbone.class_num,use_quantile=use_quantile,clip_thresh=clip_thresh,momentum=ema_p)
        self.lambda_e = ent_loss_ratio

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
            'freematch_mask_avg': torch.zeros(iteration, device=self.local_rank),
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
                    
                    with torch.no_grad():
                        mask = self.freematch_mask.masking(algorithm=None, logits_x_ulb=logits_x_ulb_w)
                        pseudo_label = logits_x_ulb_w.softmax(1).argmax(-1)
                    
                    # calculate entropy loss
                    if mask.sum() > 0:
                        ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.freematch_mask.p_model, self.freematch_mask.label_hist)
                    else:
                        ent_loss = 0.0
                    
                    unsup_loss = (nn.functional.cross_entropy(logits_x_ulb_s, pseudo_label, reduction='none') * (mask)).mean()
                    warm_up_coef = math.exp(-5 * (1 - min(self.trained_iteration/self.warm_up_end, 1))**2)
                    loss = sup_loss + warm_up_coef*unsup_loss + self.lambda_e*ent_loss

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
                result["freematch_mask_avg"][i] = mask.mean().item()

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

class FreeMatchThresholingHook:
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, use_quantile, clip_thresh, momentum=0.999, *args, **kwargs):
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes

        self.time_p = self.p_model.mean()

        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb):

        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if self.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if self.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        self.p_model = self.p_model 
        self.label_hist = self.label_hist 
        self.time_p = self.time_p 

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask
    
def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val