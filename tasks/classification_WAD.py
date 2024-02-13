import collections
import os
import random
import numpy as np

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader

from datasets.svhn import Selcted_DATA
from datasets.transforms import SemiAugment
from tasks.classification import Classification as Task
from tasks.classification_OPENMATCH import DistributedSampler
from utils import TopKAccuracy
from utils.logging import make_epoch_description


class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(self,
            train_set,
            eval_set,
            test_set,
            save_every,
            sim_lambda, 
            temperature,
            datasets,
            trans_kwargs,
            n_bins,
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

        best_eval_acc = -float('inf')
        best_epoch    = 0

        # pre-train
        for epoch in range(1, epochs//2 + 1):
            
            train_history = self.pretrain(l_loader, u_loader, sim_lambda, temperature) 
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

        del l_sampler, l_loader, u_sampler, u_loader

        import gc
        gc.collect()
                
        # unlabeled → labeled 옮겨 간 unlabeled data 정보 저장
        query_idxs = np.array([], dtype=np.int64)
        query_labels_pseudo = np.array([], dtype=np.int64)

        # Selected loader
        l_sampler = DistributedSampler(dataset=train_set[0], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        selected_l_loader = DataLoader(train_set[0], batch_size=self.batch_size//2, sampler=l_sampler,num_workers=num_workers,drop_last=False,pin_memory=False) # 전체 labeled

        u_sampler = DistributedSampler(dataset=train_set[1], num_replicas=1, rank=self.local_rank, num_samples=num_samples)
        selected_u_loader = DataLoader(train_set[1],batch_size=self.batch_size//2, sampler=u_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)
        
        pseudo_index, u_pseudo_label_logits, u_weights = self.knowledge_generation(train_set[0], train_set[1])

        for epoch in range(epochs//2 + 1, epochs + 1):
            train_history = self.train(selected_l_loader, selected_u_loader, pseudo_index=pseudo_index, u_pseudo_label_logits=u_pseudo_label_logits, u_weights=u_weights, n_bins=n_bins)
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
            
            alpha_epoch = {60:0.1,70:0.07,80:0.04,90:0.01,95:0}
            if epoch in alpha_epoch.keys():

                alpha = alpha_epoch[epoch]
                
                # Obtain the predict logits of unlabeled instances
                predict_indices, predict_logit = self.calculate_sample_predict(train_set[1])
                query_idx, query_label_pseudo = self.calculate_loss_logits_pseudo(u_pseudo_label_logits, pseudo_index, predict_logit, predict_indices, alpha, len(predict_logit))

                # labeled data += alpha% unlabeled data 
                query_idxs = np.concatenate([query_idxs, query_idx])
                query_labels_pseudo = np.concatenate([query_labels_pseudo, query_label_pseudo])

                left_u_idxs = list(np.setdiff1d(list(range(len(datasets['u_train']['images']))), list(query_idxs)))
                
                new_l_dataset = datasets['l_train'].copy()
                new_ul_dataset = datasets['u_train'].copy()

                new_l_dataset['images'] = np.concatenate([datasets['l_train']['images'], datasets['u_train']['images'][query_idxs]])
                new_l_dataset['labels'] = np.concatenate([np.array(datasets['l_train']['labels']), np.array(query_labels_pseudo)]) # pesudo label로 추가
                new_ul_dataset['images'] = np.array(datasets['u_train']['images'])[left_u_idxs]
                new_ul_dataset['labels'] = np.array(datasets['u_train']['labels'])[left_u_idxs]
                
                selected_labeled_set = Selcted_DATA(dataset=new_l_dataset, name='train_lb', transform=SemiAugment(**trans_kwargs))
                selected_unlabeled_set = Selcted_DATA(dataset=new_ul_dataset, name='train_ulb',transform=SemiAugment(**trans_kwargs))
                
                l_sampler = DistributedSampler(dataset=selected_labeled_set, num_replicas=1, rank=self.local_rank, num_samples=num_samples)
                u_sampler = DistributedSampler(dataset=selected_unlabeled_set, num_replicas=1, rank=self.local_rank, num_samples=num_samples)
                
                selected_l_loader = DataLoader(selected_labeled_set, batch_size=self.batch_size//2, sampler=l_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)
                selected_u_loader = DataLoader(selected_unlabeled_set, batch_size=self.batch_size//2, sampler=u_sampler,num_workers=num_workers,drop_last=False,pin_memory=False)

                pseudo_index, u_pseudo_label_logits, u_weights = self.knowledge_generation(train_set[0], train_set[1])


    def pretrain(self, label_loader, unlabel_loader, sim_lambda, temperature):
        """Training defined for a single epoch."""

        iteration = len(label_loader)

        self._set_learning_phase(train=True)
        result = {
            'simclr_loss': torch.zeros(iteration, device=self.local_rank),
            'cls_loss': torch.zeros(iteration, device=self.local_rank),
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
        }
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb) in enumerate(zip(label_loader, unlabel_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    
                    # 모든 데이터 사용 = all unlabeled
                    x_lb  = data_lb["x_lb_w_0"].to(self.local_rank)     
                    x_lb_1  = data_lb["x_lb_w_1"].to(self.local_rank)
                    label_y = data_lb['y_lb'].to(self.local_rank)

                    x_ulb_w  = data_ulb["x_ulb_w_0"].to(self.local_rank)     
                    x_ulb_w_1  = data_ulb["x_ulb_w_1"].to(self.local_rank) 
                    
                    feat = self.backbone.get_only_feat(torch.cat((x_lb,x_ulb_w,x_lb_1,x_ulb_w_1)))

                    proj_feat = normalize(self.backbone.simclr_classifier(feat))
                    sim_matrix = torch.mm(proj_feat, proj_feat.t()) # get_similarity_matrix
                    sim_loss = NT_xent(sim_matrix, temperature=temperature) * sim_lambda

                    label_logit = self.backbone.output(torch.cat((feat[:x_lb.size(0)],feat[x_lb.size(0)*2:-x_ulb_w_1.size(0)])))

                    classification_loss = self.loss_function(label_logit,label_y.repeat(2))
                    loss = sim_loss + classification_loss
                  
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()
                
                result['simclr_loss'][i] = sim_loss.detach()
                result['cls_loss'][i] = classification_loss.detach()
                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(label_logit, label_y.repeat(2)).detach()

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
           
    @torch.no_grad()
    def knowledge_generation(self, label_dataset, unlabel_dataset):

        self._set_learning_phase(False)

        label_loader = DataLoader(dataset=label_dataset,batch_size=128,drop_last=False,shuffle=False,num_workers=4)
        unlabel_loader = DataLoader(dataset=unlabel_dataset,batch_size=128,drop_last=False,shuffle=False,num_workers=4)
        
        index_l, feats_l, label_l = [], [], [] 
        index_u, feats_u  = [], []

        for data_lb in label_loader:
            with torch.cuda.amp.autocast(self.mixed_precision):

                idx_lb = data_lb['idx_lb'].to(self.local_rank)
                x_lb = data_lb['x_lb'].to(self.local_rank)       
                label_y = data_lb['y_lb'].to(self.local_rank)                    
         
                feat_l = self.backbone.simclr_classifier(self.backbone.get_only_feat(x_lb))
                    
                index_l.append(idx_lb) 
                feats_l.append(feat_l)
                label_l.append(label_y)

        for data_ulb in unlabel_loader:
            with torch.cuda.amp.autocast(self.mixed_precision):

                idx_ulb = data_ulb['idx_ulb'].to(self.local_rank)
                x_ulb_w  = data_ulb["x_ulb_w_0"].to(self.local_rank)

                feat_ul = self.backbone.simclr_classifier(self.backbone.get_only_feat(x_ulb_w))
                    
                index_u.append(idx_ulb)                    
                feats_u.append(feat_ul)

        index_l, feats_l, label_l = torch.cat(index_l), torch.cat(feats_l), torch.cat(label_l)
        index_u, feats_u  = torch.cat(index_u), torch.cat(feats_u)

        # labeled data를 class별로 feature 산출
        feats_group = [[] for _ in range(label_l.max().item() + 1)]
        for i, class_label in enumerate(label_l):
            feats_group[class_label.item()].append(feats_l[i])
        feats_group = [torch.stack(feats) if feats else torch.tensor([]) for feats in feats_group]

        unlabeled_group_score = []
        for i in range(len(feats_group)):
            axis = normalize(feats_group[i], dim=1)
            unlabeled_group_score.append(self.get_scores(axis, feats_u))

        labels_logits = []
        weights = []

        # unlabeled 갯수 만큼
        for j in range(len(unlabeled_group_score[0])): 
            similarity_logits = []
            for i in range(len(unlabeled_group_score)): 
                similarity_logits.append(unlabeled_group_score[i][j])
            labels_logits.append(torch.stack(similarity_logits))
            sort_logit = sorted(torch.stack(similarity_logits))
            weights.append(sort_logit[-1] * (1 - sort_logit[-2] / sort_logit[-1])) # weights 계산

        return index_u, labels_logits, torch.stack(weights)

    def get_scores(self, axis, feats):
        
        N = feats.size(0)
        max_sim = []
        for f_sim in feats:
            f_sim = normalize(f_sim, dim=0)
            value_sim, _ = ((f_sim * axis).sum(dim=1)).sort(descending=True)
            simi_score = value_sim.max().item()
            max_sim.append(simi_score)
        max_sim = torch.tensor(max_sim)

        assert max_sim.dim() == 1 and max_sim.size(0) == N  # (N)

        return max_sim.cpu()
        
    def train(self, selected_l_loader, selected_unlabel_loader, pseudo_index, u_pseudo_label_logits, u_weights, n_bins):
        """Training defined for a single epoch."""

        iteration = len(selected_l_loader)
        
        self._set_learning_phase(train=True)
        result = {
            'loss': torch.zeros(iteration, device=self.local_rank),
            'top@1': torch.zeros(iteration, device=self.local_rank),
            'ece': torch.zeros(iteration, device=self.local_rank),
            'loss_ce_l': torch.zeros(iteration, device=self.local_rank),
            'loss_ce_ul': torch.zeros(iteration, device=self.local_rank)
        }
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb_selected) in enumerate(zip(selected_l_loader, selected_unlabel_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    label_x = data_lb['x_lb'].to(self.local_rank)
                    label_y = data_lb['y_lb'].to(self.local_rank)

                    x_ulb_w = data_ulb_selected["x_ulb_w_0"].to(self.local_rank)
                    ulb_idx = data_ulb_selected["idx_ulb"].to(self.local_rank)
                    
                    # Label alignment
                    pseudo_label, weight_ = self.alignment(ulb_idx, pseudo_index, u_pseudo_label_logits, u_weights, weight=True)
                    pseudo_label, weight_ = pseudo_label.to(self.local_rank), weight_.to(self.local_rank)
                    
                    labeled_preds_logits = self.predict(label_x)
                    labeled_loss = self.loss_function(labeled_preds_logits, label_y)
                    
                    unlabeled_preds_logits = self.predict(x_ulb_w)
                    unlabeled_loss_weight = weight_ * torch.nn.functional.cross_entropy(unlabeled_preds_logits, pseudo_label, reduction='none')
                    unlabeled_loss = torch.mean(unlabeled_loss_weight)
                    
                    loss = labeled_loss + unlabeled_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result['loss'][i] = loss.detach()
                result['top@1'][i] = TopKAccuracy(k=1)(labeled_preds_logits, label_y).detach()             
                result['ece'][i] = self.get_ece(preds=labeled_preds_logits.softmax(dim=1).detach().cpu().numpy(), targets=label_y.cpu().numpy(), n_bins=n_bins, plot=False)[0]
                result['loss_ce_l'][i] = labeled_loss.detach()
                result['loss_ce_ul'][i] = unlabeled_loss.detach()

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
     
    def alignment(self, base_index, index, label, weights=[], weight=False):
        label = torch.stack(label)

        # 초기 unlabeled loader와 변형된 loader의 index가 다를 경우 맞춰줌
        align_weight = []
        align_label = []
        if weight:
            for j in range(len(base_index)):
                i = torch.where(index == base_index[j])[0].cpu()
                align_label.append(torch.argmax(label[i]))
                align_weight.append(weights[i])
            align_weight = torch.tensor(align_weight)
            align_label = torch.stack(align_label)
        else:
            for j in range(len(base_index)):
                i = index.index(base_index[j])
                align_label.append(label[i])
        return align_label, align_weight

    @torch.no_grad()
    def calculate_sample_predict(self, unlabel_dataset):

        loader = DataLoader(dataset=unlabel_dataset,
                            batch_size=128,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

        self._set_learning_phase(False)
        all_preds_logits, all_indices = [], []

        for data_ulb_selected in loader:

            with torch.cuda.amp.autocast(self.mixed_precision):
                x_ulb_w = data_ulb_selected["x_ulb_w_0"].to(self.local_rank)
                ulb_idx = data_ulb_selected["idx_ulb"].to(self.local_rank)
                class_preds = self.predict(x_ulb_w)

            all_indices.extend(ulb_idx)
            all_preds_logits.extend(class_preds)
        
        all_indices, all_preds_logits = torch.stack(all_indices), torch.stack(all_preds_logits)

        return all_indices, all_preds_logits

    def calculate_loss_logits_pseudo(self, label_logits, pseudo_index, predict_logit, predict_index, alpha, u_number):
            ground_ = torch.argmax(torch.stack(label_logits), dim=1)
            predict_ = predict_logit
            arr_s = np.array(pseudo_index.cpu())
            arr_p = np.array(predict_index.cpu())
            sp_index = (arr_s == arr_p[:, None]).argmax(1)
            arr_ground = np.array(ground_)
            pseudo_label = list(arr_ground[sp_index])
            ground_pseudo = torch.tensor(arr_ground[sp_index]).cuda()

            loss = torch.nn.functional.cross_entropy(predict_,ground_pseudo, reduction='none')  # return the gap between predict logit and one hot label
            loss = loss * (-1)

            _, query_inside = torch.topk(loss, int(alpha * u_number))
            query_inside = query_inside.cpu().data

            reliable_label = np.asarray(pseudo_label)[query_inside]
            reliable_indices = np.asarray(predict_index.cpu())[query_inside]

            return reliable_indices, reliable_label
            
def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    device = sim_matrix.device
    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix
    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)