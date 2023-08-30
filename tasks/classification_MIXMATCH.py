import collections
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            open_test_set,
            save_every,
            consis_coef,
            warm_up_end,
            alpha,
            T,
            K,
            n_bins,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers
        self.alpha, self.T, self.K = alpha, T, K

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
        
        unlabel_loader = DataLoader(train_set[1],batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        eval_loader = DataLoader(eval_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=True)
        test_loader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)
        open_test_loader = DataLoader(open_test_set,batch_size=128,shuffle=False,num_workers=num_workers,drop_last=False,pin_memory=False)

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
            train_history, train_l_iterator, train_u_iterator = self.train(train_l_iterator, train_u_iterator, iteration=save_every, consis_coef=consis_coef, n_bins=n_bins)
            eval_history = self.evaluate(eval_loader, n_bins)
            if self.ckpt_dir.split("/")[2]=='cifar10':
                self.log_plot_history(data_loader=unlabel_loader, time=self.trained_iteration, name="unlabel")
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

                test_history = self.evaluate(test_loader,n_bins=n_bins)
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
        
    def train(self, label_iterator, unlabel_iterator, consis_coef, iteration, n_bins):
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
        
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i in range(iteration):
                with torch.cuda.amp.autocast(self.mixed_precision):
                    l_batch = next(label_iterator)
                    u_batch = next(unlabel_iterator)

                    label_x = l_batch['x'].to(self.local_rank)
                    label_y = l_batch['y'].to(self.local_rank)

                    u_x_hat = [u_batch['weak_img'].to(self.local_rank)] 
                    for k in range(1,self.K):
                        u_x_hat.append(u_batch[f'weak_img_{k+1}'].to(self.local_rank))
                    y_hat = sum([self.predict(u_x_hat[i]).softmax(1) for i in range(len(u_x_hat))]) / self.K
                    y_hat = self.sharpen(y_hat)
                    y_hat = y_hat.repeat(len(u_x_hat), 1).detach()

                    # mixup
                    all_inputs = torch.cat([label_x] + u_x_hat, dim=0)
                    all_targets = torch.cat([F.one_hot(label_y, num_classes=y_hat.size(1)), y_hat], dim=0)

                    l = np.random.beta(self.alpha, self.alpha)
                    l = max(l, 1 - l)

                    idx = torch.randperm(all_inputs.size(0))

                    input_a, input_b = all_inputs, all_inputs[idx]
                    target_a, target_b = all_targets, all_targets[idx]

                    mixed_input = l * input_a + (1 - l) * input_b
                    mixed_target = l * target_a + (1 - l) * target_b

                    mixed_logits = self.predict(mixed_input)
                    label_loss = -torch.mean(torch.sum(F.log_softmax(mixed_logits[:label_y.size(0)], dim=1) * mixed_target[:label_y.size(0)], dim=1))
                    unlabel_loss = torch.mean((mixed_logits[label_y.size(0):].softmax(dim=-1)-mixed_target[label_y.size(0):])**2)
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
                result['warm_up_coef'][i] = warm_up_coef
                result["N_used_unlabeled"][i] = label_y.size(0)

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
    
    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)