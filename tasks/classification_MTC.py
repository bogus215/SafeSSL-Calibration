import collections
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from rich.progress import Progress
from skimage.filters import threshold_otsu
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from tasks.classification import Classification as Task
from utils import TopKAccuracy
from utils.logging import make_epoch_description


class Classification(Task):
    def __init__(self, backbone: nn.Module):
        super(Classification, self).__init__(backbone)

    def run(
        self,
        train_set,
        domain_set,
        eval_set,
        test_set,
        n_bins,
        save_every,
        T,
        alpha,
        lambda_u,
        **kwargs,
    ):  # pylint: disable=unused-argument

        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader
        epochs = self.iterations // save_every
        per_epoch_steps = self.iterations // epochs
        num_samples = per_epoch_steps * self.batch_size // 2

        l_sampler = DistributedSampler(
            dataset=train_set[0],
            num_replicas=1,
            rank=self.local_rank,
            num_samples=num_samples,
        )
        l_loader = DataLoader(
            train_set[0],
            batch_size=self.batch_size // 2,
            sampler=l_sampler,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )

        d_sampler = DistributedSampler(
            dataset=domain_set,
            num_replicas=1,
            rank=self.local_rank,
            num_samples=num_samples,
        )
        domain_loader = DataLoader(
            domain_set,
            batch_size=self.batch_size // 2,
            sampler=d_sampler,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )

        eval_loader = DataLoader(
            eval_set,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
        )

        # Logging
        logger = kwargs.get("logger", None)
        enable_plot = kwargs.get("enable_plot", False)

        # Supervised training
        best_eval_acc = -float("inf")
        best_epoch = 0

        threshold = 0.5
        for epoch in range(1, epochs // 10 + 1):

            if (epoch - 1) % 10 == 0:
                self.logging_unlabeled_dataset(
                    unlabeled_dataset=train_set[1],
                    current_epoch=((epoch - 1) // 10) + 1,
                    threshold=threshold,
                )

            train_history, threshold = self.domain_train(
                d_loader=domain_loader, l_loader=l_loader, threshold=threshold
            )
            eval_history = self.evaluate(eval_loader, n_bins)

            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]["train"] = v1
                try:
                    v2 = eval_history[k]
                    epoch_history[k]["eval"] = v2
                except KeyError:
                    continue

            if (epoch - 1) % 10 == 0:

                # Write TensorBoard summary
                if self.writer is not None:
                    for k, v in epoch_history.items():
                        for k_, v_ in v.items():
                            self.writer.add_scalar(
                                f"{k}_{k_}", v_, global_step=((epoch - 1) // 10) + 1
                            )
                    if self.scheduler is not None:
                        lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar(
                            "lr", lr, global_step=((epoch - 1) // 10) + 1
                        )

            # Save best model checkpoint and Logging
            eval_acc = eval_history["top@1"]

            if logger is not None and eval_acc == 1:
                logger.info("Eval acc == 1 --> Stop training")
                break

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader, n_bins)
                for k, v1 in test_history.items():
                    epoch_history[k]["test"] = v1

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

        for epoch in range(epochs // 10 + 1, epochs + 1):

            if (epoch - 1) % 10 == 0:
                self.logging_unlabeled_dataset(
                    unlabeled_dataset=train_set[1],
                    current_epoch=((epoch - 1) // 10) + 1,
                    threshold=threshold,
                )

            # Selection related to unlabeled data
            threshold = self.exclude_dataset(
                unlabeled_dataset=train_set[1],
                selected_dataset=train_set[-1],
                threshold=threshold,
            )

            # Train & evaluate
            u_sel_sampler = DistributedSampler(
                dataset=train_set[-1],
                num_replicas=1,
                rank=self.local_rank,
                num_samples=num_samples,
            )
            selected_u_loader = DataLoader(
                train_set[-1],
                sampler=u_sel_sampler,
                batch_size=self.batch_size // 2,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=False,
                shuffle=False,
            )

            train_history = self.train(
                label_loader=l_loader,
                selected_unlabel_loader=selected_u_loader,
                domain_loader=domain_loader,
                T=T,
                alpha=alpha,
                lambda_u=lambda_u,
            )
            eval_history = self.evaluate(eval_loader, n_bins)

            if enable_plot:
                raise NotImplementedError

            epoch_history = collections.defaultdict(dict)
            for k, v1 in train_history.items():
                epoch_history[k]["train"] = v1
                try:
                    v2 = eval_history[k]
                    epoch_history[k]["eval"] = v2
                except KeyError:
                    continue

            if (epoch - 1) % 10 == 0:

                # Write TensorBoard summary
                if self.writer is not None:
                    for k, v in epoch_history.items():
                        for k_, v_ in v.items():
                            self.writer.add_scalar(
                                f"{k}_{k_}", v_, global_step=((epoch - 1) // 10) + 1
                            )
                    if self.scheduler is not None:
                        lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar(
                            "lr", lr, global_step=((epoch - 1) // 10) + 1
                        )

            # Save best model checkpoint and Logging
            eval_acc = eval_history["top@1"]

            if logger is not None and eval_acc == 1:
                logger.info("Eval acc == 1 --> Stop training")
                break

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_epoch = epoch
                if self.local_rank == 0:
                    ckpt = os.path.join(self.ckpt_dir, "ckpt.best.pth.tar")
                    self.save_checkpoint(ckpt, epoch=epoch)

                test_history = self.evaluate(test_loader, n_bins)
                for k, v1 in test_history.items():
                    epoch_history[k]["test"] = v1

            # Write logs
            log = make_epoch_description(
                history=epoch_history,
                current=epoch,
                total=epochs,
                best=best_epoch,
            )
            if logger is not None:
                logger.info(log)

    def exclude_dataset(self, unlabeled_dataset, selected_dataset, threshold):

        loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=128,
            drop_last=False,
            shuffle=False,
            num_workers=4,
        )

        self._set_learning_phase(train=False)

        with torch.no_grad():
            with Progress(transient=True, auto_refresh=False) as pg:
                if self.local_rank == 0:
                    task = pg.add_task(f"[bold red] Extracting...", total=len(loader))
                for batch_idx, data in enumerate(loader):

                    x = data["x_ulb_w_0"].cuda(self.local_rank)
                    y = data["y_ulb"].cuda(self.local_rank)

                    feat = self.backbone.get_only_feat(x)
                    domain_logit = self.backbone.domain_classifier(feat)

                    score = domain_logit.sigmoid().squeeze()

                    select_idx = score > threshold
                    gt_idx = y < self.backbone.class_num

                    if batch_idx == 0:
                        select_all = select_idx
                        gt_all = gt_idx
                        score_all = score
                    else:
                        select_all = torch.cat([select_all, select_idx], 0)
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        score_all = torch.cat([score_all, score], 0)

                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(loader)}] "
                        pg.update(task, advance=1.0, description=desc)
                        pg.refresh()

        selected_idx = torch.arange(0, len(select_all), device=self.local_rank)[
            select_all
        ]

        # Write TensorBoard summary
        if len(selected_idx) > 0:
            selected_dataset.set_index(selected_idx)

        return threshold_otsu(score_all.cpu().numpy())

    def domain_train(self, d_loader, l_loader, threshold):

        iteration = len(d_loader)

        self._set_learning_phase(train=True)

        result = {
            "domain_loss": torch.zeros(iteration, device=self.local_rank),
            "domain_top@1": torch.zeros(iteration, device=self.local_rank),
            "top@1": torch.zeros(iteration, device=self.local_rank),
        }

        pred_soft_labels = torch.zeros(len(d_loader.dataset), device=self.local_rank)
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (d_data, l_data) in enumerate(zip(d_loader, l_loader)):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    weak_img = d_data["weak_img"].to(self.local_rank)
                    target = d_data["target"].to(self.local_rank)
                    soft_label = d_data["soft_label"].to(self.local_rank)
                    idx = d_data["idx"].to(self.local_rank)

                    x_lb = l_data["x_lb"].to(self.local_rank)
                    y_lb = l_data["y_lb"].to(self.local_rank)

                    logit, feat = self.get_feature(torch.cat([weak_img, x_lb]))

                    domain_logit = self.backbone.domain_classifier(feat)
                    domain_loss = nn.functional.binary_cross_entropy_with_logits(
                        domain_logit.squeeze(),
                        torch.cat((soft_label, torch.ones_like(soft_label))),
                    )
                    sup_loss = self.loss_function(logit[-x_lb.size(0) :], y_lb.long())

                    loss = sup_loss + domain_loss

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                result["domain_loss"][i] = domain_loss.detach()
                result["domain_top@1"][i] = (
                    (target < self.backbone.class_num)
                    == (
                        domain_logit[: weak_img.size(0)].sigmoid() > threshold
                    ).squeeze()
                ).sum() / weak_img.size(0)
                result["top@1"][i] = TopKAccuracy(k=1)(
                    logit[-x_lb.size(0) :], y_lb
                ).detach()

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

                pred_soft_labels[idx] = (
                    domain_logit[: weak_img.size(0)]
                    .sigmoid()
                    .squeeze()
                    .detach()
                    .float()
                )

        d_loader.dataset.label_update(pred_soft_labels)

        return {k: v.mean().item() for k, v in result.items()}, threshold_otsu(
            pred_soft_labels[len(l_loader.dataset) :].cpu().numpy()
        )

    def train(
        self, label_loader, selected_unlabel_loader, domain_loader, T, alpha, lambda_u
    ):
        """Training defined for a single epoch."""

        iteration = len(label_loader)

        self._set_learning_phase(train=True)
        result = {
            "loss": torch.zeros(iteration, device=self.local_rank),
            "top@1": torch.zeros(iteration, device=self.local_rank),
        }

        pred_soft_labels = torch.zeros(
            len(domain_loader.dataset), device=self.local_rank
        )
        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Training...", total=iteration)

            for i, (data_lb, data_ulb_selected, data_domain) in enumerate(
                zip(label_loader, selected_unlabel_loader, domain_loader)
            ):
                with torch.cuda.amp.autocast(self.mixed_precision):

                    label_x = data_lb["x_lb"].to(self.local_rank)
                    label_y = data_lb["y_lb"].to(self.local_rank)

                    unlabel_weak_x = data_ulb_selected["x_ulb_w"].to(self.local_rank)
                    unlabel_weak_x_1 = data_ulb_selected["x_ulb_w_1"].to(
                        self.local_rank
                    )

                    d_label_x = data_domain["weak_img"].to(self.local_rank)
                    d_soft_label = data_domain["soft_label"].to(self.local_rank)
                    d_indexs = data_domain["idx"].to(self.local_rank)

                    logits, features = self.get_feature(
                        torch.cat(
                            (label_x, d_label_x, unlabel_weak_x, unlabel_weak_x_1)
                        )
                    )
                    domain_logit = self.backbone.domain_classifier(
                        features[: 2 * label_x.size(0)]
                    )

                    L_d = nn.functional.binary_cross_entropy_with_logits(
                        domain_logit.squeeze(),
                        torch.cat((torch.ones_like(d_soft_label), d_soft_label)),
                    )

                    one_hot_label_y = torch.nn.functional.one_hot(
                        label_y, self.backbone.class_num
                    )

                    with torch.no_grad():
                        # compute guessed labels of unlabel samples
                        outputs_u, outputs_u2 = logits[
                            -2 * unlabel_weak_x.size(0) :
                        ].chunk(2)
                        p = (
                            torch.softmax(outputs_u, dim=1)
                            + torch.softmax(outputs_u2, dim=1)
                        ) / 2
                        pt = p ** (1 / T)
                        targets_u = pt / pt.sum(dim=1, keepdim=True)
                        targets_u = targets_u.detach()

                    all_inputs = torch.cat(
                        [label_x, unlabel_weak_x, unlabel_weak_x_1], dim=0
                    )
                    all_targets = torch.cat(
                        [one_hot_label_y, targets_u, targets_u], dim=0
                    )

                    l = np.random.beta(alpha, alpha)
                    l = max(l, 1 - l)
                    idx = torch.randperm(all_inputs.size(0))

                    input_a, input_b = all_inputs, all_inputs[idx]
                    target_a, target_b = all_targets, all_targets[idx]

                    mixed_input = l * input_a + (1 - l) * input_b
                    mixed_target = l * target_a + (1 - l) * target_b

                    logits_full = self.predict(mixed_input)
                    logits_x = logits_full[: label_x.size(0)]
                    logits_u = logits_full[-label_x.size(0) :]

                    L_x, L_u = mixmatch_loss(
                        logits_x,
                        mixed_target[: label_x.size(0)],
                        logits_u,
                        mixed_target[-label_x.size(0) :],
                    )

                    loss = L_d + L_x + lambda_u * L_u

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.optimizer.zero_grad()

                result["loss"][i] = loss.detach()
                result["top@1"][i] = TopKAccuracy(k=1)(
                    logits[: label_x.size(0)], label_y
                ).detach()

                pred_soft_labels[d_indexs] = (
                    domain_logit[label_x.size(0) :].sigmoid().squeeze().detach().float()
                )

                if self.local_rank == 0:
                    desc = f"[bold green] [{i+1}/{iteration}]: "
                    for k, v in result.items():
                        desc += f" {k} : {v[:i+1].mean():.4f} |"
                    pg.update(task, advance=1.0, description=desc)
                    pg.refresh()

                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()

        domain_loader.dataset.label_update(pred_soft_labels)

        return {k: v.mean().item() for k, v in result.items()}

    def get_feature(self, x: torch.FloatTensor):
        """Make a prediction provided a batch of samples."""
        return self.backbone(x, True)

    def logging_unlabeled_dataset(self, unlabeled_dataset, current_epoch, threshold):

        loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=128,
            drop_last=False,
            shuffle=False,
            num_workers=4,
        )

        self._set_learning_phase(train=False)

        with torch.no_grad():
            with Progress(transient=True, auto_refresh=False) as pg:
                if self.local_rank == 0:
                    task = pg.add_task(f"[bold red] Extracting...", total=len(loader))
                for batch_idx, data in enumerate(loader):

                    x = data["x_ulb_w_0"].cuda(self.local_rank)
                    y = data["y_ulb"].cuda(self.local_rank)

                    logit, feat = self.get_feature(x)
                    domain_logit = self.backbone.domain_classifier(feat)

                    score = domain_logit.sigmoid().squeeze()

                    select_idx = score > threshold
                    gt_idx = y < self.backbone.class_num

                    if batch_idx == 0:
                        select_all = select_idx
                        gt_all = gt_idx
                        score_all = score
                        logits_all = logit
                        labels_all = y
                    else:
                        select_all = torch.cat([select_all, select_idx], 0)
                        gt_all = torch.cat([gt_all, gt_idx], 0)
                        score_all = torch.cat([score_all, score], 0)
                        logits_all = torch.cat([logits_all, logit], 0)
                        labels_all = torch.cat([labels_all, y], 0)

                    if self.local_rank == 0:
                        desc = f"[bold pink] Extracting .... [{batch_idx+1}/{len(loader)}] "
                        pg.update(task, advance=1.0, description=desc)
                        pg.refresh()

        select_accuracy = accuracy_score(
            gt_all.cpu().numpy(), select_all.cpu().numpy()
        )  # positive : inlier, negative : out of distribution
        select_precision = precision_score(
            gt_all.cpu().numpy(), select_all.cpu().numpy()
        )
        select_recall = recall_score(gt_all.cpu().numpy(), select_all.cpu().numpy())
        select_f1 = f1_score(gt_all.cpu().numpy(), select_all.cpu().numpy())

        selected_idx = torch.arange(0, len(select_all), device=self.local_rank)[
            select_all
        ]

        probs_all = logits_all.softmax(-1)

        # Write TensorBoard summary
        if self.writer is not None:
            self.writer.add_scalar(
                "Selected accuracy", select_accuracy, global_step=current_epoch
            )
            self.writer.add_scalar(
                "Selected precision", select_precision, global_step=current_epoch
            )
            self.writer.add_scalar(
                "Selected recall", select_recall, global_step=current_epoch
            )
            self.writer.add_scalar("Selected f1", select_f1, global_step=current_epoch)
            self.writer.add_scalar(
                "Selected ratio",
                len(selected_idx) / len(select_all),
                global_step=current_epoch,
            )

            self.writer.add_scalar(
                "In distribution: ECE",
                self.get_ece(
                    probs_all[gt_all].cpu().numpy(), labels_all[gt_all].cpu().numpy()
                )[0],
                global_step=current_epoch,
            )
            self.writer.add_scalar(
                "In distribution: ACC",
                TopKAccuracy(k=1)(logits_all[gt_all], labels_all[gt_all]).item(),
                global_step=current_epoch,
            )

            if ((gt_all) & (probs_all.max(1)[0] >= 0.95)).sum() > 0:
                self.writer.add_scalar(
                    "In distribution over conf 0.95: ECE",
                    self.get_ece(
                        probs_all[(gt_all) & (probs_all.max(1)[0] >= 0.95)]
                        .cpu()
                        .numpy(),
                        labels_all[(gt_all) & (probs_all.max(1)[0] >= 0.95)]
                        .cpu()
                        .numpy(),
                    )[0],
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "In distribution over conf 0.95: ACC",
                    TopKAccuracy(k=1)(
                        logits_all[(gt_all) & (probs_all.max(1)[0] >= 0.95)],
                        labels_all[(gt_all) & (probs_all.max(1)[0] >= 0.95)],
                    ).item(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Selected ratio of i.d over conf 0.95",
                    ((gt_all) & (probs_all.max(1)[0] >= 0.95)).sum() / gt_all.sum(),
                    global_step=current_epoch,
                )

            if ((gt_all) & (select_all)).sum() > 0:
                self.writer.add_scalar(
                    "In distribution under ood score 0.5: ECE",
                    self.get_ece(
                        probs_all[(gt_all) & (select_all)].cpu().numpy(),
                        labels_all[(gt_all) & (select_all)].cpu().numpy(),
                    )[0],
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "In distribution under ood score 0.5: ACC",
                    TopKAccuracy(k=1)(
                        logits_all[(gt_all) & (select_all)],
                        labels_all[(gt_all) & (select_all)],
                    ).item(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Selected ratio of i.d under ood score 0.5",
                    ((gt_all) & (select_all)).sum() / gt_all.sum(),
                    global_step=current_epoch,
                )

            if (probs_all.max(1)[0] >= 0.95).sum() > 0:
                self.writer.add_scalar(
                    "Seen-class ratio over conf 0.95",
                    (
                        labels_all[(probs_all.max(1)[0] >= 0.95)]
                        < self.backbone.class_num
                    ).sum()
                    / (probs_all.max(1)[0] >= 0.95).sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class ratio over conf 0.95",
                    (
                        labels_all[(probs_all.max(1)[0] >= 0.95)]
                        >= self.backbone.class_num
                    ).sum()
                    / (probs_all.max(1)[0] >= 0.95).sum(),
                    global_step=current_epoch,
                )

                self.writer.add_scalar(
                    "Seen-class over conf 0.95",
                    (
                        labels_all[(probs_all.max(1)[0] >= 0.95)]
                        < self.backbone.class_num
                    ).sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class over conf 0.95",
                    (
                        labels_all[(probs_all.max(1)[0] >= 0.95)]
                        >= self.backbone.class_num
                    ).sum(),
                    global_step=current_epoch,
                )

            if select_all.sum() > 0:
                self.writer.add_scalar(
                    "Seen-class ratio under ood score 0.5",
                    (labels_all[select_all] < self.backbone.class_num).sum()
                    / select_all.sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class ratio under ood score 0.5",
                    (labels_all[select_all] >= self.backbone.class_num).sum()
                    / select_all.sum(),
                    global_step=current_epoch,
                )

                self.writer.add_scalar(
                    "Seen-class under ood score 0.5",
                    (labels_all[select_all] < self.backbone.class_num).sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class under ood score 0.5",
                    (labels_all[select_all] >= self.backbone.class_num).sum(),
                    global_step=current_epoch,
                )

            if ((select_all) & (probs_all.max(1)[0] >= 0.95)).sum() > 0:
                self.writer.add_scalar(
                    "Seen-class ratio both under ood score 0.5 and over conf 0.95",
                    (
                        labels_all[((select_all) & (probs_all.max(1)[0] >= 0.95))]
                        < self.backbone.class_num
                    ).sum()
                    / ((select_all) & (probs_all.max(1)[0] >= 0.95)).sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class ratio both under ood score 0.5 and over conf 0.95",
                    (
                        labels_all[((select_all) & (probs_all.max(1)[0] >= 0.95))]
                        >= self.backbone.class_num
                    ).sum()
                    / ((select_all) & (probs_all.max(1)[0] >= 0.95)).sum(),
                    global_step=current_epoch,
                )

                self.writer.add_scalar(
                    "Seen-class both under ood score 0.5 and over conf 0.95",
                    (
                        labels_all[((select_all) & (probs_all.max(1)[0] >= 0.95))]
                        < self.backbone.class_num
                    ).sum(),
                    global_step=current_epoch,
                )
                self.writer.add_scalar(
                    "Unseen-class both under ood score 0.5 and over conf 0.95",
                    (
                        labels_all[((select_all) & (probs_all.max(1)[0] >= 0.95))]
                        >= self.backbone.class_num
                    ).sum(),
                    global_step=current_epoch,
                )


def mixmatch_loss(outputs_x, targets_x, outputs_u, targets_u):

    probs_u = torch.softmax(outputs_u, dim=1)
    Lx = -torch.mean(
        torch.sum(nn.functional.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
    )
    Lu = torch.mean(torch.mean((probs_u - targets_u) ** 2, dim=1))

    return Lx, Lu


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, num_samples=None):

        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integeral "
                "value, but got num_samples={}".format(num_samples)
            )

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.total_size = num_samples
        assert num_samples % self.num_replicas == 0, (
            f"{num_samples} samples cant"
            f"be evenly distributed among {num_replicas} devices."
        )
        self.num_samples = int(num_samples // self.num_replicas)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        n = len(self.dataset)
        n_repeats = self.total_size // n
        n_remain = self.total_size % n
        indices = [torch.randperm(n, generator=g) for _ in range(n_repeats)]
        indices.append(torch.randperm(n, generator=g)[:n_remain])
        indices = torch.cat(indices, dim=0).tolist()

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
