import os

import numpy as np
import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from tasks.base import Task
from utils.metrics import TopKAccuracy
from sklearn.metrics import f1_score, roc_auc_score


class Testing(Task):
    def __init__(self, backbone: nn.Module):
        super(Testing, self).__init__()

        '''
        calculate : Accuracy about corrupted data,
                    ECE about Clean & corrupted data,
                    NLL about Clean & corrupted data,
                    Brier Score about Clean & corrupted data,
                    OOD AUPR about Clean & corrupted data,
        '''

        self.backbone = backbone
        self.loss_function = None
        self.prepared = False

    def prepare(self,
                ckpt_dir: str,
                model_ckpt_dir: str,
                batch_size: int = 256,
                num_workers: int = 0,
                local_rank: int = 0,
                **kwargs):  # pylint: disable=unused-argument

        """Add function docstring."""
        # Set attributes
        self.ckpt_dir = ckpt_dir                                            # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size                                        # pylint: disable=attribute-defined-outside-init
        self.num_workers = num_workers                                      # pylint: disable=attribute-defined-outside-init
        self.local_rank = 'cpu' if local_rank<0 else local_rank             # pylint: disable=attribute-defined-outside-init

        # load ckpoint
        ckpt = torch.load(os.path.join(model_ckpt_dir, "ckpt.best.pth.tar"))['backbone']
        try:
            self.backbone.load_state_dict(ckpt)
        except:
            for old_key, new_key in zip(["output.weight","output.bias"],["output.linear.weight","output.linear.bias"]):
                ckpt[new_key] = ckpt.pop(old_key)
            self.backbone.load_state_dict(ckpt)
        self.backbone.to(local_rank)

        # Ready
        self.prepared = True

    def run(self,
            for_what,
            open_test_set: torch.utils.data.Dataset = None,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (Evaluation)
        open_test_loader = DataLoader(
            open_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False
        )

        # Logging
        logger = kwargs.get('logger', None)

        # Evaluate (Validation)
        if for_what=="Proposed":
            eval_history = self.proposed_evaluate(open_test_loader)
        elif for_what=="Ablation1":
            eval_history = self.ablation1_evaluate(open_test_loader)
        elif for_what=="Ablation2":
            eval_history = self.ablation2_evaluate(open_test_loader)
        elif for_what=="Ablation3":
            eval_history = self.ablation3_evaluate(open_test_loader)
        elif for_what=="IOMATCH":
            eval_history = self.iomatch_evaluate(open_test_loader)
        elif for_what=="OPENMATCH":
            eval_history = self.openmatch_evaluate(open_test_loader)
        elif for_what in ["FIXMATCH","SL"]:
            eval_history = self.fixmatch_evaluate(open_test_loader)
        elif for_what=="MTC":
            eval_history = self.mtc_evaluate(open_test_loader)
        else:
            pass

        log=""
        for k, v in eval_history.items():
            log += f" {k}: {v:.5f} |"

        if logger is not None:
            logger.info(log)

    @torch.no_grad()
    def mtc_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            "F1": np.zeros(1),
            'In distribution over conf 0.95: ECE': np.zeros(1),
            'In distribution under ood score 0.5: ECE': np.zeros(1)
        }

        labels, logits, domain_score = [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, feat = self.backbone(x, return_feature=True)
                logits_domain = self.backbone.domain_classifier(feat)

                score = logits_domain.sigmoid().squeeze()

                labels.append(y.cpu())
                logits.append(logit.cpu())
                domain_score.append(score.cpu())

                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, domain_score = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(domain_score, axis=0)

        threshold = threshold_otsu(domain_score.cpu().numpy())
    
        in_pred = (domain_score >= threshold).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['In distribution over conf 0.95: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].numpy(), plot_title="_conf_over_95")
        result['In distribution under ood score 0.5: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (in_pred)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (in_pred)].numpy(), plot_title="_ood_score_under_05")

        return {k: v.mean().item() for k, v in result.items()}
    
    @torch.no_grad()
    def iomatch_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            "F1": np.zeros(1)
        }

        labels, logits, out_scores = [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logits_x_ulb_w, feature = self.backbone(x, return_feature=True)
                feat_proj = self.backbone.mlp_proj(feature)
                logits_open = self.backbone.openset_classifier(feat_proj)  # (k+1)-way logits
                logits_mb_x_ulb_w = self.backbone.mb_classifiers(feat_proj)  # shape: [bsz, 2K]

                p = nn.functional.softmax(logits_x_ulb_w, dim=-1)
                targets_p = p.detach()

                logits_mb = logits_mb_x_ulb_w.view(x.size(0), 2, -1)
                r = nn.functional.softmax(logits_mb, 1)
                tmp_range = torch.arange(0, x.size(0)).long().cuda(self.local_rank)
                out_score = torch.sum(targets_p * r[tmp_range, 0, :], 1)

                labels.append(y.cpu())
                logits.append(logits_x_ulb_w.cpu())
                out_scores.append(out_score)
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, out_scores = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(out_scores,axis=0)
        
        in_pred = (out_scores < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)

        return {k: v.mean().item() for k, v in result.items()}
    
    @torch.no_grad()
    def fixmatch_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            "F1": np.zeros(1)
        }

        labels, logits  = [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit = self.predict(x)
                
                labels.append(y.cpu())
                logits.append(logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits = torch.cat(labels,axis=0), torch.cat(logits,axis=0)
        
        in_pred = (logits.softmax(1).max(1)[0] >= 0.95).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)

        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def openmatch_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'ECE-ova': np.zeros(1),
            "F1": np.zeros(1),
            "AUROC": np.zeros(1),
            "SEEN-DETECTION-ECE": np.zeros(1),
        }

        labels, logits, out_scores, in_scores = [], [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, feat = self.backbone(x, return_feature=True)
                logits_open = self.backbone.ova_classifiers(feat)

                logits_open = nn.functional.softmax(logits_open.view(logits_open.size(0), 2, -1), 1)
                tmp_range = torch.arange(0, logits_open.size(0)).long().cuda(self.local_rank)

                pred_close = logit.data.max(1)[1]
                unk_score = logits_open[tmp_range, 0, pred_close]
                sen_score = logits_open[tmp_range, 1, pred_close]

                labels.append(y.cpu())
                logits.append(logit.cpu())
                out_scores.append(unk_score.cpu())
                in_scores.append(sen_score.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, out_scores, in_scores = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(out_scores,axis=0), torch.cat(in_scores,axis=0)
        
        in_pred = (out_scores < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['AUROC'][0] = roc_auc_score(y_true=in_label, y_score=(1-out_scores).cpu())
        result['SEEN-DETECTION-ECE'][0] = self.seen_unseen_detection_get_ece(predicted_label=in_pred.numpy(),confidences=in_scores.numpy(),targets=in_label.numpy(),n_bins=15)

        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def ablation3_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'top@1-ova': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'ECE-ova': np.zeros(1),
            "F1": np.zeros(1),
            'In distribution over conf 0.95: ECE': np.zeros(1),
            'In distribution under ood score 0.5: ECE': np.zeros(1),
            "SEEN-DETECTION-ECE": np.zeros(1),
        }

        labels, logits, ova_scores, ova_logits = [], [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, features = self.backbone(x, return_feature=True)

                probs = nn.functional.softmax(logit, 1)

                ova_logit = (self.backbone.ova_classifiers(features)).view(features.size(0),2,-1)
                ova_score = (ova_logit.softmax(1)*probs.unsqueeze(1)).sum(-1)
                
                labels.append(y.cpu())
                logits.append(logit.cpu())
                ova_scores.append(ova_score.cpu())
                ova_logits.append(ova_logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, ova_scores, ova_logits = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(ova_scores,axis=0), torch.cat(ova_logits,axis=0)
        
        in_pred = (ova_scores[:,0] < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['top@1-ova'][0] = TopKAccuracy(k=1)(ova_logits.softmax(dim=1)[labels<logits.size(1),1,:], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['ECE-ova'][0] = self.get_ece(preds=ova_logits.softmax(dim=1)[labels<logits.size(1),1,:].numpy(), targets = labels[labels<logits.size(1)].numpy(), plot_title="_all_ova")
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['In distribution over conf 0.95: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].numpy(), plot_title="_conf_over_95")
        result['In distribution under ood score 0.5: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (in_pred)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (in_pred)].numpy(), plot_title="_ood_score_under_05")
        result['SEEN-DETECTION-ECE'][0] = self.seen_unseen_detection_get_ece(predicted_label=in_pred.numpy(),confidences=ova_scores.max(1)[0].numpy(),targets=in_label.numpy(),n_bins=15)
        
        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def ablation2_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'top@1-ova': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'ECE-ova': np.zeros(1),
            "F1": np.zeros(1),
            'In distribution over conf 0.95: ECE': np.zeros(1),
            'In distribution under ood score 0.5: ECE': np.zeros(1),
            "SEEN-DETECTION-ECE": np.zeros(1),
        }

        labels, logits, ova_scores, ova_logits = [], [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, features = self.backbone(x, return_feature=True)
                logit = self.backbone.scaling_logits(logit)

                probs = nn.functional.softmax(logit, 1)

                ova_logit = (self.backbone.ova_classifiers(features)).view(features.size(0),2,-1)
                ova_score = (ova_logit.softmax(1)*probs.unsqueeze(1)).sum(-1)
                
                labels.append(y.cpu())
                logits.append(logit.cpu())
                ova_scores.append(ova_score.cpu())
                ova_logits.append(ova_logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, ova_scores, ova_logits = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(ova_scores,axis=0), torch.cat(ova_logits,axis=0)
        
        in_pred = (ova_scores[:,0] < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['top@1-ova'][0] = TopKAccuracy(k=1)(ova_logits.softmax(dim=1)[labels<logits.size(1),1,:], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['ECE-ova'][0] = self.get_ece(preds=ova_logits.softmax(dim=1)[labels<logits.size(1),1,:].numpy(), targets = labels[labels<logits.size(1)].numpy(), plot_title="_all_ova")
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['In distribution over conf 0.95: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].numpy(), plot_title="_conf_over_95")
        result['In distribution under ood score 0.5: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (in_pred)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (in_pred)].numpy(), plot_title="_ood_score_under_05")
        result['SEEN-DETECTION-ECE'][0] = self.seen_unseen_detection_get_ece(predicted_label=in_pred.numpy(),confidences=ova_scores.max(1)[0].numpy(),targets=in_label.numpy(),n_bins=15)
        
        return {k: v.mean().item() for k, v in result.items()}

    @torch.no_grad()
    def ablation1_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'top@1-ova': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'ECE-ova': np.zeros(1),
            "F1": np.zeros(1),
            'In distribution over conf 0.95: ECE': np.zeros(1),
            'In distribution under ood score 0.5: ECE': np.zeros(1),
            "SEEN-DETECTION-ECE": np.zeros(1),
        }

        labels, logits, ova_scores, ova_logits = [], [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, features = self.backbone(x, return_feature=True)
                probs = nn.functional.softmax(logit, 1)

                ova_logit = self.backbone.scaling_logits(self.backbone.ova_classifiers(features),name='ova_cali_scaler').view(features.size(0),2,-1)
                ova_score = (ova_logit.softmax(1)*probs.unsqueeze(1)).sum(-1)
                
                labels.append(y.cpu())
                logits.append(logit.cpu())
                ova_scores.append(ova_score.cpu())
                ova_logits.append(ova_logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, ova_scores, ova_logits = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(ova_scores,axis=0), torch.cat(ova_logits,axis=0)
        
        in_pred = (ova_scores[:,0] < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['top@1-ova'][0] = TopKAccuracy(k=1)(ova_logits.softmax(dim=1)[labels<logits.size(1),1,:], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['ECE-ova'][0] = self.get_ece(preds=ova_logits.softmax(dim=1)[labels<logits.size(1),1,:].numpy(), targets = labels[labels<logits.size(1)].numpy(), plot_title="_all_ova")
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['In distribution over conf 0.95: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].numpy(), plot_title="_conf_over_95")
        result['In distribution under ood score 0.5: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (in_pred)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (in_pred)].numpy(), plot_title="_ood_score_under_05")
        result['SEEN-DETECTION-ECE'][0] = self.seen_unseen_detection_get_ece(predicted_label=in_pred.numpy(),confidences=ova_scores.max(1)[0].numpy(),targets=in_label.numpy(),n_bins=15)
        
        return {k: v.mean().item() for k, v in result.items()}
        
    @torch.no_grad()
    def proposed_evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'top@1': torch.zeros(1, device=self.local_rank),
            'top@1-ova': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'ECE-ova': np.zeros(1),
            "F1": np.zeros(1),
            'In distribution over conf 0.95: ECE': np.zeros(1),
            'In distribution under ood score 0.5: ECE': np.zeros(1),
            "SEEN-DETECTION-ECE": np.zeros(1),
        }

        labels, logits, ova_scores, ova_logits = [], [], [], []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)

                logit, features = self.backbone(x, return_feature=True)
                logit = self.backbone.scaling_logits(logit)
                probs = nn.functional.softmax(logit, 1)

                ova_logit = self.backbone.scaling_logits(self.backbone.ova_classifiers(features),name='ova_cali_scaler').view(features.size(0),2,-1)
                ova_score = (ova_logit.softmax(1)*probs.unsqueeze(1)).sum(-1)
                
                labels.append(y.cpu())
                logits.append(logit.cpu())
                ova_scores.append(ova_score.cpu())
                ova_logits.append(ova_logit.cpu())
                
                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        labels, logits, ova_scores, ova_logits = torch.cat(labels,axis=0), torch.cat(logits,axis=0), torch.cat(ova_scores,axis=0), torch.cat(ova_logits,axis=0)
        
        in_pred = (ova_scores[:,0] < 0.5).cpu()
        in_label = torch.where(labels<logits.size(1),1,0)

        result['top@1'][0] = TopKAccuracy(k=1)(logits[labels<logits.size(1)], labels[labels<logits.size(1)])
        result['top@1-ova'][0] = TopKAccuracy(k=1)(ova_logits.softmax(dim=1)[labels<logits.size(1),1,:], labels[labels<logits.size(1)])
        result['ECE'][0] = self.get_ece(preds=logits[labels<logits.size(1)].softmax(dim=1).numpy(), targets = labels[labels<logits.size(1)].numpy())
        result['ECE-ova'][0] = self.get_ece(preds=ova_logits.softmax(dim=1)[labels<logits.size(1),1,:].numpy(), targets = labels[labels<logits.size(1)].numpy(), plot_title="_all_ova")
        result['F1'][0] = f1_score(y_true=in_label, y_pred=in_pred)
        result['In distribution over conf 0.95: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (logits.softmax(1).max(1)[0]>=0.95)].numpy(), plot_title="_conf_over_95")
        result['In distribution under ood score 0.5: ECE'][0] = self.get_ece(preds=logits[(labels<logits.size(1)) & (in_pred)].softmax(dim=1).numpy(), targets = labels[(labels<logits.size(1)) & (in_pred)].numpy(), plot_title="_ood_score_under_05")
        result['SEEN-DETECTION-ECE'][0] = self.seen_unseen_detection_get_ece(predicted_label=in_pred.numpy(),confidences=ova_scores.max(1)[0].numpy(),targets=in_label.numpy(),n_bins=15)
        
        return {k: v.mean().item() for k, v in result.items()}

    def predict(self, x: torch.FloatTensor):
        """Make a prediction provided a batch of samples."""
        return self.backbone(x)

    def _set_learning_phase(self, train=False):
        if train:
            self.backbone.train()
        else:
            self.backbone.eval()

    def get_ece(self, preds: np.array, targets: np.array, n_bins: int=15, **kwargs):

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

        plot_title = kwargs.get("plot_title", "_all")

        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None], np.array(confs_ticks)[np.array(x_ticks)!=None], label = 'Confidence', alpha = .7, ecolor = 'blue', width = bin_lowers[1] , edgecolor='black')
        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None] , np.array(acc_ticks)[np.array(x_ticks)!=None], label = 'Accuracy', alpha = .7, ecolor = 'pink', width = bin_lowers[1] , edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks(bin_boundaries,bin_boundaries.round(2))
        plt.legend()
        plt.savefig(os.path.join(self.ckpt_dir,f'Reliability_diagrams{plot_title}.png'))
        plt.close('all')

        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None], np.array(y_ticks_second_ticks)[np.array(x_ticks)!=None], ecolor = 'blue', width = bin_lowers[1] , edgecolor='black')
        plt.xticks(bin_boundaries,bin_boundaries.round(2))
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0])
        plt.axvline(x=accuracies.mean() , label='Accuracy', color='red', linestyle="--")
        plt.axvline(x=confidences.mean() , label='Avg. confidence', color='blue', linestyle="--")
        plt.legend()
        plt.xlabel('Confidence')
        plt.ylabel('% of Samples')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig(os.path.join(self.ckpt_dir,f'Confidence_histogram{plot_title}.png'))
        plt.close('all')

        return ece
    
    def seen_unseen_detection_get_ece(self, predicted_label: np.array, confidences: np.array, targets: np.array, n_bins: int=15):

        bin_boundaries = np.linspace(0,1,n_bins+1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = (predicted_label == targets)

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

        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None], np.array(confs_ticks)[np.array(x_ticks)!=None], label = 'Confidence', alpha = .7, ecolor = 'blue', width = bin_lowers[1] , edgecolor='black')
        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None] , np.array(acc_ticks)[np.array(x_ticks)!=None], label = 'Accuracy', alpha = .7, ecolor = 'pink', width = bin_lowers[1] , edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks(bin_boundaries,bin_boundaries.round(2))
        plt.legend()
        plt.savefig(os.path.join(self.ckpt_dir,'SEEN-DETECTION-ECE.png'))
        plt.close('all')

        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None], np.array(y_ticks_second_ticks)[np.array(x_ticks)!=None], ecolor = 'blue', width = bin_lowers[1] , edgecolor='black')
        plt.xticks(bin_boundaries,bin_boundaries.round(2))
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0])
        plt.axvline(x=accuracies.mean() , label='Accuracy', color='red', linestyle="--")
        plt.axvline(x=confidences.mean() , label='Avg. confidence', color='blue', linestyle="--")
        plt.legend()
        plt.xlabel('Confidence')
        plt.ylabel('% of Samples')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig(os.path.join(self.ckpt_dir,'CONFIDENCE_HISTOGRAM.png'))
        plt.close('all')
        
        return ece