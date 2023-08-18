import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import torch.nn as nn
from rich.progress import Progress
from torch.utils.data import DataLoader

from tasks.base import Task
from utils.metrics import TopKAccuracy


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
        ckpt = torch.load(os.path.join(self.ckpt_dir, "ckpt.best.pth.tar"))['backbone']
        try:
            self.backbone.load_state_dict(ckpt)
        except:
            ckpt = OrderedDict((key.replace("module.",""), value) for key, value in ckpt.items())
            self.backbone.load_state_dict(ckpt)

        self.backbone.to(local_rank)

        # metric function (negative log likelihood)
        self.nll = nn.CrossEntropyLoss()

        # Ready
        self.prepared = True

    def run(self,
            eval_set: torch.utils.data.Dataset = None,
            test_set: torch.utils.data.Dataset = None,
            **kwargs):  # pylint: disable=unused-argument

        batch_size = self.batch_size
        num_workers = self.num_workers

        if not self.prepared:
            raise RuntimeError("Training not prepared.")

        # DataLoader (Evaluation)
        eval_loader = DataLoader(
            eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False
        )

        # DataLoader (Clean test, Corrupted test)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False
        )

        # Logging
        logger = kwargs.get('logger', None)

        # Evaluate (Validation)
        eval_history = self.evaluate(eval_loader, plot_title="evaluation")
        log=""
        for k, v in eval_history.items():
            log += f" Eval_{k}: {v:.3f} |"

        if logger is not None:
            logger.info(log)

        # Evaluate (Testing)
        test_history = self.evaluate(test_loader, plot_title="test:original")

        log=""
        for k, v in test_history.items():
            log += f" Clean_{k}: {v:.3f} |"

        if logger is not None:
            logger.info(log)

    @torch.no_grad()
    def evaluate(self, data_loader, **kwargs):
        """Evaluation defined for a single epoch."""

        steps = len(data_loader)
        self._set_learning_phase(train=False)
        result = {
            'NLL': np.zeros(1),
            'top@1': torch.zeros(1, device=self.local_rank),
            'ECE': np.zeros(1),
            'Brier_scores': torch.zeros(1, device=self.local_rank)
        }

        pred,true=[],[]
        IDX = []
        UNCERTAINTY = []

        with Progress(transient=True, auto_refresh=False) as pg:

            if self.local_rank == 0:
                task = pg.add_task(f"[bold red] {kwargs['plot_title']} Data Testing...", total=steps)

            for i, batch in enumerate(data_loader):

                x = batch['x'].to(self.local_rank)
                y = batch['y'].to(self.local_rank)
                idx = batch['idx'].to(self.local_rank)

                logits = self.predict(x)

                true.append(y.cpu())
                pred.append(logits.cpu())

                uncertainty = 1- logits.softmax(dim=1).max(dim=1)[0]

                IDX += [idx]
                UNCERTAINTY += [uncertainty]

                if self.local_rank == 0:
                    desc = f"[bold pink] [{i+1}/{steps}] |"
                    pg.update(task, advance=1., description=desc)
                    pg.refresh()

        preds, trues = torch.cat(pred,axis=0), torch.cat(true,axis=0) # preds, pred are logit vectors
        one_hot_targets = torch.zeros(trues.size(0), logits.size(1)).scatter_(1,trues.view(-1,1).long(),1)
        
        IDX = torch.cat(IDX, dim=0)
        UNCERTAINTY = torch.cat(UNCERTAINTY, dim=0)

        result['top@1'][0] = TopKAccuracy(k=1)(preds, trues)
        result['NLL'][0] = self.nll(input=preds, target = trues.long())
        result['ECE'][0] = self.get_ece(preds=preds.softmax(dim=1).numpy(), targets = trues.numpy(), plot_title=kwargs["plot_title"])
        result['Brier_scores'][0] = torch.square(torch.subtract(one_hot_targets,preds.softmax(dim=1))).mean()

        RESULTS = pandas.DataFrame({"LABEL":trues.cpu().numpy(),"PRED_LABEL":preds.argmax(dim=1).cpu().numpy(), "IDX":IDX.cpu().numpy()})
        RESULTS = pandas.concat([RESULTS, pandas.DataFrame(preds.cpu().numpy(), columns = [f"logit_{i}" for i in range(logits.size(1))])], axis=1)
        RESULTS.to_csv(os.path.join(self.ckpt_dir,f"{data_loader.dataset.data_name.upper()}_{kwargs['plot_title']}_RESULT.csv"),index=False)

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


        plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None], np.array(confs_ticks)[np.array(x_ticks)!=None], label = 'Confidence', alpha = .7, ecolor = 'blue', width = bin_lowers[1] , edgecolor='black')
        if "OOD" not in kwargs["plot_title"]:
            plt.bar(np.array(x_ticks)[np.array(x_ticks)!=None] , np.array(acc_ticks)[np.array(x_ticks)!=None], label = 'Accuracy', alpha = .7, ecolor = 'pink', width = bin_lowers[1] , edgecolor='black')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')

        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks(bin_boundaries,bin_boundaries.round(2))
        plt.legend()
        plt.savefig(os.path.join(self.ckpt_dir,f'ECE_{kwargs["plot_title"]}.png'))
        plt.close('all')

        if "OOD" not in kwargs["plot_title"]:
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
            plt.savefig(os.path.join(self.ckpt_dir,f'CONFIDENCE_HISTOGRAM_{kwargs["plot_title"]}.png'))
            plt.close('all')

        return ece