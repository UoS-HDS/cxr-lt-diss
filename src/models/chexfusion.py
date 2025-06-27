"""
chexFusion lightning module. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/model/cxr_model.py
"""

from typing import Any, Literal

import torch
from torch import Tensor
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers.optimization import get_cosine_schedule_with_warmup

from src.models.layers import Backbone, FusionBackbone
from src.losses import get_loss


class CxrModel(LightningModule):
    """CNN backbone for stage 1"""

    def __init__(
        self,
        lr: float,
        classes: list[str],
        loss_init_args: dict,
        timm_init_args: dict,
    ):
        """
        Args:
            lr (float): learning rate
            classes (list[str]): list of class names
            loss_init_args (dict): arguments for the loss function
            timm_init_args (dict): arguments for the timm model initialization
        """
        super().__init__()
        self.lr = lr
        self.classes = classes
        self.num_classes = len(classes)
        self.backbone = Backbone(timm_init_args)
        self.validation_step_outputs = []
        self.prediction_step_outputs = []
        self.val_ap = AveragePrecision(task="binary")
        self.val_auc = AUROC(task="binary")

        self.criterion_cls = get_loss(**loss_init_args)

        self.save_hyperparameters()

    def forward(self, image):
        return self.backbone(image)

    def accumulate_metrics(self, step_outputs: list):
        """
        Accumulate metrics from the step outputs and log them.
        """
        preds = torch.cat([x["pred"] for x in step_outputs])
        labels = torch.cat([x["label"] for x in step_outputs])

        aps = []
        auc_rocs = []
        for i in range(self.num_classes):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            aps.append(ap)
            auc_rocs.append(auroc)
            print(f"\n{self.classes[i]}_ap: {ap}")

        head_idxs = [9, 24, 7, 29, 1, 25, 4, 21, 37]
        medium_idxs = [11, 13, 27, 38, 0, 8, 14, 3, 22, 23, 34, 17, 12, 31, 6]
        tail_idxs = [19, 5, 35, 2, 30, 26, 15, 28, 16, 18, 33, 10, 32, 36, 39, 20]

        ap = sum(aps) / self.num_classes
        auc_roc = sum(auc_rocs) / self.num_classes
        head_ap = sum([aps[i] for i in head_idxs]) / len(head_idxs)
        medium_ap = sum([aps[i] for i in medium_idxs]) / len(medium_idxs)
        tail_ap = sum([aps[i] for i in tail_idxs]) / len(tail_idxs)

        return {
            "ap": ap,
            "auroc": auc_roc,
            "head_ap": head_ap,
            "medium_ap": medium_ap,
            "tail_ap": tail_ap,
        }

    def shared_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        image, label = batch
        pred = self(image)

        loss = self.criterion_cls(pred, label)

        pred = torch.sigmoid(pred).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        res = self.shared_step(batch, batch_idx)
        self.log("loss", res["loss"].detach(), prog_bar=True, sync_dist=True)
        self.log(
            "train_loss",
            res["loss"].detach(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return res["loss"]

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        res = self.shared_step(batch, batch_idx)
        self.log("val_loss", res["loss"].detach(), prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        metrics = self.accumulate_metrics(self.validation_step_outputs)
        self.log(
            "val_ap",
            metrics["ap"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_auroc",
            metrics["auroc"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_head_ap",
            metrics["head_ap"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_medium_ap",
            metrics["medium_ap"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_tail_ap",
            metrics["tail_ap"],
            prog_bar=True,
            sync_dist=True,
        )
        self.validation_step_outputs = []

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        res = self.shared_step(batch, batch_idx)
        preds = res["pred"]
        images, labels = batch
        batch_1 = (images.flip(-1), labels)
        preds_1 = self.shared_step(batch_1, batch_idx)["pred"]
        preds = (preds + preds_1) / 2
        res["pred"] = preds
        self.prediction_step_outputs.append(res)

        return preds

    def on_predict_epoch_end(self):
        metrics = self.accumulate_metrics(self.prediction_step_outputs)
        print(f"\nPrediction AP: {metrics['ap']}")
        print(f"Prediction AUROC: {metrics['auroc']}")
        print(f"Prediction Head AP: {metrics['head_ap']}")
        print(f"Prediction Medium AP: {metrics['medium_ap']}")
        print(f"Prediction Tail AP: {metrics['tail_ap']}\n")

        self.prediction_step_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]


class CxrModelFusion(CxrModel):
    """ChexFusion"""

    def __init__(
        self,
        lr: float,
        classes: list[str],
        loss_init_args: dict[str, Any],
        timm_init_args: dict[str, Any],
        pretrained_path: str | None = None,
    ):
        super().__init__(lr, classes, loss_init_args, timm_init_args)
        self.backbone = FusionBackbone(timm_init_args, pretrained_path)
