"""
chexFusion lightning module. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/model/cxr_model.py
"""

from typing import Any

import torch
from torch import Tensor
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelRecall,
    MultilabelSpecificity,
    MultilabelAccuracy,
    MultilabelConfusionMatrix,
)
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup

from src.models.layers import Backbone, FusionBackbone
from src.losses import get_loss


class CxrModel(LightningModule):
    """CNN backbone for stage 1"""

    def __init__(
        self,
        lr: float,
        classes: list[str],
        loss_init_args: dict[str, Any],
        model_type: str,
        model_init_args: dict[str, Any],
        embedding: str | None = None,
        zsl: int = 0,
        skip_predict_metrics: bool = True,
        conf_matrix_path: str | None = None,
    ):
        """
        Args:
            lr (float): learning rate
            classes (list[str]): list of class names
            loss_init_args (dict): arguments for the loss function
            model_init_args (dict): arguments for the model model initialization
            skip_predict_metrics (bool): whether to skip metrics during prediction
            conf_matrix_path (str | None): path to save the confusion matrix plot
        """
        super().__init__()
        self.lr = lr
        self.classes = classes
        self.num_classes = len(classes)
        self.backbone = Backbone(
            model_type=model_type,
            model_init_args=model_init_args,
            classes=classes,
            embedding=embedding,
            zsl=zsl,
        )
        self.skip_predict_metrics = skip_predict_metrics
        self.conf_matrix_path = conf_matrix_path

        self.validation_step_outputs = []
        self.prediction_step_outputs = []

        self.val_ap = MultilabelAveragePrecision(
            num_labels=self.num_classes,
            average=None,
        )
        self.val_auc = MultilabelAUROC(num_labels=self.num_classes, average="macro")
        self.f1_score = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.val_acc = MultilabelAccuracy(num_labels=self.num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_labels=self.num_classes, average="macro")
        self.val_specificity = MultilabelSpecificity(
            num_labels=self.num_classes,
            average="macro",
        )
        self.val_conf_mat = MultilabelConfusionMatrix(
            num_labels=self.num_classes,
            threshold=0.5,
        )

        self.criterion_cls = get_loss(**loss_init_args)
        self.loss_type = loss_init_args["type"]
        self.model_type = model_type
        self.zsl = zsl

        self.save_hyperparameters()

    def forward(self, image):
        return self.backbone(image)

    def accumulate_metrics(self, step_outputs: list):
        """
        Accumulate metrics from the step outputs
        """
        preds = torch.cat([x["pred"] for x in step_outputs])
        labels = torch.cat([x["label"] for x in step_outputs])

        aps = self.val_ap(preds, labels.long()).cpu().numpy()
        aucroc = self.val_auc(preds, labels.long())
        f1 = self.f1_score(preds, labels.long())
        acc = self.val_acc(preds, labels.long())
        recall = self.val_recall(preds, labels.long())
        specificity = self.val_specificity(preds, labels.long())
        balanced_acc = (recall + specificity) / 2

        classes = np.array(self.classes)
        for i, c in enumerate(classes):
            print(f"{c} ap: {aps[i]:.4f}")

        if self.num_classes == 40:
            head_idxs = [9, 24, 7, 29, 1, 25, 4, 21, 37]
            medium_idxs = [11, 13, 27, 38, 0, 8, 14, 3, 22, 23, 34, 17, 12, 31, 6]
            tail_idxs = [19, 5, 35, 2, 30, 26, 15, 28, 16, 18, 33, 10, 32, 36, 39, 20]
        elif self.num_classes == 26:
            head_idxs = [6, 14, 4, 20, 0, 16, 2, 12, 24]
            medium_idxs = [25, 18, 5, 9, 1, 13, 15, 10, 8, 22, 3]
            tail_idxs = [21, 17, 19, 7, 23, 11]
        else:
            raise ValueError(
                f"Unexpected number of classes: {self.num_classes}. Expected 40 or 26."
            )

        ap = aps.sum() / self.num_classes

        head_ap = aps[head_idxs].sum() / len(head_idxs)
        medium_ap = aps[medium_idxs].sum() / len(medium_idxs)
        tail_ap = aps[tail_idxs].sum() / len(tail_idxs)

        return {
            "ap": ap,
            "acc": acc,
            "balanced_acc": balanced_acc,
            "auroc": aucroc,
            "f1": f1,
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
            "val_acc",
            metrics["acc"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_balanced_acc",
            metrics["balanced_acc"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            metrics["f1"],
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

        if not self.skip_predict_metrics:
            self.prediction_step_outputs.append(res)

        return preds

    def on_predict_epoch_end(self):
        if self.skip_predict_metrics:
            return

        metrics = self.accumulate_metrics(self.prediction_step_outputs)
        conf_mat = self.val_conf_mat(
            torch.cat([x["pred"] for x in self.prediction_step_outputs]),
            torch.cat([x["label"] for x in self.prediction_step_outputs]).long(),
        )

        try:
            fig, ax = self.val_conf_mat.plot(conf_mat, labels=self.classes)
            fig.set_size_inches(30, 30)
            fig.tight_layout()
            fig.savefig(self.conf_matrix_path)
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")

        print(f"\nPrediction mAP: {metrics['ap']}")
        print(f"Prediction mAUROC: {metrics['auroc']}")
        print(f"Prediction mAccuracy: {metrics['acc']}")
        print(f"Prediction Balanced Accuracy: {metrics['balanced_acc']}")
        print(f"Prediction mF1: {metrics['f1']}")
        print(f"Prediction Head mAP: {metrics['head_ap']}")
        print(f"Prediction Medium mAP: {metrics['medium_ap']}")
        print(f"Prediction Tail mAP: {metrics['tail_ap']}\n")

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
        model_type: str,
        model_init_args: dict[str, Any],
        embedding: str | None = None,
        zsl: int = 0,
        skip_predict_metrics: bool = True,
        conf_matrix_path: str | None = None,
        pretrained_path: str | None = None,
    ):
        super().__init__(
            lr,
            classes,
            loss_init_args,
            model_type,
            model_init_args,
            embedding,
            zsl,
            skip_predict_metrics,
            conf_matrix_path,
        )
        print(f"Using pretrained backbone: {pretrained_path}")
        self.backbone = FusionBackbone(
            model_type=model_type,
            model_init_args=model_init_args,
            classes=classes,
            embedding=embedding,
            zsl=zsl,
            pretrained_path=pretrained_path,
        )


class CxrModelWithEmbeddings(CxrModel):
    """Backbone with embeddings"""

    def __init__(
        self,
        lr: float,
        classes: list[str],
        loss_init_args: dict[str, Any],
        model_type: str,
        model_init_args: dict[str, Any],
        zsl: int = 0,
        skip_predict_metrics: bool = True,
        pretrained_path: str | None = None,
    ):
        pass


class CxrModelFusionWithEmbeddings(CxrModel):
    """ChexFusion with embeddings"""

    def __init__(
        self,
        lr: float,
        classes: list[str],
        loss_init_args: dict[str, Any],
        model_type: str,
        model_init_args: dict[str, Any],
        zsl: int = 0,
        skip_predict_metrics: bool = True,
        pretrained_path: str | None = None,
    ):
        pass
