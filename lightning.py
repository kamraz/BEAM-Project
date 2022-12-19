from models import get_model
import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.constants import OUTPUT_REDUCED_MAP


class LitCNN(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        scheduler_name,
        scheduler_hparams,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(
            model_hparams["num_classes"], frozen=model_hparams["frozen"]
        )

        self.validation_map = MeanAveragePrecision(
            num_classes=self.hparams.model_hparams["num_classes"]
        )

        self.test_map = MeanAveragePrecision(
            num_classes=self.hparams.model_hparams["num_classes"],
            class_metrics=True,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "SGD":
            optimizer = optimizer = torch.optim.SGD(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        elif self.hparams.optimizer_name == "ADAM":
            optimizer = torch.optim.Adam(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams.optimizer_name} not implemented"
            )

        if self.hparams.scheduler_name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **self.hparams.scheduler_hparams
            )
        elif self.hparams.scheduler_name == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, **self.hparams.scheduler_hparams
            )
        else:
            raise NotImplementedError(
                f"Scheduler {self.hparams.scheduler_name} not implemented"
            )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log(
            "train_loss", losses, on_step=False, on_epoch=True, batch_size=len(images)
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        results = self.model(images, targets)
        self.validation_map.update(results, targets)

    def validation_epoch_end(self, outputs):
        metrics = self.validation_map.compute()
        self.log("val_map", metrics["map"], on_step=False, on_epoch=True)
        self.log("val_ar", metrics["mar_10"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        results = self.model(images, targets)
        self.test_map.update(results, targets)

    def test_epoch_end(self, outputs):
        metrics = self.test_map.compute()
        self.log("test_map", metrics["map"], on_step=False, on_epoch=True)
        self.log("test_ar", metrics["mar_10"], on_step=False, on_epoch=True)

        # Log mAP for each class
        for key, v in OUTPUT_REDUCED_MAP.items():
            title = "test_map_" + v
            self.log(
                title, metrics["map_per_class"][key - 1], on_step=False, on_epoch=True
            )
