from models import get_model
import torch
import pytorch_lightning as pl


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
        self.model = get_model(model_hparams["num_classes"])

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "SGD":
            optimizer = optimizer = torch.optim.SGD(
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
        else:
            raise NotImplementedError(
                f"Scheduler {self.hparams.scheduler_name} not implemented"
            )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log("train_loss", losses, on_step=False, on_epoch=True)
        return losses

    # def validation_step(self, batch, batch_idx):
    #     images, targets = batch
    #     loss_dict = self.model(images, targets)
    #     losses = sum(loss for loss in loss_dict.values())

    #     self.log("val_loss", losses, on_step=False, on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     images, targets = batch
    #     loss_dict = self.model(images, targets)
    #     losses = sum(loss for loss in loss_dict.values())

    #     self.log("test_loss", losses, on_step=False, on_epoch=True)
