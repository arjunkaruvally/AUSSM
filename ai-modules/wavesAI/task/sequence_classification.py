import os
import sys

import torch
import torch.nn.functional as F
import lightning as L
from numpy import flatiter
from torchmetrics.functional import accuracy
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable

import wandb
import numpy as np

# ModelCallable = Callable[[Iterable], L.LightningModule]


class SequenceClassification(L.LightningModule):
    def __init__(self, network: torch.nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: LRSchedulerCallable = None,
                 scheduler_interval: str = 'epoch',
                 scheduler_frequency: int = 1,
                 scheduler_monitor: str = 'train_loss',
                 learning_rate_multiplier: float = 1.0):
        """Task that trains regression models that convert an input sequence to a terminating/non-terminating sequence.
        :param network:
        :param optimizer:
        :param scheduler:
        """
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.scheduler_monitor = scheduler_monitor
        self.best_val_accuracy = 0.0

        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.model = network
        self.optimizer = optimizer
        self.scheduler = scheduler

        torch.autograd.set_detect_anomaly(True)  # this helps in debugging error messages

        # self.model.double()  # make sure model is in double precision
        self.model = self.model.to(self.device)

        # logic for computing the best validation accuracy so far
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.val_samples = []
        # self.model.log(self.logger.experiment)

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.parameters())

        optimization_config = {
            "optimizer": self.optimizer,
        }

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)
            optimization_config["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
                "monitor": self.scheduler_monitor
            }
        return optimization_config

    def forward(self, batch):
        # Forward function
        return self.model(batch)

    def on_train_start(self) -> None:
        # self.model.log(self.logger)
        pass

    def load_saved_model(self):
        print("walking")
        for root, dirs, files in os.walk(self.trainer.default_root_dir, topdown=True):
            for file in files:
                print(root, dirs, file)
                if file.endswith(".ckpt"):
                    self.load_from_checkpoint(os.path.join(root, file))
                    return
        raise FileNotFoundError("Saved model not found")

    def on_test_start(self) -> None:
        # load the saved model from here
        # self.load_saved_model()
        pass

    def training_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]

        x = x.to(self.device)
        y = y.to(self.device)

        if len(x.shape) > 2:
            batch_size, seq_length, dim = x.shape
        else:
            batch_size, seq_length = x.shape

        # print(f"in training step: {batch_idx}")

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}
        y_pred = self.model(x, **model_kwargs)

        y_pred = y_pred.squeeze()
        y = y.squeeze()

        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task="multiclass", num_classes=self.model.num_classes)

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)

        if self.scheduler is not None:
            self.log("current_lr", self.scheduler.get_last_lr()[-1], prog_bar=True)

        return loss  # Return tensor to call ".backward" on

    def on_validation_epoch_end(self) -> None:
        self.val_samples = np.array(self.val_samples)
        self.val_accuracies = np.array(self.val_accuracies)
        acc = np.sum(self.val_accuracies * self.val_samples) / np.sum(self.val_samples)
        if acc > self.best_val_accuracy:
            self.best_val_accuracy = acc

        self.log("best_val_accuracy", self.best_val_accuracy, prog_bar=True)
        self.val_accuracies = []  # reset
        self.val_samples = []

    def validation_step(self, batch, batch_idx):
        # if self.trainer.global_step == 0:
        #     self.logger.define_metric("val_accuracy", summary="max")
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]
        x = x.to(self.device)
        y = y.to(self.device)
        if len(x.shape) > 2:
            batch_size, seq_length, dim = x.shape
        else:
            batch_size, seq_length = x.shape

        # print(f"in validation step: {batch_idx}")

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}
        y_pred = self.model(x, **model_kwargs)

        y_pred = y_pred.squeeze()
        y = y.squeeze()

        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task="multiclass", num_classes=self.model.num_classes)

        self.val_accuracies.append(acc.cpu().item())

        self.log("validation_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.val_samples.append(x.shape[0])

    def test_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]
        x = x.to(self.device)
        y = y.to(self.device)
        if len(x.shape) > 2:
            batch_size, seq_length, dim = x.shape
        else:
            batch_size, seq_length = x.shape

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}
        y_pred = self.model(x, **model_kwargs)

        y_pred = y_pred.squeeze()
        y = y.squeeze()

        loss = F.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, task="multiclass", num_classes=self.model.num_classes)

        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
