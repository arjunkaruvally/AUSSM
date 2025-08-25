import os
import sys

import torch
import lightning as L
from einops import rearrange
import numpy as np
from numpy.core.numeric import infty
from torchmetrics.functional import accuracy, perplexity
from torchmetrics import Perplexity
import torch.functional as F
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from wavesAI.utils.loss import CEWithChunkedOutputLoss

from icecream import ic
import inspect
from datetime import datetime

ic.configureOutput(includeContext=True)


# ModelCallable = Callable[[Iterable], L.LightningModule]


class SequenceRegression(L.LightningModule):
    def __init__(self, network: torch.nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: LRSchedulerCallable = None,
                 scheduler_interval: str = 'epoch',
                 scheduler_frequency: int = 1,
                 scheduler_monitor: str = 'train_loss',
                 ignore_index: int = -100):
        """ Pytorch Lightning Task that trains sequence models that convert an input sequence to a target terminating/non-terminating sequence. The
        DataLoader that is used with the class needs to have the following properties:
        (1) A single batch should return (x, y, { `masks`: mask }) where x, y are tokenized and converted to Longs

        :param network (torch.nn.Module): network to train. The network must have the following properties:
                                          (1) forward/call function should take optional kwargs
                                          (2) must have a class member `output_vocab_size` which stores the size of the output vocabulary
                                          (3) the output of the model should have size (B, L, C) where B is the batch size and L is the sequence length and C is the output vocabulary size
                                          (4) (optional) an initialize function that is called before forward. e.g. when hidden states needs to be initialized in a Vanilla RNN
        :param optimizer (torch.optim.Optimizer): optimizer to use
        :param scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler to use
        :param scheduler_interval (str, optional): Part of additional arguments sent to the scheduler. Defaults to 'epoch'.
        :param scheduler_frequency (int): Part of additional arguments sent to the scheduler. frequency of scheduler to use
        :param scheduler_monitor (str): Part of additional arguments sent to the scheduler. monitor to use
        """
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.scheduler_monitor = scheduler_monitor
        self.best_val_accuracy = 0.0
        self.ignore_index = ignore_index

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
        self.ppls = []
        self.best_val_accuracy = 0.0
        self.best_ppl = None
        # self.model.log(self.logger.experiment)

        self.ppl_metric_valid = Perplexity(ignore_index=self.ignore_index)
        ce_loss = CEWithChunkedOutputLoss(ignore_index=self.ignore_index)
        self.ce_loss = torch.compile(ce_loss.compute_cross_entropy)

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

        # print(f"in training step: {batch_idx}")

        y_length = y.shape[1]

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}

        if hasattr(self.model, "initialize"):
            self.model.initialize(batch_size=x.shape[0], device=x.device)

        y_pred = self.model(x, **model_kwargs)

        ## error signal is only for the parts of y that are available (0 indicates no data available)
        # signal_mask = torch.abs(y) > 0
        mask = model_kwargs.get("masks", None)

        if mask is not None:
            y_pred = y_pred[mask]
            y = y[mask]
        else:
            # input is longer, and the output will start from
            # the middle of the input
            y_pred = y_pred[:, -y_length:, :]

        ## DEBUG CODE
        # self.trainer.datamodule.timeline(x[0].cpu().tolist()[:],
        #                                  y[0].cpu().tolist()[:],
        #                                  mask[0].cpu().tolist()[:])
        #
        # sys.exit()
        # print("pred ", torch.argmax(y_pred[mask], dim=1))
        # print("actual ", y[mask])
        ############

        loss = torch.mean((y_pred - y)**2)
        # loss = torch.nn.functional.cross_entropy(y_pred, y)

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("train_loss", loss, prog_bar=True)
        if self.scheduler is not None:
            self.log("current_lr", self.scheduler.get_last_lr()[-1], prog_bar=True)

        return loss  # Return tensor to call ".backward" on


    def validation_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]

        x = x.to(self.device)
        y = y.to(self.device)

        # print(f"in training step: {batch_idx}")

        y_length = y.shape[1]

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}

        if hasattr(self.model, "initialize"):
            # print("xshape ", x.shape, x.shape[0])
            self.model.initialize(batch_size=x.shape[0], device=x.device)

        y_pred = self.model(x, **model_kwargs)

        mask = model_kwargs.get("masks", None)
        if mask is not None:
            y_pred = y_pred[mask]
            y = y[mask]
        else:
            y_pred = y_pred[:, -y_length:, :]

        loss = torch.mean((y_pred - y)**2)
        mae = torch.mean(torch.abs(y_pred - y))

        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.log("validation_mae", mae, prog_bar=True, on_epoch=True)
        # self.ppl_metric_valid.reset()

    def test_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]

        x = x.to(self.device)
        y = y.to(self.device)

        y_length = y.shape[1]
        # print(f"in training step: {batch_idx}")

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}

        if hasattr(self.model, "initialize"):
            self.model.initialize(batch_size=x.shape[0], device=x.device)

        y_pred = self.model(x, **model_kwargs)

        ## error signal is only for the parts of y that are available (0 indicates no data available)
        # signal_mask = torch.abs(y) > 0
        # mask = model_kwargs["masks"]
        y_pred = y_pred[:, -y_length:, :]

        loss = torch.mean((y_pred - y)**2)
        mae = torch.mean(torch.abs(y_pred - y))

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_mae", mae, prog_bar=True, on_epoch=True)
