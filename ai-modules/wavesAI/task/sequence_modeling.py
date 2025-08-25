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


class SequenceModeling(L.LightningModule):
    def __init__(self, network: torch.nn.Module,
                 optimizer: OptimizerCallable,
                 scheduler: LRSchedulerCallable,
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
        if getattr(self, 'get_parameter_groups_for_optimizer', None) is None:
            self.optimizer = self.optimizer(self.parameters())
        else:
            self.optimizer = self.optimizer(self.model.get_parameter_groups_for_optimizer())

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        self.scheduler = self.scheduler(self.optimizer)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                     "scheduler": self.scheduler,
                     "interval": self.scheduler_interval,
                     "frequency": self.scheduler_frequency,
                     "monitor": self.scheduler_monitor
                 }
        }

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
        batch_size, seq_length = x.shape

        # print(f"in training step: {batch_idx}")

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
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)

        ## DEBUG CODE
        # self.trainer.datamodule.timeline(x[0].cpu().tolist()[:],
        #                                  y[0].cpu().tolist()[:],
        #                                  mask[0].cpu().tolist()[:])
        #
        # sys.exit()
        # print("pred ", torch.argmax(y_pred[mask], dim=1))
        # print("actual ", y[mask])
        ############

        loss = self.ce_loss(y_pred, y)
        # loss = torch.nn.functional.cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y, num_classes=self.model.output_vocab_size, task="multiclass")

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", acc, prog_bar=True)
        self.log("current_lr", self.scheduler.get_last_lr()[-1], prog_bar=True)

        return loss  # Return tensor to call ".backward" on

    def on_validation_epoch_end(self) -> None:
        acc = np.mean(self.val_accuracies)
        if acc > self.best_val_accuracy:
            self.best_val_accuracy = acc

        if self.ignore_index != -100:
            ppl = np.mean(self.ppls)
            if self.best_ppl is None or ppl < self.best_ppl:
                self.best_ppl = ppl
                self.log("best_ppl", self.best_ppl, prog_bar=True)
                self.ppls = []

        self.log("best_val_accuracy", self.best_val_accuracy, prog_bar=True)
        self.val_accuracies = []  # reset


    def validation_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]

        x = x.to(self.device)
        y = y.to(self.device)
        batch_size, seq_length = x.shape

        # print(f"in training step: {batch_idx}")

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}

        if hasattr(self.model, "initialize"):
            # print("xshape ", x.shape, x.shape[0])
            self.model.initialize(batch_size=x.shape[0], device=x.device)

        y_pred = self.model(x, **model_kwargs)

        ## error signal is only for the parts of y that are available (0 indicates no data available)
        # signal_mask = torch.abs(y) > 0

        if self.ignore_index != -100:
            val_ppl = perplexity(y_pred, y, ignore_index=self.ignore_index).item()
            self.log("val_ppl", val_ppl, prog_bar=True)
            self.ppls.append(val_ppl)
        # ic(y_pred.shape)
        # ic(y.shape)
        # self.ppl_metric_valid.update(y_pred, y)
        # val_ppl = self.ppl_metric_valid.compute()

        mask = model_kwargs.get("masks", None)
        if mask is not None:
            y_pred = y_pred[mask]
            y = y[mask]
        else:
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)

        loss = torch.nn.functional.cross_entropy(y_pred, y,
                                                 ignore_index=self.ignore_index)
        acc = accuracy(y_pred, y,
                       num_classes=self.model.output_vocab_size, task="multiclass"
                       )


        self.val_accuracies.append(acc.cpu().item())

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("validation_loss", loss, prog_bar=True)
        self.log("validation_accuracy", acc, prog_bar=True)
        # self.ppl_metric_valid.reset()

    def test_step(self, batch, batch_idx):
        batch_out = batch
        x = batch_out[0]
        y = batch_out[1]

        x = x.to(self.device)
        y = y.to(self.device)
        batch_size, seq_length = x.shape

        # print(f"in training step: {batch_idx}")

        model_kwargs = batch_out[2] if len(batch_out) > 2 else {}

        if hasattr(self.model, "initialize"):
            self.model.initialize(batch_size=x.shape[0], device=x.device)

        y_pred = self.model(x, **model_kwargs)

        ## error signal is only for the parts of y that are available (0 indicates no data available)
        # signal_mask = torch.abs(y) > 0
        mask = model_kwargs["masks"]
        loss = torch.nn.functional.cross_entropy(y_pred[mask], y[mask])
        acc = accuracy(y_pred[mask], y[mask], num_classes=self.model.output_vocab_size, task="multiclass")

        # # Logs the loss per epoch to wandb (weighted average over batches)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", acc, prog_bar=True)
