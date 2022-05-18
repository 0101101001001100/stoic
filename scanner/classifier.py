# define the pytorch lightning module for STOIC classifier training

from typing import Callable, List
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics.functional as tmf
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from .utils import weighted_cross_entropy, weighted_bce_logit


class STOICNet(LightningModule):
    def __init__(self, 
        encoder: torch.nn.Module, 
        batch_size: int = 8, 
        max_epochs:int = 50, 
        loss_function: Callable = weighted_bce_logit, 
        covid_loss_weight: list = [5.64, 1.21, 1.51]
    ) -> None:
        '''
        Classifier network module for the STOIC dataset.
        '''
        super().__init__()
        self.encoder = encoder
        # self.classifier = nn.Linear(512, 3)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.loss_function = loss_function
        self.covid_loss_weight = torch.tensor(covid_loss_weight)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        # load target transposed for the loss function
        target = torch.stack(batch['target']).T
        # get prediction for images in batch
        pred = self.forward(batch['image']['data'])
        loss = self.loss_function(pred, target, self.covid_loss_weight.to(self.device))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        target = torch.stack(batch['target']).T
        pred = self.forward(batch['image']['data'])
        loss = self.loss_function(pred, target, self.covid_loss_weight.to(self.device))
        self.log("val_loss", loss, batch_size=self.batch_size)
        return torch.stack((pred, target), dim=0)

    def validation_epoch_end(self, validation_step_outputs):
        preds, _targets = torch.cat(validation_step_outputs, dim=1)
        targets = _targets.long()
        all_acc = tmf.accuracy(preds, targets)
        all_f1 = tmf.f1_score(preds, targets)
        all_auroc = tmf.auroc(preds, targets, num_classes=3)
        self.log('val_all_metrics', {'accuracy': all_acc, 'F1-score': all_f1, 'AUC-ROC': all_auroc}, batch_size=self.batch_size)
        severe_acc = tmf.accuracy(preds[:, 0], targets[:, 0])
        severe_f1 = tmf.f1_score(preds[:, 0], targets[:, 0])
        severe_auroc = tmf.auroc(preds[:, 0], targets[:, 0])
        self.log('val_severe_metrics', {'accuracy': severe_acc, 'F1-score': severe_f1, 'AUC-ROC': severe_auroc}, batch_size=self.batch_size)
        covid_acc = tmf.accuracy(preds[:, 1], targets[:, 1])
        covid_f1 = tmf.f1_score(preds[:, 1], targets[:, 1])
        covid_auroc = tmf.auroc(preds[:, 1], targets[:, 1])
        self.log('val_covid_metrics', {'accuracy': covid_acc, 'F1-score': covid_f1, 'AUC-ROC': covid_auroc}, batch_size=self.batch_size)

    def configure_optimizers(self):
        """
        Setup the optimizer and scheduler
        """

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam([
        #     {'params': self.encoder.layer3.parameters(), 'lr': 1e-6},
        #     {'params': self.encoder.layer4.parameters()},
        #     {'params': self.encoder.fc.parameters()}
        # ], lr=1e-3)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, warmup_start_lr=1e-6, max_epochs=self.max_epochs)
        return [optimizer], [scheduler]