import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models
from torchvision import transforms, datasets
import torchsummary
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils

from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.functional import accuracy

import pytorch_lightning as pl


class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #self.feature = resnet18(pretrained=True)
        #再学習対策？
        self.feature = resnet18(weights=ResNet18_Weights.DEFAULT)

        # モデルの重みを固定(転移学習)
        for param in self.feature.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(1000, 3)


    def forward(self, x):
        #x = torch.tensor(x, dtype=torch.float32, requires_grad=False)
        h = self.feature(x)
        h = self.fc(h)
        return h


    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer
