import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import auroc
from utils import set_seed

set_seed(42)


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_classes,
        n_hidden,
        n_layers,
        dropout,
        bidirectional,
        learning_rate,
        weight_decay,
        class_weights,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        hidden_size = n_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Linear(hidden_size, n_classes)
        class_weights_tensor = torch.tensor(class_weights) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        if self.hparams.bidirectional:
            hidden_fwd = hidden[-2]
            hidden_bwd = hidden[-1]
            out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            out = hidden[-1]
        return self.classifier(out)

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "train_auroc",
            step_auroc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "val_auroc",
            step_auroc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log(
            "test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self.log(
            "test_auroc",
            step_auroc,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
