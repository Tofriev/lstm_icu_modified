import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import auroc

class LSTMModelWithStatic(pl.LightningModule):
    def __init__(self, n_seq_features, n_static_features, n_classes,
                 n_hidden, n_layers, dropout, bidirectional,
                 learning_rate, weight_decay, class_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # LSTM processes only the sequential (time series) features.
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        hidden_size = n_hidden * (2 if bidirectional else 1)
        # Final classifier now concatenates LSTM output with static features.
        self.classifier = nn.Linear(hidden_size + n_static_features, n_classes)
        if class_weights:
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float())
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, seq_x, static_x):
        """
        seq_x: Tensor of shape (batch, time_steps, n_seq_features)
        static_x: Tensor of shape (batch, n_static_features)
        """
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(seq_x)
        if self.hparams.bidirectional:
            hidden_fwd = hidden[-2]
            hidden_bwd = hidden[-1]
            hidden_out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            hidden_out = hidden[-1]
        combined = torch.cat((hidden_out, static_x), dim=1)
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        seq = batch["sequence"]
        static = batch["static"]
        labels = batch["label"]
        outputs = self(seq, static)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_auroc", step_auroc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq = batch["sequence"]
        static = batch["static"]
        labels = batch["label"]
        outputs = self(seq, static)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_auroc", step_auroc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        seq = batch["sequence"]
        static = batch["static"]
        labels = batch["label"]
        outputs = self(seq, static)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task="binary")
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_auroc", step_auroc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
