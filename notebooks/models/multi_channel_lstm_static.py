import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import auroc
from utils import set_seed

set_seed(42)

class MultiChannelLSTMStatic(pl.LightningModule):
    def __init__(self, n_seq_features, n_static_features, n_classes, n_hidden, n_layers, dropout, bidirectional, learning_rate, weight_decay, class_weights, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        

        self.lstm_channels = nn.ModuleList([
            nn.LSTM(
                input_size=1,  # one feature per channel
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            ) for _ in range(n_seq_features)
        ])
        
        hidden_size = n_hidden * (2 if bidirectional else 1)

        self.classifier = nn.Linear(hidden_size * n_seq_features + n_static_features, n_classes)
        
        class_weights_tensor = torch.tensor(class_weights) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, seq, static):
        """
        seq: Tensor of shape (batch, time_steps, n_seq_features)
        static: Tensor of shape (batch, n_static_features)
        """
        channel_outputs = []
        
        for i, lstm in enumerate(self.lstm_channels):
            feature_sequence = seq[:, :, i].unsqueeze(-1)
            lstm_out, (hidden, _) = lstm(feature_sequence)
            
            if self.hparams.bidirectional:
                hidden_fwd = hidden[-2]
                hidden_bwd = hidden[-1]
                channel_output = torch.cat((hidden_fwd, hidden_bwd), dim=1)
            else:
                channel_output = hidden[-1]
            
            channel_outputs.append(channel_output)
        
        combined_seq_output = torch.cat(channel_outputs, dim=1)
        # concatenate with static features
        combined = torch.cat((combined_seq_output, static), dim=1)
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        static = batch['static']
        labels = batch['label']
        outputs = self(sequences, static)
        loss = self.criterion(outputs, labels)
        
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_auroc', step_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        static = batch['static']
        labels = batch['label']
        outputs = self(sequences, static)
        loss = self.criterion(outputs, labels)
        
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_auroc', step_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        static = batch['static']
        labels = batch['label']
        outputs = self(sequences, static)
        loss = self.criterion(outputs, labels)
        
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('test_auroc', step_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
