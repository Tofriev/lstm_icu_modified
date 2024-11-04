import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import auroc
from utils import set_seed

set_seed(42)

class CNN_LSTM(pl.LightningModule):
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
        architecture,  
        cnn_out_channels,
        kernel_size,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.architecture = architecture

        #  CNN
        if architecture in ['cnn_lstm', 'parallel']:
            self.cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=n_features,
                    out_channels=cnn_out_channels,
                    kernel_size=kernel_size,
                    padding='same'
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # out shape: (batch_size, cnn_out_channels, 1)
                nn.Flatten()  # out shape: (batch_size, cnn_out_channels)
            )

        #  LSTM
        lstm_input_size = cnn_out_channels if architecture == 'cnn_lstm' else n_features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        hidden_size = n_hidden * (2 if bidirectional else 1)

        if architecture == 'lstm_cnn':
            # CNN after LSTM
            self.post_lstm_cnn = nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=cnn_out_channels,
                    kernel_size=kernel_size,
                    padding='same'
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            )
            self.classifier = nn.Linear(cnn_out_channels, n_classes)
        elif architecture == 'parallel':
            concatenated_size = hidden_size + cnn_out_channels
            self.classifier = nn.Linear(concatenated_size, n_classes)
        else:
            # for 'cnn_lstm' 
            self.classifier = nn.Linear(hidden_size, n_classes)

        # loss 
        class_weights_tensor = torch.tensor(class_weights) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        if self.architecture == 'cnn_lstm':
            # shape in: (batch_size, seq_len, n_features)
            x_cnn = x.permute(0, 2, 1)  # shape: (batch_size, n_features, seq_len)
            x_cnn = self.cnn(x_cnn)  # shape: (batch_size, cnn_out_channels)
            x_lstm = x_cnn.unsqueeze(1)  # shape: (batch_size, 1, cnn_out_channels)
            self.lstm.flatten_parameters()
            _, (hidden, _) = self.lstm(x_lstm)
            if self.hparams.bidirectional:
                hidden_fwd = hidden[-2]
                hidden_bwd = hidden[-1]
                out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
            else:
                out = hidden[-1]
            out = self.classifier(out)
            return out

        elif self.architecture == 'lstm_cnn':
            # shape in: (batch_size, seq_len, n_features)
            self.lstm.flatten_parameters()
            outputs, _ = self.lstm(x)
            # shapw out (batch_size, seq_len, hidden_size)
            outputs = outputs.permute(0, 2, 1)  # shape: (batch_size, hidden_size, seq_len)
            x_cnn = self.post_lstm_cnn(outputs)  # shape: (batch_size, cnn_out_channels)
            out = self.classifier(x_cnn)
            return out

        elif self.architecture == 'parallel':
            # LSTM 
            self.lstm.flatten_parameters()
            _, (hidden, _) = self.lstm(x)
            if self.hparams.bidirectional:
                hidden_fwd = hidden[-2]
                hidden_bwd = hidden[-1]
                lstm_out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
            else:
                lstm_out = hidden[-1]

            # CNN 
            x_cnn = x.permute(0, 2, 1)  # sape: (batch_size, n_features, seq_len)
            cnn_out = self.cnn(x_cnn)  # Sahape: (batch_size, cnn_out_channels)

            # concat outputs
            out = torch.cat((lstm_out, cnn_out), dim=1)
            out = self.classifier(out)
            return out

        

    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_auroc', step_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_auroc', step_auroc, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        outputs = self(sequences)
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
