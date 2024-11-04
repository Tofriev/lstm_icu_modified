import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import auroc
from utils import set_seed

set_seed(42)

class AttentionLSTM(pl.LightningModule):
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
        attention_type,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bidirectional = bidirectional

        #  LSTM
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.num_directions = 2 if bidirectional else 1
        lstm_output_size = n_hidden * self.num_directions

        #  Attention 
        if attention_type == 'additive':
            self.attention = AdditiveAttention(lstm_output_size)
        elif attention_type == 'dot':
            self.attention = DotProductAttention()
        else:
            raise ValueError("attention_type must be 'additive' or 'dot'")

        # classifier
        self.classifier = nn.Linear(lstm_output_size, n_classes)

        # Loss 
        class_weights_tensor = torch.tensor(class_weights) if class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        self.lstm.flatten_parameters()
        lstm_outputs, _ = self.lstm(x)
        # lstm_outputs shape: (batch_size, seq_len, lstm_output_size)

        # Attention
        context_vector, attention_weights = self.attention(lstm_outputs)
        # context_vector shape: (batch_size, lstm_output_size)

        # classification
        out = self.classifier(context_vector)
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

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        score = torch.tanh(self.W(lstm_outputs))  # shape: (batch_size, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.v(score), dim=1)  # shape: (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)  # shpe: (batch_size, hidden_dim)
        return context_vector, attention_weights

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        #  query is the last hidden state
        query = lstm_outputs[:, -1, :].unsqueeze(1)  # shape: (batch_size, 1, hidden_dim)
        keys = lstm_outputs  # sape: (batch_size, seq_len, hidden_dim)
        attention_scores = torch.bmm(keys, query.transpose(1, 2))  # shape: (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  #shape: (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_outputs, dim=1)  # swhape: (batch_size, hidden_dim)
        return context_vector, attention_weights
