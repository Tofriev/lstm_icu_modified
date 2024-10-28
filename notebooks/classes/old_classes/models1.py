import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC


###########################################
## LSTM Model
###########################################

class IcuModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.n_hidden,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=config.bidirectional        
        )
        hidden_size = config.n_hidden * (2 if config.bidirectional else 1)
        self.classifier = nn.Linear(hidden_size, config.n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        hidden_fwd, hidden_bwd = hidden[-2], hidden[-1]
        out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        return self.classifier(out)

class MortalityPredictor(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = IcuModel(config)
        class_weights = torch.tensor(config.class_weights) if config.class_weights else None
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.auroc = AUROC(task='binary')

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = self.criterion(output, labels) if labels is not None else 0
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences, labels = batch['sequence'], batch['label']
        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = self.auroc(probabilities, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def validation_step(self, batch, batch_idx):
        sequences, labels = batch['sequence'], batch['label']
        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = self.auroc(probabilities, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def test_step(self, batch, batch_idx):
        sequences, labels = batch['sequence'], batch['label']
        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = self.auroc(probabilities, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        return optimizer
