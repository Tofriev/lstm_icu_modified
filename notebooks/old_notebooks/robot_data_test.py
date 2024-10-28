# %%
import pandas as pd 
import numpy as np
import os
import sys
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from tqdm.auto import tqdm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

#import multiprocess as mp

import pytorch_lightning as pl
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator 

from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder
import torchmetrics
#from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter



device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)  

pl.seed_everything(42)

# %%

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

X_train = pd.read_csv(os.path.join(project_root, 'data/raw/robot/X_train.csv'))
y_train = pd.read_csv(os.path.join(project_root, 'data/raw/robot/y_train.csv'))
# %%
y_train.surface.value_counts().plot(kind='bar')
# %%

# Prepro
# convert labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train.surface)
y_train['label'] = encoded_labels

FEATURE_COLS = X_train.columns.tolist()[3:]

# split into sequences
# here we use a list for each observation not a np array (yet)
sequences = []
for series_id, group in X_train.groupby('series_id'):
    sequences_features = group[FEATURE_COLS]
    label = y_train[y_train.series_id == series_id].iloc[0].label

    sequences.append((sequences_features, label))

sequences[0]









# %%[
sequences[0][1]
# %%
train_seq, test_seq = train_test_split(sequences, test_size=0.2)
len(train_seq), len(test_seq)


#%%
sequence,label = train_seq[0]
print(dict(sequence=torch.tensor(sequence.to_numpy(), dtype=torch.float32),
            label=torch.tensor(label).long()))
# %%
# Dataset 

class RobotDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.tensor(sequence.to_numpy(), dtype=torch.float32),
            label=torch.tensor(label).long(),
        )
# %%
class RobotDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = RobotDataset(self.train_sequences)
        self.test_dataset = RobotDataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0) 
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0) 
# %%
N_EPOCHS = 200
BATCH_SIZE = 64

data_module = RobotDataModule(train_seq, test_seq, BATCH_SIZE)
# %%
# Model 

class RobotModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=256, n_layers=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75
        )

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        return self.classifier(out)
# %%
# pl lightning wrapper
class SurfacePredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = RobotModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = torchmetrics.functional.accuracy(predictions, 
                        labels, 
                        task='multiclass',
                        num_classes = self.n_classes
                        )
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = torchmetrics.functional.accuracy(predictions, 
                        labels, 
                        task='multiclass',
                        num_classes = self.n_classes
                        )
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_accuracy = torchmetrics.functional.accuracy(predictions, 
                        labels, 
                        task='multiclass',
                        num_classes = self.n_classes
                        )
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss': loss, 'accuracy': step_accuracy}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

# %%
model = SurfacePredictor(
    n_features=len(FEATURE_COLS), 
    n_classes=len(label_encoder.classes_)
    )

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True, 
    monitor='val_loss',
    mode= 'min',
)

logger = TensorBoardLogger('lighning_logs', name='surface')

trainer = pl.Trainer(
    logger = logger,
    callbacks=checkpoint_callback,
    max_epochs = N_EPOCHS,
    accelerator='mps',
    enable_progress_bar=True
)

trainer.fit(model, data_module)
# %%
