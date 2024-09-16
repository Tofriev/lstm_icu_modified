#%%
import os
import sys
import pandas as pd
import math
from sklearn.model_selection import train_test_split

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
from torchmetrics.functional import auroc



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
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import WeightedRandomSampler

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)  

y = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/mort2.csv'))
mbp = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/mbp.csv'))
gcs = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/gcs_total.csv'))
glc = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/glc.csv'))
rr = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/resprate_mortality.csv'))
rr = rr.drop(columns=['mortality'])
X = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/hr.csv'))

# age and gender
static = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/static.csv'))


# only use intersecting stay_ids
y = y[y['stay_id'].isin(X['stay_id'])]
X = X[X['stay_id'].isin(y['stay_id'])]
mbp = mbp[mbp['stay_id'].isin(y['stay_id'])]
gcs = gcs[gcs['stay_id'].isin(y['stay_id'])]
rr = rr[rr['stay_id'].isin(y['stay_id'])]
glc = glc[glc['stay_id'].isin(y['stay_id'])]

# floor charttime
X['charttime'] = pd.to_datetime(X['charttime']).dt.floor('H')
mbp['charttime'] = pd.to_datetime(mbp['charttime']).dt.floor('H')
gcs['charttime'] = pd.to_datetime(gcs['charttime']).dt.floor('H')
rr['charttime'] = pd.to_datetime(rr['charttime']).dt.floor('H')
glc['charttime'] = pd.to_datetime(glc['charttime']).dt.floor('H')

# merge with X
X = pd.merge(X, mbp, on=['stay_id', 'charttime'], how='inner')
X = pd.merge(X, gcs, on=['stay_id', 'charttime'], how='inner')
X = pd.merge(X, rr, on=['stay_id', 'charttime'], how='inner')
X = pd.merge(X, glc, on=['stay_id', 'charttime'], how='inner')


# only use 50% of the data if preferred 
y_small = train_test_split(y, test_size=0.9, stratify=y['mortality'])[0]
X_small = X[X['stay_id'].isin(y_small['stay_id'])]
print(f'shape y: {y.shape[0]}')
print(f'shape y_small: {y_small.shape[0]}')
print(f'shape X: {X.shape[0]}')
print(f'shape X_small: {X_small.shape[0]}')


y_final = y.copy()
X_final = X.copy()

# undersample
print('before undersampling: ')
y_final.mortality.value_counts().plot(kind="bar")
plt.show()
undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
y_undersampled_stayids, _ = undersampler.fit_resample(y_final[['stay_id']], y_final['mortality'])
y_undersampled = y_final[y_final['stay_id'].isin(y_undersampled_stayids['stay_id'])]
X_undersampled = X_final[X_final['stay_id'].isin(y_undersampled['stay_id'])]
print('after undersampling: ')
y_undersampled.mortality.value_counts().plot(kind="bar")
plt.show()
#%%
# create time index
X_undersampled['charttime'] = pd.to_datetime(X['charttime'])

def time_index(group):
    group['charttime'] = group['charttime'].dt.floor('H')
    
    # aggregate 
    group = group.groupby('charttime', as_index=False).agg({
        'stay_id': 'first',
        'hr_value': 'mean',
        'mbp_value': 'mean',  
        'total_gcs': 'mean',  
        'rr_value': 'mean',
        'glc_value': 'mean',
    })
    
    group = group.sort_values('charttime').reset_index(drop=True)
    
    # 24-hour range 
    first_time = group['charttime'].iloc[0].floor('H')
    full_range = pd.date_range(start=first_time, periods=24, freq='H') 
    full_range_df = pd.DataFrame({'charttime': full_range})
    
    # merge  full range 
    merged = pd.merge(full_range_df, group, how='left', on='charttime')
    merged['stay_id'] = group['stay_id'].iloc[0]  # Ensure stay_id is filled
    
    merged['time'] = range(24)
    
    return merged


X_timed= X_undersampled.groupby('stay_id').apply(time_index).reset_index(drop=True)
X_timed = X_timed.sort_values(['stay_id', 'charttime'])
X_timed = X_timed.drop(columns=['charttime'])
print(X_timed.head(10))

# impute 
X_timed_imputed = X_timed.copy()
X_timed_imputed = X_timed_imputed.ffill()
print('naaaaans')
print(X_timed_imputed.isna().sum())


# normalization 
scaler = StandardScaler()
X_timed_imputed[['hr_value', 'mbp_value', 'rr_value', 'total_gcs']] = scaler.fit_transform(X_timed_imputed[['hr_value', 'mbp_value', 'rr_value',  'total_gcs']])


print(X_timed_imputed.head(10))
print(X_timed_imputed.describe())

# %%
sequences = []
for stay_id, group in X_timed_imputed.groupby('stay_id'):
    features = group[['hr_value', 'mbp_value', 'rr_value', 'total_gcs']]
    label = y_undersampled[y_undersampled['stay_id'] == stay_id].iloc[0].mortality
    sequences.append((features, label))


labels = [seq[1] for seq in sequences]    # Labels (mortality)


train_seq, test_seq = train_test_split(sequences, test_size=0.2, stratify=labels, random_state=42)
print(len(train_seq), len(test_seq))


plabes_train = [seq[1] for seq in train_seq if seq[1] == 1]
plabels_test = [seq[1] for seq in test_seq if seq[1] == 1]

print(f'Positive labels in train: {len(plabes_train)/len(train_seq)}')
print(f'Positive labels in test: {len(plabels_test)/len(test_seq)}')

#%%
sequence,label= train_seq[0]
print(dict(
            sequence=torch.tensor(sequence.to_numpy(), dtype=torch.float32),
            label=torch.tensor(label).long(),
        ))

# %%
# Dataset

class IcuDataset(Dataset):
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


class IcuDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = IcuDataset(self.train_sequences)
        self.test_dataset = IcuDataset(self.test_sequences)

    def train_dataloader(self):
        labels = [seq[1] for seq in self.train_sequences]
        sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / sample_count
        samples_weight = np.array([weight[t] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          sampler=sampler,
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
N_EPOCHS = 60
BATCH_SIZE = 32

data_module = IcuDataModule(train_seq, test_seq, BATCH_SIZE)


#%%
# Model
class IcuModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=100, n_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.75,
            bidirectional=True        
        )

        #self.classifier = nn.Linear(n_hidden, n_classes)
        # if bidirectional = True: 
        self.classifier = nn.Linear(2*n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        #out = hidden[-1]
        #return self.classifier(out)

        # if bidirectional = True: 
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        return self.classifier(out)
    

# pl lightning wrapper
class MortalityPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes = 2):
        super().__init__()
        self.model = IcuModel(n_features, n_classes)
        class_weights = torch.tensor([1.0, 3.0]) 
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)


        #self.criterion = nn.CrossEntropyLoss()
        self.n_classes = 2

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        #print(f"Batch {batch_idx} labels: {labels.tolist()}")


        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def validation_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def test_step(self, batch, batch_idx):
        sequences = batch['sequence']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        step_auroc = auroc(probabilities, labels, task='binary')
        
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_auroc', step_auroc, prog_bar=True, logger=True)
        return {'loss': loss, 'auroc': step_auroc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer
    

# %%



model = MortalityPredictor(
    n_features=4, 
    n_classes=2,
    )

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    verbose=True, 
    monitor='val_loss',
    mode= 'min',
)

logger = TensorBoardLogger(os.path.join(project_root, 'notebooks/lightning_logs'), name='mortality')


trainer = pl.Trainer(
    logger = logger,
    callbacks=checkpoint_callback,
    max_epochs = N_EPOCHS,
    accelerator='mps',
    enable_progress_bar=True,
    gradient_clip_val=1.0,
)

trainer.fit(model, data_module)
# %%
trainer.test(model, datamodule=data_module)

# %%
