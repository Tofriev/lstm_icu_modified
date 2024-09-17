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
creatinine = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/creatinine.csv'))
potassium = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/potassium.csv'))
hr = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/hr.csv'))
sodium = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/sodium.csv'))
wbc = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/wbc.csv')) # leukocytes
platelets = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/platelets.csv')) # thrombocyten
inr = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/inr.csv')) # Prothrombin Time (quick)
anion_gap = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/anion_gap.csv')) 
lactate = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/lactate.csv'))
urea = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/urea.csv')) 
temperature = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/temperature.csv')) 
weight = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/weights.csv'))
# age, gender, height, timeframe
static = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/static3.csv'))


SEQUENCE_FEATURES = [
                    'hr_value'
                     ,'mbp_value'
                     ,'rr_value'
                     ,'total_gcs'
                     ,'glc_value'
                     ,'creatinine_value'
                     ,'potassium_value'
                     ,'sodium_value'
                     ,'wbc_value'
                     ,'platelets_value'
                     ,'inr_value'
                     ,'anion_gap_value'
                     ,'lactate_value'
                     ,'urea_value'
                     ,'temperature_value'
                     ,'weight_value'
                     ]
NUMERICAL_FEATURES = SEQUENCE_FEATURES + ['age', 'height']
CAT_FEATURES = ['gender']
ALL_FEATURES = NUMERICAL_FEATURES + CAT_FEATURES
# best results were archived with the following feature datasets
# [mbp, gcs, glc, rr, creatinine, hr, potassium, sodium]
# [mbp, gcs, glc, rr, creatinine, hr, potassium, sodium, wbc, platelets, inr, anion_gap, lactate]
DATASETS = [mbp, gcs, glc, rr, creatinine, hr, potassium, sodium, wbc, platelets, inr, anion_gap, lactate, urea, temperature, weight]

# use small amount of data for testing
small = False
y_small = train_test_split(y, test_size=0.9, stratify=y['mortality'])[0]
if small == True:
    for i in range(len(DATASETS)): 
        DATASETS[i] = DATASETS[i][DATASETS[i]['stay_id'].isin(y_small['stay_id'])]
    static = static[static['stay_id'].isin(y_small['stay_id'])]
    y = y[y['stay_id'].isin(y_small['stay_id'])]


# datetime format and aggregation
aggregated_datasets = []
for df in DATASETS:
    df['charttime'] = pd.to_datetime(df['charttime']).dt.floor('H')
    measurement_cols = df.columns.difference(['stay_id', 'charttime'])
    df_agg = df.groupby(['stay_id', 'charttime'], as_index=False)[measurement_cols.tolist()].mean()
    aggregated_datasets.append(df_agg)

static['intime'] = pd.to_datetime(static['intime']).dt.floor('H')
static['first_day_end'] = pd.to_datetime(static['first_day_end']).dt.floor('H')
static['gender'] = static['gender'].map({'M': 0, 'F': 1})

# create time grid for first 24h -> results in 25 rows due to flooring
def create_time_grid(df):
    df_list = []
    for index, row in df.iterrows():
        stay_id = row['stay_id']
        start_time = row['intime']
        end_time = row['first_day_end']
        time_diff = end_time - start_time
        time_range = pd.date_range(start=start_time, end=end_time, freq='H')
        time_df = pd.DataFrame({'stay_id': stay_id, 'charttime': time_range})
        df_list.append(time_df)
    return pd.concat(df_list, ignore_index=True)

time_grid = create_time_grid(static)

# merge vars on time grid
merged_df = time_grid.copy()
for data in aggregated_datasets: 
    merged_df = pd.merge(merged_df, data, on=['stay_id', 'charttime'], how='left')
merged_df = pd.merge(merged_df, static[['stay_id', 'age', 'gender', 'height']], on='stay_id', how='left')
X = merged_df.copy()

print(X.columns)

# impute 
print(f'number of missing values: {X.isna().sum()}')
X.sort_values(['stay_id', 'charttime'], inplace=True)
X = X.groupby('stay_id').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
global_means = X[NUMERICAL_FEATURES].mean()
X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].fillna(global_means)

X['height'].fillna(X['height'].mean(), inplace=True)
X['age'].fillna(X['age'].mean(), inplace=True)
X['gender'].fillna(X['gender'].mode()[0], inplace=True)
print(f'number of missing values after imputation: {X.isna().sum()}')

# undersample
print('before undersampling: ')
y.mortality.value_counts().plot(kind="bar")
plt.show()
undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
y_undersampled_stayids, _ = undersampler.fit_resample(y[['stay_id']], y['mortality'])
y_undersampled = y[y['stay_id'].isin(y_undersampled_stayids['stay_id'])]
X_undersampled = X[X['stay_id'].isin(y_undersampled['stay_id'])]
print('after undersampling: ')
y_undersampled.mortality.value_counts().plot(kind="bar")
plt.show()

# scaling 
scaler = StandardScaler()
X_undersampled[NUMERICAL_FEATURES] = scaler.fit_transform(X_undersampled[NUMERICAL_FEATURES])

# check sequences
stay_ids_with_missing_rows = []
for stay_id, group in X_undersampled.groupby('stay_id'):
    if len(group) != 25:        
        stay_ids_with_missing_rows.append(stay_id)

if len(stay_ids_with_missing_rows) > 0:    
    print("stayid with != 25 rows: ")    
    print(stay_ids_with_missing_rows)
else:
    print("all stayids have 25 rows ")

#%%
sequences = []
for stay_id, group in X_undersampled.groupby('stay_id'):
    group = group.sort_values('charttime')
    features = group[ALL_FEATURES].values
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
            sequence=torch.tensor(sequence, dtype=torch.float32),
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
            sequence=torch.tensor(sequence, dtype=torch.float32),
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
N_EPOCHS = 15
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
        # 'if bidirectional = True': 
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
    n_features=len(ALL_FEATURES), 
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
