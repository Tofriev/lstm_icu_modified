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
                     #,'rr_value'
                     ,'total_gcs'
                     ,'glc_value'
                     ,'creatinine_value'
                     ,'potassium_value'
                    # ,'sodium_value'
                     ,'wbc_value'
                     ,'platelets_value'
                     ,'inr_value'
                     ,'anion_gap_value'
                     ,'lactate_value'
                     #,'urea_value'
                     ,'temperature_value'
                     ,'weight_value'
                     ]
NUMERICAL_FEATURES = SEQUENCE_FEATURES + ['age', 'height']
CAT_FEATURES = ['gender']
ALL_FEATURES = NUMERICAL_FEATURES + CAT_FEATURES
# best results were archived with the following feature datasets
# [mbp, gcs, glc, rr, creatinine, hr, potassium, sodium]
# [mbp, gcs, glc, rr, creatinine, hr, potassium, sodium, wbc, platelets, inr, anion_gap, lactate]
DATASETS = [mbp, gcs, glc, creatinine, hr, potassium, wbc, platelets, inr, anion_gap, lactate, temperature, weight]# urea, sodium, rr

# use small amount of data for testing
small = True 
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

print(X_undersampled.columns)
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
print(f"shape MIMIC: {train_seq[0][0].shape}")

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
        samples_weight = np.array([weight[int(t)] for t in labels])
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
N_EPOCHS = 3
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

#trainer.fit(model, data_module)
# %%
#trainer.test(model, datamodule=data_module)
#%%%

# %%

#################################################
####### Extract the TUDD data for testing #######
#################################################

measurements = pd.read_csv(os.path.join(project_root, 'data/raw/tudd/tudd_complete.csv'), sep='|')
mortality_info_x = pd.read_csv(os.path.join(project_root, 'data/raw/tudd/stays_ane.csv'), sep='|')
mortality_info_y = pd.read_csv(os.path.join(project_root, 'data/raw/tudd/stays_others2_ane.csv'), sep='|')

mortality_info = pd.concat([mortality_info_x, mortality_info_y])

measurements['measurement_offset'] = pd.to_numeric(measurements['measurement_offset'], errors='coerce')

measurements = pd.merge(
    measurements,
    mortality_info[['caseid','stay_duration', 'age', 'gender', 'bodyheight', 'bodyweight', 'exitus']],
    on='caseid',
    how='left'
)
measurements['stay_duration_hours'] = measurements['stay_duration'] * 24
measurements['measurement_time_from_admission'] = measurements['stay_duration_hours'] + measurements['measurement_offset']


# clean negative values in measurement_time_from_admission as that indicates a measurement before admission
# small negative values are allowed due to possible errors and capped to 0
measurements = measurements[measurements['measurement_time_from_admission'] > -1]
measurements.loc[measurements['measurement_time_from_admission'] <= -1, 'measurement_time_from_admission'] = 0

# only keep first 24h
measurements = measurements[(measurements['measurement_time_from_admission'] >= 0) &
                            (measurements['measurement_time_from_admission'] <= 24)]

# round down to hour
measurements['measurement_time_from_admission'] = np.floor(measurements['measurement_time_from_admission'])


# aggregate measurements 
measurements['value'] = pd.to_numeric(measurements['value'], errors='coerce')
measurements_agg = measurements.groupby(['caseid', 'measurement_time_from_admission', 'treatmentname'])['value'].mean().reset_index()

# pivot treatmentnames to columns
measurements_pivot = measurements_agg.pivot_table(
    index=['caseid', 'measurement_time_from_admission'],
    columns='treatmentname',
    values='value'
).reset_index()

# time grid
def create_time_grid(mortality_info):
    df_list = []
    for _, row in mortality_info.iterrows():
        caseid = row['caseid']
        time_range = np.arange(0, 25)  # hour 0 to 24 inclusive
        time_df = pd.DataFrame({'caseid': caseid, 'measurement_time_from_admission': time_range})
        df_list.append(time_df)
    return pd.concat(df_list, ignore_index=True)

time_grid = create_time_grid(mortality_info)

# merge on time grid
merged_df = pd.merge(time_grid, measurements_pivot, on=['caseid', 'measurement_time_from_admission'], how='left')
merged_df = pd.merge(
    merged_df,
    mortality_info[['caseid', 'age', 'gender', 'bodyheight', 'bodyweight', 'exitus']],
    on='caseid',
    how='left'
)

# map treatmentnames
treatmentnames_mapping = {
    'HF': 'hr_value',
    'AGAP': 'anion_gap_value',
    'GLUC': 'glc_value',
    'CREA': 'creatinine_value',
    'K': 'potassium_value',
    'LEU': 'wbc_value',
    'THR': 'platelets_value',
    'Q': 'inr_value',
    'LAC': 'lactate_value',
    'T': 'temperature_value',
    'GCS': 'total_gcs',
    'MAP': 'mbp_value',
    'bodyweight': 'weight_value',
    'bodyheight': 'height_value'
}
cols_to_rename = {old_name: treatmentnames_mapping.get(old_name, old_name) for old_name in merged_df.columns}
merged_df.rename(columns=cols_to_rename, inplace=True)


#%%
SEQUENCE_FEATURES = [
    'hr_value',
    'mbp_value',
    'total_gcs',
    'glc_value',
    'creatinine_value',
    'potassium_value',
    'wbc_value',
    'platelets_value',
    'inr_value',
    'anion_gap_value',
    'lactate_value',
    'temperature_value',
    'weight_value'
]

NUMERICAL_FEATURES = SEQUENCE_FEATURES + ['age', 'height_value']
CAT_FEATURES = ['gender']
ALL_FEATURES = NUMERICAL_FEATURES + CAT_FEATURES

# drop 'ALAT', 'ALB', 'ASAT', 'BR', 'BRc', 'CRP', 'FIO2', 'GFR', 'HB',
#  'HCO3', 'HFF', 'PACO2', 'PAO2', 'PCT', 'PFR', 'PH', 'Q', 'RASS', 'RF', 'TROPT'
merged_df.drop(['ALAT', 'ALB', 'ASAT', 'BR', 'BRc', 'CRP', 'FIO2', 'GFR', 'HB',
                'HCO3', 'HFF', 'PACO2', 'PAO2', 'PCT', 'PFR', 'PH', 'RASS', 'RF', 'TROPT'], axis=1, inplace=True)
#%%
# define bnoundaries
bounds = {
    'age': (18, 90),
    'weight_value': (20, 500),
    'height_value': (20, 260),
    'temperature_value': (20, 45),
    'hr_value': (10, 300),  
    'glc_value': (5, 2000),  
    'mbp_value': (20, 400),  
    'potassium_value': (2.5, 7),
    'wbc_value': (1, 200),  
    'platelets_value': (10, 1000),  
    'inr_value': (0.2, 6),  
    'anion_gap_value': (1, 25),
    'lactate_value': (0.1, 200),
    'creatinine_value': (0.1, 20)
}

# drop observbation for age < 18, set to 90 if > 90
merged_df = merged_df[merged_df['age'] >= 18] 
merged_df['age'] = merged_df['age'].apply(lambda x: min(x, 90))

for feature, (lower, upper) in bounds.items():
    if feature in merged_df.columns:
        merged_df.loc[merged_df[feature] < lower, feature] = np.nan
        merged_df.loc[merged_df[feature] > upper, feature] = np.nan


# impute ffill and bfill and mean 
merged_df.sort_values(['caseid', 'measurement_time_from_admission'], inplace=True)
merged_df = merged_df.groupby('caseid').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

global_means = merged_df[NUMERICAL_FEATURES].mean()
merged_df[NUMERICAL_FEATURES] = merged_df[NUMERICAL_FEATURES].fillna(global_means)

merged_df['gender'] = merged_df['gender'].map({'m': 0, 'w': 1})
merged_df['gender'].fillna(merged_df['gender'].mode()[0], inplace=True)

merged_df['exitus'].fillna(0, inplace=True)
print(merged_df.head(50))
# scaling
# zero or near-zero variance features
variance_threshold = 1e-8
feature_variances = merged_df[NUMERICAL_FEATURES].var()
zero_variance_features = feature_variances[feature_variances < variance_threshold].index.tolist()
print("Zero or near-zero variance features:", zero_variance_features)


unique_counts = merged_df[NUMERICAL_FEATURES].nunique()
constant_features = unique_counts[unique_counts == 1].index.tolist()
print("Constant features:", constant_features)


NUMERICAL_FEATURES = [feature for feature in NUMERICAL_FEATURES if feature not in zero_variance_features]


scaler = StandardScaler()
merged_df[NUMERICAL_FEATURES] = scaler.fit_transform(merged_df[NUMERICAL_FEATURES])
print(merged_df.head(50))
print(merged_df.columns)
column_order = [
    'caseid', 'measurement_time_from_admission','mbp_value', 'total_gcs', 'glc_value', 'creatinine_value', 'hr_value', 
    'potassium_value', 'wbc_value', 'platelets_value', 'inr_value', 
    'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value', 
    'age', 'gender', 'height_value', 'exitus'
]
sorted_merged_df = merged_df[column_order]
print(sorted_merged_df.columns)

# checking data
sequences = []
invalid_caseids = []

for caseid, group in sorted_merged_df.groupby('caseid'):
    if len(group) != 25:
        invalid_caseids.append(caseid)
        print(f'caseid {caseid} has {len(group)} rows')
    else:
        group = group.sort_values('measurement_time_from_admission')
        features = group[ALL_FEATURES].values
        label = group['exitus'].iloc[0]  
        sequences.append((features, label))

if not invalid_caseids:
    print("All caseids have 25 rows.")

labels = [seq[1] for seq in sequences]


train_seq, test_seq = train_test_split(sequences, test_size=0.2, stratify=labels, random_state=42)
print(f'Training sequences: {len(train_seq)}, Test sequences: {len(test_seq)}')







# %%
labels = [seq[1] for seq in sequences]    # Labels (mortality)


tudd_train_seq, tudd_test_seq = train_test_split(sequences, test_size=0.2, stratify=labels, random_state=42)
print(len(train_seq), len(test_seq))
print(f'shape TUDD: {train_seq[0][0].shape}')

plabes_train = [seq[1] for seq in train_seq if seq[1] == 1]
plabels_test = [seq[1] for seq in test_seq if seq[1] == 1]

print(f'Positive labels in train: {len(plabes_train)/len(train_seq)}')
print(f'Positive labels in test: {len(plabels_test)/len(test_seq)}')


tudd_data_module = IcuDataModule(tudd_train_seq, tudd_test_seq, batch_size=BATCH_SIZE)


# %%
trainer.fit(model, tudd_data_module)
trainer.test(model, datamodule=data_module)
# %%
