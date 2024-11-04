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

from scipy.stats import cumfreq
from scipy.stats import ttest_ind

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
weight = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/weight.csv'))
# age, gender, height, timeframe
static = pd.read_csv(os.path.join(project_root, 'data/raw/mimiciv/static_data.csv'))


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
X_MIMIC = merged_df.copy()









#######################
######## TUDD #########
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

X_TUDD = merged_df.copy()


print(f'MIMIC: {X_MIMIC.shape}')
print(f'TUDD: {X_TUDD.shape}')

print(f'MIMC: {X_MIMIC.describe()}')
print(f'TUDD: {X_TUDD.describe()}')




#%%
# compare distributions
for feature in SEQUENCE_FEATURES:
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(X_MIMIC[feature].dropna(), label='MIMIC', fill=True, color='blue')
    sns.kdeplot(X_TUDD[feature].dropna(), label='TUDD', fill=True, color='orange')
    
    plt.title(f'Distribution of {feature} in MIMIC vs TUDD')
    plt.legend()
    plt.show()

#%%
# cdf

def plot_cdf(feature, data_mimic, data_tudd):
    plt.figure(figsize=(12, 6))

    cumfreq_mimic = cumfreq(data_mimic.dropna(), numbins=100)
    x_mimic = cumfreq_mimic.lowerlimit + np.linspace(0, cumfreq_mimic.binsize * cumfreq_mimic.cumcount.size, cumfreq_mimic.cumcount.size)

    cumfreq_tudd = cumfreq(data_tudd.dropna(), numbins=100)
    x_tudd = cumfreq_tudd.lowerlimit + np.linspace(0, cumfreq_tudd.binsize * cumfreq_tudd.cumcount.size, cumfreq_tudd.cumcount.size)

    plt.plot(x_mimic, cumfreq_mimic.cumcount/cumfreq_mimic.cumcount.max(), label='MIMIC', color='blue')

    plt.plot(x_tudd, cumfreq_tudd.cumcount/cumfreq_tudd.cumcount.max(), label='TUDD', color='orange')

    plt.title(f'CDF of {feature} in MIMIC vs TUDD')
    plt.xlabel(f'{feature} values')
    plt.ylabel('Cumulative Density')
    plt.legend()
    plt.show()

for feature in SEQUENCE_FEATURES:
    plot_cdf(feature, X_MIMIC[feature], X_TUDD[feature])

# %%
# compare correlations
plt.figure(figsize=(12, 8))
sns.heatmap(X_MIMIC[SEQUENCE_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix for MIMIC")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(X_TUDD[SEQUENCE_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix for TUDD")
plt.show()

# %%
# missings
# Check missing data percentage for MIMIC and TUDD
print(f'Missing values in MIMIC:\n{X_MIMIC.isnull().sum()/len(X_MIMIC) * 100}')
print(f'Missing values in TUDD:\n{X_TUDD.isnull().sum()/len(X_TUDD) * 100}')

# %%
# ttets
# %%

for feature in SEQUENCE_FEATURES:
    t_stat, p_value = ttest_ind(X_MIMIC[feature].dropna(), X_TUDD[feature].dropna(), equal_var=False)
    print(f'T-test for {feature}: t-stat = {t_stat}, p-value = {p_value}')

# %%
