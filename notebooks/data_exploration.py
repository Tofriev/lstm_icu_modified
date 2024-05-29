# %%
import sys
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from src.Dataset import CustomDataset
from src.Dataset import TensorDataset
from src.Models import LSTMModel
from src.Models import LSTMModelWithAttention
from src.DataExplorer import DataExplorer

csv_path = os.path.join(project_root, "data/raw/mimiciv/mimic_not_aggregated.csv")
print("CSV Path:", csv_path)


# %%
# load data and preprocess
variables = ["rr_value"]
data = CustomDataset(csv_path, variables, aggregation_freq="30T", impute=True)
data.load_data()
data.preprocess_data()
data.print_shapes()

X, y = data.get_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# explore data
# exploration = DataExplorer(data)
# exploration.run_all()
