# %%

import sys
import os
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from src.Dataset import CustomDataset
from src.Dataset import TensorDataset
from src.Models import LSTMModel
from src.Models import LSTMModelWithAttention
from src.DataExplorer import DataExplorer


# %%
# load data and preprocess


filepaths = {
    "respiratory_rate": os.path.join(
        project_root, "data/raw/mimiciv/resprate_mortality.csv"
    ),
    "heart_rate": os.path.join(project_root, "data/raw/mimiciv/hr.csv"),
    "glucose": os.path.join(project_root, "data/raw/mimiciv/glc.csv"),
    "mean_blood_pressure": os.path.join(project_root, "data/raw/mimiciv/mbp.csv"),
    "gcs": os.path.join(project_root, "data/raw/mimiciv/gcs_total.csv"),
}


variables = [
    "rr_value",
    "hr_value",
    "glc_value",
    "mbp_value",
    "total_gcs",
]  # for column names in csvs


data = CustomDataset(
    filepaths,
    variables,
    aggregation_freq="1H",  # T for minutes, H for hour
    impute=True,
    small_data=True,
)  # imputation with foreward fill
data.load_data()
data.preprocess_data()
#data.print_shapes()


# %%
X, y = data.get_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


















# %%

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


# fot DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TimeSeriesDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


input_size = X_train.shape[2]  #  variables
hidden_size = 128
num_layers = 2
num_classes = 2  # binarty

model = CNN_LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)


# %%
# Training
def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in train_loader:
            data = data.float().to(device)
            labels = labels.long().to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


num_epochs = 50
train(model, train_loader, criterion, optimizer, num_epochs, device)


# %%
# Evaluation
def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            labels = labels.long().to(device)

            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")


evaluate(model, test_loader, device)
