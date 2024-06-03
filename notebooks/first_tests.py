# %%

import sys
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
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
}


variables = ["rr_value", "hr_value"]
data = CustomDataset(
    filepaths, variables, aggregation_freq="30T", impute=True
)  # imputation with foreward fill
data.load_data()
data.preprocess_data()
data.print_shapes()

X, y = data.get_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


lstm_model = LSTMModel(input_size=2, hidden_size=50, num_layers=2, output_size=1).to(
    device
)
lstm_at_model = LSTMModelWithAttention(
    input_size=2, hidden_size=50, num_layers=2, output_size=1
).to(device)


criterion = nn.BCELoss(reduction="mean")
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
optimizer_lstm_at = torch.optim.Adam(lstm_at_model.parameters(), lr=0.001)


# training function


def train_model(model, optimizer, train_loader, num_epochs=50):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            if torch.isnan(outputs).any():
                print("NaN values found in outputs during training")
            loss = criterion(outputs, y_batch.unsqueeze(1)) + 1e-07
            if torch.isnan(loss).any():
                print("NaN values found in loss during training")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        average_epoch_loss = epoch_loss / len(train_loader)
        losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {average_epoch_loss}")
    return losses


# train models
losses_lstm = train_model(lstm_model, optimizer_lstm, train_loader)
losses_lstmat = train_model(lstm_at_model, optimizer_lstm_at, train_loader)

# plot losses
plt.figure(figsize=(10, 5))
plt.plot(losses_lstm, label="LSTM")
plt.plot(losses_lstmat, label="LSTM with Attention")
plt.title("Model training loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()


# evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if torch.isnan(outputs).any():
                print("NaN values found in outputs")
            predictions = outputs.round()
            if torch.isnan(predictions).any():
                print("NaN values found in predictions")
            all_labels.extend(y_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    return accuracy, precision, recall, roc_auc


# evaluate models
accuracy_lstm, precision_lstm, recall_lstm, roc_auc_lstm = evaluate_model(
    lstm_model, test_loader
)

accuracy_lstm_at, precision_lstm_at, recall_lstm_at, roc_auc_lstm_at = evaluate_model(
    lstm_at_model, test_loader
)

print(
    f"LSTM - Accuracy: {accuracy_lstm:.4f}, Precision: {precision_lstm:.4f}, Recall: {recall_lstm:.4f}, ROC AUC: {roc_auc_lstm:.4f}"
)
print(
    f"LSTM-AT - Accuracy: {accuracy_lstm_at:.4f}, Precision: {precision_lstm_at:.4f}, Recall: {recall_lstm_at:.4f}, ROC AUC: {roc_auc_lstm_at:.4f}"
)

# %%
