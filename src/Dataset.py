import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CustomDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.pivoted_data = None
        self.X = None
        self.y = None

    def load_data(self):
        chunks = []
        chunk_size = 100000
        for chunk in pd.read_csv(self.filepath, chunksize=chunk_size):
            chunk["stay_id"] = chunk["stay_id"].astype(np.int32)
            chunk["hour"] = pd.to_datetime(chunk["hour"])
            chunk["rr_mean"] = chunk["rr_mean"].round().astype(np.int16)
            chunk["mortality"] = chunk["mortality"].astype(np.int8)
            chunks.append(chunk)
        self.data = pd.concat(chunks)

    def preprocess_data(self):
        self.make_hours()
        self.pivot()
        self.impute()

        print(self.pivoted_data.head())

        self.X = self.reshape()
        self.normalize()

        # get mortality  from original data
        self.y = (
            self.data.drop_duplicates(subset=["stay_id"])
            .set_index("stay_id")["mortality"]
            .values
        )

    def make_hours(self):
        data = self.data.copy()
        data.sort_values(by=["stay_id", "hour"], inplace=True)
        data["adjusted_hour"] = data.groupby("stay_id").cumcount()
        data["adjusted_hour"] = data["adjusted_hour"].astype(np.int8)
        self.data = data

    def pivot(self):
        data = self.data.copy()
        pivoted_data = data.pivot(
            index="stay_id", columns="adjusted_hour", values="rr_mean"
        )
        pivoted_data = pivoted_data.sort_index(axis=1)
        self.pivoted_data = pivoted_data

    def impute(self):
        data = self.pivoted_data.copy()
        data = data.reindex(columns=[i for i in range(24)], fill_value=np.nan)
        data = data.fillna(method="ffill", axis=1)
        self.pivoted_data = data

    def reshape(self):
        X = self.pivoted_data.values.reshape(
            -1, 24, 1
        )  # (number_of_patients, sequence_length, 1)
        return X

    def normalize(self):
        X = self.X.copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X.reshape(-1, 24)).reshape(-1, 24, 1)
        self.X = X

    def get_data(self):
        if self.X is None or self.y is None:
            self.preprocess_data()
        return self.X, self.y

    def print_shapes(self):
        if self.X is None or self.y is None:
            self.preprocess_data()
        print("X shape:", self.X.shape)  # (number_of_patients, 24, 1)
        print("y shape:", self.y.shape)  # (number_of_patients,)


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
