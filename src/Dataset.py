import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CustomDataset:
    def __init__(self, filepath, variables, aggregation_freq=None, impute=False):
        self.filepath = filepath
        self.variables = variables
        self.data = None
        self.pivoted_data = None
        self.X = None
        self.y = None
        self.aggregation_freq = aggregation_freq
        self.impute = impute

    def load_data(self):
        chunks = []
        chunk_size = 100000
        for chunk in pd.read_csv(self.filepath, chunksize=chunk_size):
            chunk["stay_id"] = chunk["stay_id"].astype(np.int32)
            chunk["charttime"] = pd.to_datetime(chunk["charttime"])
            for var in self.variables:
                chunk[var] = chunk[var].astype(np.float32)
            chunk["mortality"] = chunk["mortality"].astype(np.int8)
            chunks.append(chunk)
        self.data = pd.concat(chunks)

    def preprocess_data(self):
        self.load_data()
        if self.aggregation_freq:
            self.aggregate_data()
        self.make_time_indices()
        self.pivot()
        if self.impute:
            self.imputer()

        self.X = self.reshape()
        self.normalize()

        # get mortality from original data
        self.y = (
            self.data.drop_duplicates(subset=["stay_id"])
            .set_index("stay_id")["mortality"]
            .values
        )

    def aggregate_data(self):
        if self.aggregation_freq:
            # mortality data is seperated
            mortality_data = self.data[["stay_id", "mortality"]].drop_duplicates()

            # charttimes are aggregated
            aggregated_data = []
            for stay_id, group in self.data.groupby("stay_id"):
                group = (
                    group.set_index("charttime").resample(self.aggregation_freq).mean()
                )
                group["stay_id"] = stay_id
                aggregated_data.append(group.reset_index())

            self.data = pd.concat(aggregated_data)

            # nortality is merged back
            self.data = self.data.drop(columns="mortality", errors="ignore").merge(
                mortality_data, on="stay_id", how="left"
            )
            print(
                "Data aggregated:",
                self.data.head(),
            )

    def make_time_indices(self):
        data = self.data.copy()
        data.sort_values(by=["stay_id", "charttime"], inplace=True)
        data["adjusted_time"] = data.groupby("stay_id").cumcount()
        data["adjusted_time"] = data["adjusted_time"].astype(np.int32)
        self.data = data
        print("Data time indices", self.data.head())

    def pivot(self):
        data = self.data.copy()
        pivoted_data = {}  # store each var in dict entry
        for var in self.variables:
            pivoted_data[var] = data.pivot(
                index="stay_id", columns="adjusted_time", values=var
            )
        self.pivoted_data = pivoted_data
        print("Data pivoted:")
        for var in self.variables:
            print(f"{var}:")
            print(self.pivoted_data[var].head())

    def imputer(self):
        print("nans before imputation:")
        for var in self.variables:
            print(f"{var}: {self.pivoted_data[var].isna().sum().sum()}")

        for var in self.variables:
            data = self.pivoted_data[var].copy()
            max_time_steps = self.data["adjusted_time"].max() + 1
            data = data.reindex(
                columns=[i for i in range(max_time_steps)], fill_value=np.nan
            )
            data = data.fillna(method="ffill", axis=1)
            self.pivoted_data[var] = data

        print("nans after imputation:")
        for var in self.variables:
            print(f"{var}: {self.pivoted_data[var].isna().sum().sum()}")

    def reshape(self):
        arrays = []
        for var in self.variables:
            arrays.append(self.pivoted_data[var].values)
        X = np.stack(
            arrays, axis=-1
        )  # (number_of_patients, sequence_length, number_of_variables)
        print("Data reshaped:")
        print(X.head())
        print(X.shape)
        return X

    def normalize(self):
        X = self.X.copy()
        num_vars = X.shape[-1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        for i in range(num_vars):
            X[:, :, i] = scaler.fit_transform(X[:, :, i])
        self.X = X
        print("Data normalized:")
        print(self.X.head())
        print(self.X.shape)

    def get_data(self):
        if self.X is None or self.y is None:
            self.preprocess_data()
        return self.X, self.y

    def print_shapes(self):
        if self.X is None or self.y is None:
            self.preprocess_data()
        print(
            "X shape:", self.X.shape
        )  # (number_of_patients, sequence_length, number_of_variables)
        print("y shape:", self.y.shape)  # (number_of_patients,)


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
