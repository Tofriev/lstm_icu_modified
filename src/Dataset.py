import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CustomDataset:
    """
    Custom dataset class for loading

    Approach: load data for each varriable seperately, aggregate each seperatly and then merge them. Solvesthe problem of having a massive csv file due to different charttimes for each variable.
    """

    def __init__(self, filepaths, variables, aggregation_freq=None, impute=False):
        self.filepaths = filepaths
        self.variables = variables
        self.data = {}
        self.pivoted_data = None
        self.X = None
        self.y = None
        self.aggregation_freq = aggregation_freq
        self.impute = impute
        self.primary_df = None
        self.mortality = None

    def load_data(self):
        self.primary_df = self._load_csv(
            self.filepaths["respiratory_rate"], primary=True
        )
        self.mortality = self.primary_df[["stay_id", "mortality"]].drop_duplicates()
        for var, filepath in self.filepaths.items():
            if var != "respiratory_rate":
                self.data[var] = self._load_csv(filepath, primary=False)

    def _load_csv(self, filepath, primary=False):
        chunks = []
        chunk_size = 100000
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunk["stay_id"] = chunk["stay_id"].astype(np.int32)
            chunk["charttime"] = pd.to_datetime(chunk["charttime"])
            if not primary:
                chunk = chunk[
                    chunk["stay_id"].isin(self.primary_df["stay_id"].unique())
                ]  # in primary df we have the relevant subjects whi did not die in first 24h
            chunks.append(chunk)
        return pd.concat(chunks)

    def preprocess_data(self):
        self.load_data()
        if self.aggregation_freq:
            self.aggregate_data()
        self.make_time_indices()

        # here the different DFs of each variable are merged
        self.merge_data()

        self.pivot()
        if self.impute:
            self.imputer()

        self.X = self.reshape_to_3d_array()
        self.normalize()

        self.y = (
            self.primary_df.drop_duplicates(subset=["stay_id"])
            .set_index("stay_id")["mortality"]
            .reindex(self.pivoted_data[self.variables[0]].index)
            .values
        )

    def aggregate_data(self):
        aggregated_data = []
        for stay_id, group in self.primary_df.groupby("stay_id"):
            group = group.set_index("charttime").resample(self.aggregation_freq).mean()
            group["stay_id"] = stay_id
            aggregated_data.append(group.reset_index())
        self.primary_df = pd.concat(aggregated_data)

        for var, df in self.data.items():
            aggregated_data = []
            for stay_id, group in df.groupby("stay_id"):
                group = (
                    group.set_index("charttime").resample(self.aggregation_freq).mean()
                )
                group["stay_id"] = stay_id
                aggregated_data.append(group.reset_index())
            self.data[var] = pd.concat(aggregated_data)

    def make_time_indices(self):
        self.primary_df.sort_values(by=["stay_id", "charttime"], inplace=True)
        self.primary_df["time_index"] = self.primary_df.groupby("stay_id").cumcount()
        self.primary_df["time_index"] = self.primary_df["time_index"].astype(np.int32)

        for var, df in self.data.items():
            df.sort_values(by=["stay_id", "charttime"], inplace=True)
            df["time_index"] = df.groupby("stay_id").cumcount()
            df["time_index"] = df["time_index"].astype(np.int32)

    def merge_data(self):
        merged_data = self.primary_df.copy()
        for var, df in self.data.items():
            merged_data = pd.merge(
                merged_data, df, on=["stay_id", "time_index"], suffixes=("", f"_{var}")
            )
        self.data = merged_data

    def pivot(self):
        data = self.data.copy()
        pivoted_data = {}  # store each var in dict entry
        for var in self.variables:
            pivoted_data[var] = data.pivot(
                index="stay_id", columns="time_index", values=var
            )
        self.pivoted_data = pivoted_data
        print("Data pivoted:")
        for var in self.variables:
            print(f"{var}:")
            print(self.pivoted_data[var].head())
        print("\n")

    def imputer(self):
        print("nans before imputation:")
        for var in self.variables:
            print(f"{var}: {self.pivoted_data[var].isna().sum().sum()}")

        for var in self.variables:
            data = self.pivoted_data[var].copy()
            max_time_steps = self.data["time_index"].max() + 1
            data = data.reindex(
                columns=[i for i in range(max_time_steps)], fill_value=np.nan
            )
            data = data.fillna(method="ffill", axis=1)
            self.pivoted_data[var] = data

        print("nans after imputation:")
        for var in self.variables:
            print(f"{var}: {self.pivoted_data[var].isna().sum().sum()}")

    def reshape_to_3d_array(self):
        arrays = []
        for var in self.variables:
            arrays.append(self.pivoted_data[var].values)
        X = np.stack(
            arrays, axis=-1
        )  # (number_of_patients, sequence_length, number_of_variables)
        print("Data reshaped:")
        print(X.shape)
        return X

    def normalize(self):
        X = self.X.copy()
        num_vars = X.shape[-1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        for i in range(num_vars):
            X[:, :, i] = scaler.fit_transform(X[:, :, i])
        self.X = X

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
