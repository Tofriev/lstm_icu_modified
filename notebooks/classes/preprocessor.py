import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from utils import set_seed
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import pickle

set_seed(42)


class Preprocessor:
    def __init__(
        self,
        data_type: str,
        data,
        variables,
        parameters,
        scaler=None,
        ALL_FEATURES_MIMIC=None,
    ):
        self.data_type = data_type
        self.data_process = {
            # records the data processing steps:
            # aggregated, merged, imputed, scaled, sequences,
            # sequences_train, sequences_test
            "pre_processing": data
        }
        self.variables = variables
        self.parameters = parameters
        self.scaler = scaler
        self.aggregation_freq = self.parameters.get("aggregation_frequency", "H")
        self.make_feature_lists()
        self.imputation = self.parameters["imputation"]

    def print_missing_stats(self, df):
        time_col = (
            "charttime"
            if self.data_type == "mimic"
            else "measurement_time_from_admission"
        )
        feature_cols = [col for col in df.columns if col in self.ALL_FEATURES]
        missing_per_row = df[feature_cols].isnull().sum(axis=1)
        avg_missing_per_row = missing_per_row.mean()
        avg_missing_per_100 = avg_missing_per_row * 100
        print(
            f"Average missing values per 100 samples (rows): {avg_missing_per_100:.2f}"
        )

        def seq_missing_info(group):
            total_cells = group.shape[0] * len(feature_cols)
            missing_cells = group[feature_cols].isnull().sum().sum()
            pct_missing = (missing_cells / total_cells) * 100
            return missing_cells, pct_missing

        seq_info = df.groupby("stay_id").apply(seq_missing_info)
        missing_counts = [info[0] for info in seq_info]
        missing_pcts = [info[1] for info in seq_info]
        avg_missing_count_seq = np.mean(missing_counts)
        avg_missing_pct_seq = np.mean(missing_pcts)
        print(
            f"Average missing values per sequence (absolute count): {avg_missing_count_seq:.2f}"
        )
        print(f"Average missing percentage per sequence: {avg_missing_pct_seq:.2f}%")

    def process(self):
        if self.data_type == "mimic":
            print("Processing MIMIC data...")
            self.process_mimic()
            print("Processing MIMIC done...")
        elif self.data_type == "tudd":
            print("Processing TUDD data...")
            self.process_tudd()
            print("Processing TUDD done...")

    def process_mimic(self):
        self.variable_conversion_and_aggregation()
        self.create_time_grid()
        self.merge_on_time_grid()
        print("MIMIC missing values statistics:")
        self.print_missing_stats(self.data_process["merged"])
        self.impute()
        self.scale_normalize()
        self.create_sequences()
        self.split_train_test_sequences()

    def variable_conversion_and_aggregation(self):
        print("aggregating...")
        if "aggregated" not in self.data_process:
            self.data_process["aggregated"] = {}
        else:
            raise ValueError(
                "Aggregated data already exists. Exiting to prevent overwriting."
            )
        for variable in self.variables.keys():
            if variable == "static_data":
                static_df = self.data_process["pre_processing"]["static_data"]
                static_df["intime"] = pd.to_datetime(static_df["intime"]).dt.floor(
                    self.aggregation_freq
                )
                static_df["first_day_end"] = pd.to_datetime(
                    static_df["first_day_end"]
                ).dt.floor(self.aggregation_freq)
                static_df["gender_value"] = static_df["gender_value"].map(
                    {"M": 0, "F": 1}
                )
                self.data_process["aggregated"]["static_data"] = static_df
            else:
                df = self.data_process["pre_processing"][variable]
                df["charttime"] = pd.to_datetime(df["charttime"]).dt.floor(
                    self.aggregation_freq
                )
                measurement_cols = df.columns.difference(["stay_id", "charttime"])
                df_agg = df.groupby(["stay_id", "charttime"], as_index=False)[
                    measurement_cols.tolist()
                ].mean()
                self.data_process["aggregated"][variable] = df_agg

    def create_time_grid(self):
        print("creating time grid...")
        static_df = self.data_process["aggregated"]["static_data"]
        df_list = []
        for _, row in static_df.iterrows():
            stay_id = row["stay_id"]
            start_time = row["intime"]
            end_time = row["first_day_end"] - pd.Timedelta(hours=1)
            time_range = pd.date_range(
                start=start_time, end=end_time, freq=self.aggregation_freq
            )
            time_df = pd.DataFrame({"stay_id": stay_id, "charttime": time_range})
            df_list.append(time_df)
        self.time_grid = pd.concat(df_list, ignore_index=True)

    def merge_on_time_grid(self):
        print("merging on time grid...")
        merged_df_without_static = self.time_grid.copy()
        for variable in self.data_process["aggregated"].keys():
            if variable == "static_data":
                continue
            else:
                merged_df_without_static = pd.merge(
                    merged_df_without_static,
                    self.data_process["aggregated"][variable],
                    on=["stay_id", "charttime"],
                    how="left",
                )
        static_columns = [
            col
            for col in self.data_process["aggregated"]["static_data"].columns
            if col not in ["intime", "first_day_end"]
        ]
        merged_df_with_static = pd.merge(
            merged_df_without_static,
            self.data_process["aggregated"]["static_data"][static_columns],
            on="stay_id",
            how="left",
        )
        self.data_process["merged"] = merged_df_with_static

    def make_feature_lists(self):
        static_data = self.variables["static_data"]
        self.SEQUENCE_FEATURES = [
            f"{var}_value" for var in self.variables if var != "static_data"
        ]
        self.NUMERICAL_FEATURES = self.SEQUENCE_FEATURES + [
            f"{var}_value"
            for var, attr in static_data.items()
            if attr["type"] == "numerical"
        ]
        self.CAT_FEATURES = [
            f"{var}_value"
            for var, attr in static_data.items()
            if attr["type"] == "categorical"
        ]
        self.ALL_FEATURES = self.NUMERICAL_FEATURES + self.CAT_FEATURES

    def impute(self):
        df = self.data_process["merged"].copy()
        if self.imputation["method"] == "ffill_bfill":
            imputed_df = self.impute_with_ffill_bfill(df)
        df[self.NUMERICAL_FEATURES] = df[self.NUMERICAL_FEATURES].fillna(
            df[self.NUMERICAL_FEATURES].mean()
        )
        df[self.CAT_FEATURES] = df[self.CAT_FEATURES].fillna(
            df[self.CAT_FEATURES].mode()
        )
        self.data_process["imputed"] = imputed_df

    def impute_with_ffill_bfill(self, df):
        print("imputing with ffill and bfill...")
        if self.data_type == "mimic":
            df.sort_values(["stay_id", "charttime"], inplace=True)
        elif self.data_type == "tudd":
            df.sort_values(["stay_id", "measurement_time_from_admission"], inplace=True)
        for num_feature in self.SEQUENCE_FEATURES:
            df[num_feature] = df.groupby("stay_id")[num_feature].ffill()
            df[num_feature] = df.groupby("stay_id")[num_feature].bfill()
        df[self.NUMERICAL_FEATURES] = df[self.NUMERICAL_FEATURES].fillna(
            df[self.NUMERICAL_FEATURES].mean()
        )
        df[self.CAT_FEATURES] = df[self.CAT_FEATURES].fillna(
            df[self.CAT_FEATURES].mode()
        )
        return df

    def scale_normalize(self):
        df = self.data_process["imputed"].copy()
        if self.parameters["scaling"] == "Standard":
            print("scaling with StandardScaler...")
            if self.scaler:
                print("using preassigned scaler")
            else:
                self.scaler = StandardScaler()
            df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
                df[self.NUMERICAL_FEATURES]
            )
        elif self.parameters["scaling"] == "MinMax":
            print("scaling with MinMaxScaler...")
            if self.scaler:
                print("using preassigned scaler")
            else:
                self.scaler = MinMaxScaler(
                    feature_range=(
                        self.parameters["scaling_range"][0],
                        self.parameters["scaling_range"][1],
                    )
                )
            df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
                df[self.NUMERICAL_FEATURES]
            )
        self.data_process["scaled"] = df

    def create_sequences(self):
        df = self.data_process["scaled"].copy()
        sequences = []
        for stay_id, group in df.groupby("stay_id"):
            group = group.sort_values("charttime")
            features = group[self.ALL_FEATURES].values
            label = group[f'{self.parameters["target"]}_value'].iloc[0]
            sequences.append((features, label))
        self.feature_index_mapping_sequences = {
            index: feature for index, feature in enumerate(self.ALL_FEATURES)
        }
        self.data_process["sequences"] = sequences

    def split_train_test_sequences(self):
        df = self.data_process["sequences"].copy()
        labels = [seq[1] for seq in df]
        sequence_dict = {}
        sequence_dict["train"], sequence_dict["test"] = train_test_split(
            df, test_size=0.2, stratify=labels, random_state=42
        )
        self.data_process["sequences_train"] = sequence_dict["train"]
        self.data_process["sequences_test"] = sequence_dict["test"]
        print("Example training sequence:", self.data_process["sequences_train"][0])

    def process_tudd(self):
        measurements = self.data_process["pre_processing"]["measurements"].copy()
        print(f"measurements00: {measurements.head(40)}")
        mortality_info = self.data_process["pre_processing"]["mortality_info"].copy()
        mortality_info = mortality_info[mortality_info["stay_duration"] != 0]
        print(f"measurements1: {measurements.head(40)}")
        measurements = pd.merge(
            measurements, mortality_info[["caseid"]], on="caseid", how="inner"
        )
        print(f"measurements2: {measurements.head(40)}")
        measurements["measurement_offset"] = (
            measurements["measurement_offset"].str.replace(",", ".").astype(float)
        )
        measurements["value"] = (
            measurements["value"]
            .str.replace(",", ".")
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float)
        )
        measurements["min_offset"] = measurements.groupby("caseid")[
            "measurement_offset"
        ].transform("min")
        measurements["measurement_time_from_admission"] = (
            measurements["measurement_offset"] - measurements["min_offset"]
        )
        measurements.rename(columns={"caseid": "stay_id"}, inplace=True)
        mortality_info.rename(columns={"caseid": "stay_id"}, inplace=True)
        measurements = measurements[
            measurements["measurement_time_from_admission"] > -1
        ]
        print(f"measurements2: {measurements.head(40)}")
        measurements.loc[
            measurements["measurement_time_from_admission"] <= -1,
            "measurement_time_from_admission",
        ] = 0
        print(f"measurements3: {measurements.head(40)}")
        measurements = measurements[
            (measurements["measurement_time_from_admission"] >= 0)
            & (measurements["measurement_time_from_admission"] <= 24)
        ]
        print(f"measurements4: {measurements.head(40)}")
        measurements["measurement_time_from_admission"] = np.floor(
            measurements["measurement_time_from_admission"]
        )
        measurements["value"] = pd.to_numeric(measurements["value"], errors="coerce")
        measurements_agg = (
            measurements.groupby(
                ["stay_id", "measurement_time_from_admission", "treatmentname"]
            )["value"]
            .mean()
            .reset_index()
        )
        print(f"measurements: {measurements.head(40)}")
        print(f"measurements_agg: {measurements_agg.head(40)}")
        measurements_pivot = measurements_agg.pivot_table(
            index=["stay_id", "measurement_time_from_admission"],
            columns="treatmentname",
            values="value",
        ).reset_index()
        print(f"measurements_pivot: {measurements_pivot.head(40)}")

        def create_time_grid(mortality_info):
            df_list = []
            for _, row in mortality_info.iterrows():
                stay_id = row["stay_id"]
                time_range = np.arange(0, 24)
                time_df = pd.DataFrame(
                    {"stay_id": stay_id, "measurement_time_from_admission": time_range}
                )
                df_list.append(time_df)
            return pd.concat(df_list, ignore_index=True)

        time_grid = create_time_grid(mortality_info)
        time_grid.to_csv("time_grid.csv", index=False)
        merged_df = pd.merge(
            time_grid,
            measurements_pivot,
            on=["stay_id", "measurement_time_from_admission"],
            how="left",
        )
        merged_df = pd.merge(
            merged_df,
            mortality_info[
                [
                    "stay_id",
                    "age_value",
                    "gender_value",
                    "bodyweight",
                    "bodyheight",
                    "exitus",
                ]
            ],
            on="stay_id",
            how="inner",
        )
        print("TUDD missing values statistics:")
        self.print_missing_stats(merged_df)
        print("Unique treatment names before mapping:")
        print(merged_df.columns.unique())
        treatmentnames_mapping = {
            "HF": "hr_value",
            "AGAP": "anion_gap_value",
            "GLUC": "glc_value",
            "CREA": "creatinine_value",
            "K": "potassium_value",
            "LEU": "wbc_value",
            "THR": "platelets_value",
            "Q": "inr_value",
            "LAC": "lactate_value",
            "T": "temperature_value",
            "GCS": "gcs_total_value",
            "MAP": "mbp_value",
            "bodyweight": "weight_value",
            "bodyheight": "height_value",
        }
        merged_df.rename(columns=treatmentnames_mapping, inplace=True)
        print("Unique treatment names after mapping:")
        print(merged_df.columns.unique())
        print(
            f'number of unique stay_ids before renaming and bounding: {merged_df["stay_id"].nunique()}'
        )
        bounds = {
            "age": (18, 90),
            "weight_value": (20, 500),
            "height_value": (20, 260),
            "temperature_value": (20, 45),
            "hr_value": (10, 300),
            "glc_value": (5, 2000),
            "mbp_value": (20, 400),
            "potassium_value": (2.5, 7),
            "wbc_value": (1, 200),
            "platelets_value": (10, 1000),
            "inr_value": (0.2, 6),
            "anion_gap_value": (1, 25),
            "lactate_value": (0.1, 200),
            "creatinine_value": (0.1, 20),
        }
        print(
            f'number of unique stay_ids before bounding: {merged_df["stay_id"].nunique()}'
        )
        print("Mean before conversion:")
        print(f"Glucose (mmol/L): {merged_df['glc_value'].mean()}")
        print(f"Creatinine (micro_mol/L): {merged_df['creatinine_value'].mean()}")
        print(f"INR (Quick): {merged_df['inr_value'].mean()}")
        print(f"Lactate (mmol/L): {merged_df['lactate_value'].mean()}")
        merged_df["glc_value"] = merged_df["glc_value"] * 18.0182
        print(f"Glucose conversion done: {merged_df['glc_value'].mean()} mg/dL")
        merged_df["creatinine_value"] = merged_df["creatinine_value"] * 0.0113
        print(
            f"Creatinine conversion done: {merged_df['creatinine_value'].mean()} mg/dL"
        )
        merged_df["inr_value"] = merged_df["inr_value"] / 100
        print(f"INR conversion done: {merged_df['inr_value'].mean()}")
        print(
            f'number of unique stay_ids before filtering: {merged_df["stay_id"].nunique()}'
        )
        merged_df = merged_df[merged_df["age_value"] >= 18]
        merged_df["age_value"] = merged_df["age_value"].apply(lambda x: min(x, 90))
        for feature, (lower, upper) in bounds.items():
            if feature in merged_df.columns:
                merged_df[feature] = pd.to_numeric(merged_df[feature], errors="coerce")
                merged_df.loc[merged_df[feature] < lower, feature] = np.nan
                merged_df.loc[merged_df[feature] > upper, feature] = np.nan
        print(
            f'number of unique stay_ids before iumputing: {merged_df["stay_id"].nunique()}'
        )
        if self.imputation["method"] == "ffill_bfill":
            merged_df = self.impute_with_ffill_bfill(merged_df)
        merged_df["exitus"].fillna(0, inplace=True)
        if self.parameters["scaling"] == "Standard":
            print("scaling....")
            if hasattr(self, "scaler") and self.scaler is not None:
                scaler = self.scaler
                merged_df[self.NUMERICAL_FEATURES] = scaler.transform(
                    merged_df[self.NUMERICAL_FEATURES]
                )
            else:
                print("using tudd scaler")
                scaler = StandardScaler()
                merged_df[self.NUMERICAL_FEATURES] = scaler.fit_transform(
                    merged_df[self.NUMERICAL_FEATURES]
                )
        print(f"exitus: {merged_df.head(40)}")
        unique_stays = merged_df.groupby("stay_id").first()
        exitus_count = unique_stays[unique_stays["exitus"] == 1].shape[0]
        print(f"Count of exitus == 1: {exitus_count}")
        sequences = []
        for stay_id, group in merged_df.groupby("stay_id"):
            features = group[self.ALL_FEATURES].values
            label = group["exitus"].iloc[0]
            sequences.append((features, label))
        self.feature_index_mapping_sequences = {
            index: feature for index, feature in enumerate(self.ALL_FEATURES)
        }
        labels = [seq[1] for seq in sequences]
        print(f"Number of 1s in labels: {labels.count(1)}")
        self.data_process["sequences_train"], self.data_process["sequences_test"] = (
            train_test_split(
                sequences,
                test_size=0.2,
                stratify=labels,
                random_state=42,
            )
        )

    # NEW: Save sequences to disk for streaming
    def save_sequences_to_disk(self, train_file: str, test_file: str):
        # Save training sequences record-by-record.
        with open(train_file, "wb") as f_train:
            for seq in self.data_process["sequences_train"]:
                pickle.dump(seq, f_train)
        print(f"Training sequences saved to {train_file}")
        # Save test sequences record-by-record.
        with open(test_file, "wb") as f_test:
            for seq in self.data_process["sequences_test"]:
                pickle.dump(seq, f_test)
        print(f"Test sequences saved to {test_file}")

    def plot_density(self, mimic_df, tudd_df, features):
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(mimic_df[feature].dropna(), label="MIMIC", fill=True, alpha=0.5)
            sns.kdeplot(tudd_df[feature].dropna(), label="TUDD", fill=True, alpha=0.5)
            plt.title(f"Density Plot for {feature}")
            plt.xlabel(feature)
            plt.ylabel("Density")
            plt.legend()
            plt.show()
            print(f"Mean {feature} MIMIC: {mimic_df[feature].mean()}")
            print(f"Mean {feature} TUDD: {tudd_df[feature].mean()}")
