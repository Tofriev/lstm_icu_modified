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
        #print(self.data_process)
        self.variables = variables
        self.parameters = parameters
        self.scaler = scaler
        self.aggregation_freq = self.parameters.get("aggregation_frequency", "H")
        self.make_feature_lists()
        self.imputation = self.parameters["imputation"]

    def print_missing_stats(self, df):
        # Determine the time column name based on data_type
        time_col = (
            "charttime"
            if self.data_type == "mimic"
            else "measurement_time_from_admission"
        )
        # Use all columns except 'stay_id' and the time column
        feature_cols = [col for col in df.columns if col in self.ALL_FEATURES]

        # Compute missing counts per row (i.e. per sample)
        missing_per_row = df[feature_cols].isnull().sum(axis=1)
        avg_missing_per_row = missing_per_row.mean()
        avg_missing_per_100 = avg_missing_per_row * 100
        # print(
        #     f"Average missing values per 100 samples (rows): {avg_missing_per_100:.2f}"
        # )

        # Compute missingness per sequence (group by stay_id)
        def seq_missing_info(group):
            total_cells = group.shape[0] * len(feature_cols)
            missing_cells = group[feature_cols].isnull().sum().sum()
            # You can report absolute missing or percentage:
            pct_missing = (missing_cells / total_cells) * 100
            return missing_cells, pct_missing

        seq_info = df.groupby("stay_id").apply(seq_missing_info)
        # Separate the information
        missing_counts = [info[0] for info in seq_info]
        missing_pcts = [info[1] for info in seq_info]
        avg_missing_count_seq = np.mean(missing_counts)
        avg_missing_pct_seq = np.mean(missing_pcts)
        # print(
        #     f"Average missing values per sequence (absolute count): {avg_missing_count_seq:.2f}"
        # )
        # print(f"Average missing percentage per sequence: {avg_missing_pct_seq:.2f}%")

    def process(self):

        if self.data_type == "mimic":
            print("Processing MIMIC data...")
            self.process_mimic()
            print("Processing MIMIC done...")
        elif self.data_type == "tudd":
            print("Processing TUDD data...")
            self.process_tudd()
            print("Processing TUDD done...")

        # if "tudd" in self.data:
        #     print("Processing TUDD data...")
        #     self.process_tudd()
        #     sequences_dict["tudd"] = self.sequence_dict["tudd"]
        #     self.feature_index_mapping = {
        #         index: feature for index, feature in enumerate(self.ALL_FEATURES)
        #     }

        # if (
        #     self.parameters.get("fractional_steps")
        #     and self.parameters["dataset_type"] == "mimic_tudd_fract"
        # ):
        #     self.generate_fractions()

        # if self.compare_distributions:
        #     if (
        #         "tudd" in self.parameters["dataset_type"]
        #         and "mimic" in self.parameters["dataset_type"]
        #     ):
        #         mimic_df = self.plot_dict["mimic"]
        #         tudd_df = self.plot_dict["tudd"]
        #         mimic_imputed_df = self.plot_dict["mimic_imputed"]
        #         tudd_imputed_df = self.plot_dict["tudd_imputed"]
        #         print("Distributions before imputation:")
        #         self.plot_density(mimic_df, tudd_df, self.MIMIC_NUMERICAL_FEATURES)
        #         print("Distributions after imputation:")
        #         self.plot_density(
        #             mimic_imputed_df, tudd_imputed_df, self.MIMIC_NUMERICAL_FEATURES
        #         )

    def process_mimic(self):
        self.variable_conversion_and_aggregation()
        self.create_time_grid()
        self.merge_on_time_grid()
       # print("MIMIC missing values statistics:")
        # self.print_missing_stats(self.data_process["merged"])
        self.impute()
        # self.scale_normalize()
        # self.create_sequences()
        # self.split_train_test_sequences()
        self.scale_split_and_create_sequences()
    def variable_conversion_and_aggregation(self):
        """
        converts vars in the static data and aggregate all data on specified time frequency
        """
        #print("aggregating...")
        if not "aggregated" in self.data_process:
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
                # print(self.data_process['aggregated']['static_data'].head(40))

            else:
                #print(self.data_process["pre_processing"][variable])
                df = self.data_process["pre_processing"][variable]
                df["charttime"] = pd.to_datetime(df["charttime"]).dt.floor(
                    self.aggregation_freq
                )
                measurement_cols = df.columns.difference(["stay_id", "charttime"])
                df_agg = df.groupby(["stay_id", "charttime"], as_index=False)[
                    measurement_cols.tolist()
                ].mean()
                self.data_process["aggregated"][variable] = df_agg
                # print(self.data_process['aggregated'][variable].head(40))

    def create_time_grid(self):
        """
        This function creates a time grid, afterwards we have a df with two colummns:
        stay_id and charttime with all the time points for the first 24hs for each stay_id
        """
        #print("creating time grid...")
        static_df = self.data_process["aggregated"]["static_data"]
        df_list = []
        for _, row in static_df.iterrows():
            stay_id = row["stay_id"]
            start_time = row["intime"]
            end_time = row["first_day_end"] - pd.Timedelta(
                hours=1
            )  # correct for 1 hour to have the first 24 hours in total
            time_range = pd.date_range(
                start=start_time, end=end_time, freq=self.aggregation_freq
            )
            time_df = pd.DataFrame({"stay_id": stay_id, "charttime": time_range})
            df_list.append(time_df)
        self.time_grid = pd.concat(df_list, ignore_index=True)
        # print(self.time_grid.head(40))

    def merge_on_time_grid(self):
        """
        This function merges all datga on the time ghrid
        """
        #print("merging on time grid...")
        merged_df_without_static = self.time_grid.copy()
        # fist merge all non static data on the time grid
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
        # print(f"merged shape: {merged_df_without_static.shape}")
        # now merge the static data except intime and first day end
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

        # print(f"merged: {self.data_process['merged'].head(40)}")

    def make_feature_lists(self):
        """
        Helper function to create lists for numerical, categorical, and sequence variables.
        """
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

        # print(f"Sequence features: {self.SEQUENCE_FEATURES}")
        # print(f"Numerical features: {self.NUMERICAL_FEATURES}")
        # print(f"Categorical features: {self.CAT_FEATURES}")
        # print(f"All features: {self.ALL_FEATURES}")

    def impute(self):
        """
        impute_with_ffill_bfill:
            - imputes sequential features with ffill and bfill
            - imputes remaining missing numerical values with global mean. This is relevant
                for features that have no value at all and hence ffill and bfill is not applicable
            - imputes categorical (only gender at the moment) features with mode
        """
        df = self.data_process["merged"].copy()

        if self.imputation["method"] == "ffill_bfill":
            imputed_df = self.impute_with_ffill_bfill(df)
        # elif self.imputation["method"] == "mean":
        #     imputed_df = self.impute_with_mean(df)
        # elif self.imputation["method"] == "rolling_mean":
        #     imputed_df = self.impute_with_rolling_mean(df)
        # elif self.imputation["method"] == "knn":
        #     imputed_df = self.impute_with_knn(df)

        self.data_process["imputed"] = imputed_df
        # print(f"imputed: {self.data_process['imputed'].head(40)}")

    def impute_with_ffill_bfill(self, df):
        #print("imputing with ffill and bfill...")
        if self.data_type == "mimic":
            df.sort_values(["stay_id", "charttime"], inplace=True)
        elif self.data_type == "tudd":
            df.sort_values(["stay_id", "measurement_time_from_admission"], inplace=True)
        for num_feature in self.SEQUENCE_FEATURES:
            df[num_feature] = df.groupby("stay_id")[num_feature].ffill()
            df[num_feature] = df.groupby("stay_id")[num_feature].bfill()

        if self.parameters.get("sparsity_check"):
            self.count_stayid_with_no_observations(df)
        # impute with global mean
        df[self.NUMERICAL_FEATURES] = df[self.NUMERICAL_FEATURES].fillna(
            df[self.NUMERICAL_FEATURES].mean()
        )
        # impute with mode
        df[self.CAT_FEATURES] = df[self.CAT_FEATURES].fillna(
            df[self.CAT_FEATURES].mode()
        )
        return df

    # def impute_with_mean(self, X):
    #     global_means = X[self.NUMERICAL_FEATURES].mean()
    #     X[self.NUMERICAL_FEATURES] = X[self.NUMERICAL_FEATURES].fillna(global_means)
    #     for cat_feature in self.CAT_FEATURES:
    #         X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)

    # def impute_with_rolling_mean(self, X):
    #     X = X.sort_values(["stay_id", "charttime"])
    #     X[self.NUMERICAL_FEATURES] = X.groupby("stay_id")[
    #         self.NUMERICAL_FEATURES
    #     ].apply(
    #         lambda group: group.fillna(group.rolling(window=3, min_periods=1).mean())
    #     )

    # def impute_with_knn(self, X):
    #     print("Starting KNN imputation...")

    #     features_to_impute = self.NUMERICAL_FEATURES  # + self.CAT_FEATURES

    #     knn_imputer = KNNImputer(n_neighbors=self.imputation["n_neighbors"])

    #     X[features_to_impute] = knn_imputer.fit_transform(X[features_to_impute])

    #     for cat_feature in self.CAT_FEATURES:
    #         X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)
    #     print("KNN imputation done.")

    def count_stayid_with_no_observations(self, df):
        """
        counts the number of stay_id's that have no observations at all for a feature
        """
        #print("checking for stay ids with missing observations for a feature")
        total_stay_ids = df["stay_id"].nunique()
        missing_observations = {}
        features = [col for col in df.columns if col in self.ALL_FEATURES]

        for feature in features:
            missing_stay_ids = df.groupby("stay_id")[feature].apply(
                lambda x: x.isna().all()
            )
            missing_stay_ids = missing_stay_ids[
                missing_stay_ids
            ].index.tolist()  # Get missing stay_id list

            missing_count = len(missing_stay_ids)
            percentage = (missing_count / total_stay_ids) * 100
            missing_observations[feature] = (
                missing_count,
                percentage,
                missing_stay_ids,
            )

        # for feature, (count, percentage, stay_ids) in missing_observations.items():
        #     print(f"{feature}: {count} stay_id(s) with no observations")
        #     print(f"{percentage:.2f}% of total stay_ids")
        #     print(f"First 100 missing stay IDs for {feature}: {stay_ids[:100]}\n")

    def scale_normalize(self):
        df = self.data_process["imputed"].copy()

        if self.parameters["scaling"] == "Standard":
            #print("scaling with StandardScaler...")
            if self.scaler:  # if scaler was given to init, use it
                print("using preassigned scaler")
            else:
                self.scaler = StandardScaler()
                print('make new scaler')
            df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
                df[self.NUMERICAL_FEATURES]
            )

        # elif self.parameters["scaling"] == "MinMax":
        #     #print("scaling with MinMaxScaler...")
        #     if self.scaler:
        #         print("using preasigned scaler")
        #     else:
        #         scaler = MinMaxScaler(
        #             feature_range=(
        #                 self.parameters["scaling_range"][0],
        #                 self.parameters["scaling_range"][1],
        #             )
        #         )
        #     df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
        #         df[self.NUMERICAL_FEATURES]
        #     )
        #     self.MIMIC_NUMERICAL_FEATURES = self.NUMERICAL_FEATURES

        self.data_process["scaled"] = df
        # print(f'scaled: {self.data_process["scaled"].head(40)}')

    def create_sequences(self):
        """
        Create sequences by separating the time-varying (sequential) features and static features.
        Assumes that the scaled data (df) contains columns for:
        - sequential features: defined as f"{var}_value" for each variable at the top level (except "static_data")
        - static features: defined as f"{key}_value" for keys in variables["static_data"] that are marked for training.
        """
        df = self.data_process["scaled"].copy()
        sequences = []
        
        # Sequential features: all top-level variables (except "static_data")
        seq_feature_names = [f"{var}_value" for var in self.variables if var != "static_data"]
        
        # Static features: include only those keys from static_data that are marked for training and exclude meta-keys.
        exclude_static = {"mortality", "intime", "first_day_end", "stay_id"}
        static_feature_names = [
            f"{key}_value" 
            for key, attr in self.variables["static_data"].items() 
            if key not in exclude_static and attr.get("training", False)
        ]
        
        for stay_id, group in df.groupby("stay_id"):
            group = group.sort_values("charttime")
            # Extract static features from the first row.
            static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
            # Extract sequential (time-varying) features.
            sequential_features = group[seq_feature_names].values.astype(np.float32)
            # Extract the label using the target key; note that the target column is expected to be named like f"{target}_value".
            label = group[f'{self.parameters["target"]}_value'].iloc[0]
            sequences.append((sequential_features, static_features, label))
        
        # Save mapping for debugging or later visualization.
        self.feature_index_mapping_sequences = {
            "sequential": {idx: feature for idx, feature in enumerate(seq_feature_names)},
            "static": {idx: feature for idx, feature in enumerate(static_feature_names)}
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
       # print(self.data_process["sequences_train"][0])

    #     stay_ids_to_drop = X.groupby('stay_id')[self.NUMERICAL_FEATURES].apply(
    #         lambda group: group.isnull().all(axis=0).any()
    #     )
    #     stay_ids_to_drop = stay_ids_to_drop[stay_ids_to_drop].index

    #     print(f"Number of MIMIC observations before dropping: {len(X)}")

    #     #DROP
    #    # X = X[~X['stay_id'].isin(stay_ids_to_drop)]
    #     print(f"Number of MIMIC observations after dropping: {len(X)}")
    #     # # Use KNN imputation for features without any values
    #     # features_to_impute = self.NUMERICAL_FEATURES
    #     # print('starting knn imputation as part of ffill_bfill')
    #     # knn_imputer = KNNImputer(n_neighbors=4)
    #     # X[features_to_impute] = knn_imputer.fit_transform(X[features_to_impute])
    #     # print('knn imputation done')
    #     # for cat_feature in self.CAT_FEATURES:
    #     #     X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)

    # TODO: put this in a separate class or one of the others
    # def generate_fractions(self):
    #     print("Generating fractional datasets...")
    #     mimic_train = self.sequence_dict["mimic"]["train"]
    #     mimic_test = self.sequence_dict["mimic"]["test"]
    #     tudd_train = self.sequence_dict["tudd"]["train"]
    #     tudd_test = self.sequence_dict["tudd"]["test"]

    #     n_tudd_train = len(tudd_train) - 2000
    #     step_size = self.parameters["fractional_steps"]
    #     fractional_datasets = {}
    #     n_sampled_tudd_train = 0

    #     while n_sampled_tudd_train + step_size < n_tudd_train:
    #         n_sampled_tudd_train += step_size

    #         # getg next tudd batch
    #         tudd_samples = tudd_train[:n_sampled_tudd_train]
    #         combined_train_set = mimic_train + tudd_samples
    #         if self.shuffle == True:
    #             random.shuffle(combined_train_set)

    #         fractional_datasets[n_sampled_tudd_train] = combined_train_set

    #         print(f"fraction {n_sampled_tudd_train} added")

    #     self.sequence_dict["fractional_mimic_tudd"] = fractional_datasets

    def scale_split_and_create_sequences(self):
        """
        Splits the imputed MIMIC data by unique stay_id, fits a scaler on the training set,
        transforms the test set, and then creates sequences for each set.
        """
        df = self.data_process["imputed"].copy()
        target_col = f'{self.parameters["target"]}_value'
        
        labels_by_id = df.groupby("stay_id")[target_col].first()
        train_ids, test_ids = train_test_split(
            labels_by_id.index,
            test_size=0.2,
            stratify=labels_by_id.values,
            random_state=42
        )
        
        train_df = df[df["stay_id"].isin(train_ids)].copy()
        test_df = df[df["stay_id"].isin(test_ids)].copy()
        
        if self.parameters["scaling"] == "Standard":
            print("Scaling training and test sets (MIMIC) using StandardScaler...")
            if self.scaler:  # use a preassigned scaler if available
                print('use preassigned scaler')
                scaler = self.scaler
                train_df[self.NUMERICAL_FEATURES] = scaler.transform(train_df[self.NUMERICAL_FEATURES])
                test_df[self.NUMERICAL_FEATURES] = scaler.transform(test_df[self.NUMERICAL_FEATURES])
            else:
                print('make new scaler')
                scaler = StandardScaler()
                self.scaler = scaler
                train_df[self.NUMERICAL_FEATURES] = scaler.fit_transform(train_df[self.NUMERICAL_FEATURES])
                test_df[self.NUMERICAL_FEATURES] = scaler.transform(test_df[self.NUMERICAL_FEATURES])
        
        self.data_process["scaled_train"] = train_df
        self.data_process["scaled_test"] = test_df

        sequences_train = []
        sequences_test = []
        
        seq_feature_names = [f"{var}_value" for var in self.variables if var != "static_data"]
        exclude_static = {"mortality", "intime", "first_day_end", "stay_id"}
        static_feature_names = [
            f"{key}_value"
            for key, attr in self.variables["static_data"].items()
            if key not in exclude_static and attr.get("training", False)
        ]
        
        for stay_id, group in train_df.groupby("stay_id"):
            group = group.sort_values("charttime")
            static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
            sequential_features = group[seq_feature_names].values.astype(np.float32)
            label = group[target_col].iloc[0]
            sequences_train.append((sequential_features, static_features, label))
        
        for stay_id, group in test_df.groupby("stay_id"):
            group = group.sort_values("charttime")
            static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
            sequential_features = group[seq_feature_names].values.astype(np.float32)
            label = group[target_col].iloc[0]
            sequences_test.append((sequential_features, static_features, label))
        
        self.feature_index_mapping_sequences = {
            "sequential": {idx: feature for idx, feature in enumerate(seq_feature_names)},
            "static": {idx: feature for idx, feature in enumerate(static_feature_names)}
        }
        self.data_process["sequences_train"] = sequences_train
        self.data_process["sequences_test"] = sequences_test

    # TODO: put this in a separate class or one of the others
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
            # print(f"Mean {feature} MIMIC: {mimic_df[feature].mean()}")
            # print(f"Mean {feature} TUDD: {tudd_df[feature].mean()}")

    def process_tudd(self):
        measurements = self.data_process["pre_processing"]["measurements"].copy()
        # print(f"measurements00: {measurements.head(40)}")
        mortality_info = self.data_process["pre_processing"]["mortality_info"].copy()

        # calculate ICU stay duration and measurement offset in hours
        # reason for adding one to the stay duaration: duiration is that stay_duration in onkly given as full days in real numbers. If stay dration is 0 there might be up to 23 hours of data and so on
        mortality_info = mortality_info[mortality_info["stay_duration"] != 0]

        # mortality_info["stay_duration_hours"] = (mortality_info["stay_duration"]) * 24
        # in tudd_incomplete.csv the measurement_offset is already in hours
        # measurements["measurement_offset_hours"] = (
        #     measurements["measurement_offset"] * 24
        # )

        # print(f"measurements1: {measurements.head(40)}")
        measurements = pd.merge(
            measurements,
            mortality_info[["caseid"]],
            on="caseid",
            how="inner",
        )
        # print(f"measurements2: {measurements.head(40)}")
        # calculate measurement time from admission
        # assuming measurement offset -0 is at discharge
        measurements["measurement_offset"] = (
            measurements["measurement_offset"].str.replace(",", ".").astype(float)
        )
        measurements["value"] = (
            measurements["value"]
            .str.replace(",", ".")
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float)
        )

        # measurements["measurement_time_from_admission"] = measurements[
        #     "measurement_offset"
        # ].abs()
        measurements["min_offset"] = measurements.groupby("caseid")[
            "measurement_offset"
        ].transform("min")
        measurements["measurement_time_from_admission"] = (
            measurements["measurement_offset"] - measurements["min_offset"]
        )

        # measurements["measurement_offset"] = pd.to_numeric(
        #     measurements["measurement_offset"], errors="coerce"
        # )

        measurements.rename(columns={"caseid": "stay_id"}, inplace=True)
        mortality_info.rename(columns={"caseid": "stay_id"}, inplace=True)

        # clean negative vals
        measurements = measurements[
            measurements["measurement_time_from_admission"] > -1
        ]
        # print(f"measurements2: {measurements.head(40)}")
        # a little fuzziness is accaptable a the time borders
        measurements.loc[
            measurements["measurement_time_from_admission"] <= -1,
            "measurement_time_from_admission",
        ] = 0
        # print(f"measurements3: {measurements.head(40)}")
        # filter first 24 h
        measurements = measurements[
            (measurements["measurement_time_from_admission"] >= 0)
            & (measurements["measurement_time_from_admission"] <= 24)
        ]
        # print(f"measurements4: {measurements.head(40)}")
        measurements["measurement_time_from_admission"] = np.floor(
            measurements["measurement_time_from_admission"]
        )

        # aggregate
        measurements["value"] = pd.to_numeric(measurements["value"], errors="coerce")
        measurements_agg = (
            measurements.groupby(
                ["stay_id", "measurement_time_from_admission", "treatmentname"]
            )["value"]
            .mean()
            .reset_index()
        )
        # print(f"measurements: {measurements.head(40)}")
        # print(f"measurements_agg: {measurements_agg.head(40)}")

        # pivot
        measurements_pivot = measurements_agg.pivot_table(
            index=["stay_id", "measurement_time_from_admission"],
            columns="treatmentname",
            values="value",
        ).reset_index()
        # (f"measurements_pivot: {measurements_pivot.head(40)}")

        # make time grid
        def create_time_grid(mortality_info):
            df_list = []
            for _, row in mortality_info.iterrows():
                stay_id = row["stay_id"]
                time_range = np.arange(0, 24)  # 24 hour grid
                time_df = pd.DataFrame(
                    {"stay_id": stay_id, "measurement_time_from_admission": time_range}
                )
                df_list.append(time_df)
            return pd.concat(df_list, ignore_index=True)

        time_grid = create_time_grid(mortality_info)
        time_grid.to_csv("time_grid.csv", index=False)

        # merge on time grid
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
        # print("TUDD missing values statistics:")
        # self.print_missing_stats(merged_df)

        # # print unique treatment names before mapping
        # print("Unique treatment names before mapping:")
        # print(merged_df.columns.unique())

        # rename
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

        # print unique treatment names after mapping
        # print("Unique treatment names after mapping:")
        # print(merged_df.columns.unique())

        # print(
        #     f'number of unique stay_ids before renaming and bounding: {merged_df["stay_id"].nunique()}'
        # )
        # bounds
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
        # print(
        #     f'number of unique stay_ids after bounding: {merged_df["stay_id"].nunique()}'
        # )
        # if self.parameters['small_data']:
        #     fraction = 0.1
        #     patient_sample = merged_df[['stay_id', 'exitus']].drop_duplicates().groupby('exitus', group_keys=False).apply(
        #         lambda x: x.sample(frac=fraction, random_state=42)
        #     )
        #     sampled_df = merged_df[merged_df['stay_id'].isin(patient_sample['stay_id'])]
        #     merged_df = sampled_df

        # mean before conversion
        # print("Mean before conversion:")
        # print(f"Glucose (mmol/L): {merged_df['glc_value'].mean()}")
        # print(f"Creatinine (micro_mol/L): {merged_df['creatinine_value'].mean()}")
        # print(f"INR (Quick): {merged_df['inr_value'].mean()}")
        # print(f"Lactate (mmol/L): {merged_df['lactate_value'].mean()}")

        # convert units
        # glucose mmol/L to mg/dL
        merged_df["glc_value"] = merged_df["glc_value"] * 18.0182
        # print(f"Glucose conversion done: {merged_df['glc_value'].mean()} mg/dL")

        # creatinine micro_mol/L to mg/dL
        merged_df["creatinine_value"] = merged_df["creatinine_value"] * 0.0113
        # print(
        #     f"Creatinine conversion done: {merged_df['creatinine_value'].mean()} mg/dL"
        # )

        # convert quick to inr
        merged_df["inr_value"] = merged_df["inr_value"] / 100
        # print(f"INR conversion done: {merged_df['inr_value'].mean()}")

        # convert lactate mmol/L to mg/dL
        # merged_df['lactate_value'] = merged_df['lactate_value'] * 9.01
        # print(f"Lactate conversion done: {merged_df['lactate_value'].mean()} mg/dL")
        # print(
        #     f'number of unique stay_ids before filtering: {merged_df["stay_id"].nunique()}'
        # )
        # filter age
        merged_df = merged_df[merged_df["age_value"] >= 18]
        merged_df["age_value"] = merged_df["age_value"].apply(lambda x: min(x, 90))

        for feature, (lower, upper) in bounds.items():
            if feature in merged_df.columns:
                merged_df[feature] = pd.to_numeric(merged_df[feature], errors="coerce")
                merged_df.loc[merged_df[feature] < lower, feature] = np.nan
                merged_df.loc[merged_df[feature] > upper, feature] = np.nan

        # print(
        #     f'number of unique stay_ids before iumputing: {merged_df["stay_id"].nunique()}'
        # )

        # imputation
        if self.imputation["method"] == "ffill_bfill":
            merged_df = self.impute_with_ffill_bfill(merged_df)
        merged_df["exitus"].fillna(0, inplace=True)
        self.data_process["imputed"] = merged_df

        # Split by stay_id
        labels_by_id = merged_df.groupby("stay_id")["exitus"].first()
        train_ids, test_ids = train_test_split(
            labels_by_id.index, 
            test_size=0.2, 
            stratify=labels_by_id.values, 
            random_state=42
        )

        train_df = merged_df[merged_df["stay_id"].isin(train_ids)].copy()
        test_df = merged_df[merged_df["stay_id"].isin(test_ids)].copy()



        if self.parameters["scaling"] == "Standard":
            print("scaling....")
            if hasattr(self, "scaler") and self.scaler is not None:
                print('using preassigned scaler')
                scaler = self.scaler
                train_df[self.NUMERICAL_FEATURES] = scaler.transform(train_df[self.NUMERICAL_FEATURES]) # train not really needed in this case but we do it for sake of completeness
                test_df[self.NUMERICAL_FEATURES] = scaler.transform(test_df[self.NUMERICAL_FEATURES])
            else:
                print("Scaling on training data only...")
                tudd_scaler = StandardScaler()
                train_df[self.NUMERICAL_FEATURES] = tudd_scaler.fit_transform(train_df[self.NUMERICAL_FEATURES])
                test_df[self.NUMERICAL_FEATURES] = tudd_scaler.transform(test_df[self.NUMERICAL_FEATURES])
                self.scaler = tudd_scaler

        sequences_train = []
        sequences_test = []

        seq_feature_names = [f"{var}_value" for var in self.variables if var != "static_data"]
        exclude_static = {"mortality", "intime", "first_day_end", "stay_id"}
        static_feature_names = [
            f"{key}_value" 
            for key, attr in self.variables["static_data"].items() 
            if key not in exclude_static and attr.get("training", False)
        ]

        for stay_id, group in train_df.groupby("stay_id"):
            group = group.sort_values("measurement_time_from_admission")
            static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
            sequential_features = group[seq_feature_names].values.astype(np.float32)
            label = group["exitus"].iloc[0]
            sequences_train.append((sequential_features, static_features, label))

        for stay_id, group in test_df.groupby("stay_id"):
            group = group.sort_values("measurement_time_from_admission")
            static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
            sequential_features = group[seq_feature_names].values.astype(np.float32)
            label = group["exitus"].iloc[0]
            sequences_test.append((sequential_features, static_features, label))

        self.feature_index_mapping_sequences = {
            "sequential": {idx: feature for idx, feature in enumerate(seq_feature_names)},
            "static": {idx: feature for idx, feature in enumerate(static_feature_names)}
        }


        self.data_process["sequences_train"] = sequences_train
        self.data_process["sequences_test"] = sequences_test