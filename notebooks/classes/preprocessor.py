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
        self.variables = variables
        self.parameters = parameters
        self.scaler = scaler
        self.aggregation_freq = self.parameters.get("aggregation_frequency", "H")
        self.make_feature_lists()
        self.imputation = self.parameters["imputation"]

    def process(self):

        if self.data_type == "mimic":
            print("Processing MIMIC data...")
            self.process_mimic()
            print("Processing MIMIC done...")

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
        self.impute()
        self.scale_normalize()
        self.create_sequences()
        self.split_train_test_sequences()

    def variable_conversion_and_aggregation(self):
        """
        converts vars in the static data and aggregate all data on specified time frequency
        """
        print("aggregating...")
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
        print("creating time grid...")
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
        print("merging on time grid...")
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
        print("imputing with ffill and bfill...")
        df.sort_values(["stay_id", "charttime"], inplace=True)
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
        print("checking for stay ids with missing observations for a feature")
        total_stay_ids = df["stay_id"].nunique()
        missing_observations = {}
        features = [col for col in df.columns if col not in ["stay_id", "charttime"]]

        for feature in features:
            missing_count = (
                df.groupby("stay_id")[feature].apply(lambda x: x.isna().all()).sum()
            )
            percentage = (missing_count / total_stay_ids) * 100
            missing_observations[feature] = (missing_count, percentage)
        for feature, (count, percentage) in missing_observations.items():
            print(f"{feature}: {count} stay_id(s) with no observaions")
            print(f"{percentage:.2f}% of total stayids")

    def scale_normalize(self):
        df = self.data_process["imputed"].copy()

        if self.parameters["scaling"] == "Standard":
            print("scaling with StandardScaler...")
            if self.scaler:  # if scaler was given to init, use it
                print("using preasigned scaler")
            else:
                self.scaler = StandardScaler()
            df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
                df[self.NUMERICAL_FEATURES]
            )

        elif self.parameters["scaling"] == "MinMax":
            print("scaling with MinMaxScaler...")
            if self.scaler:
                print("using preasigned scaler")
            else:
                scaler = MinMaxScaler(
                    feature_range=(
                        self.parameters["scaling_range"][0],
                        self.parameters["scaling_range"][1],
                    )
                )
            df[self.NUMERICAL_FEATURES] = self.scaler.fit_transform(
                df[self.NUMERICAL_FEATURES]
            )
            self.MIMIC_NUMERICAL_FEATURES = self.NUMERICAL_FEATURES

        self.data_process["scaled"] = df
        # print(f'scaled: {self.data_process["scaled"].head(40)}')

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
        print(self.data_process["sequences_train"][0])

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
    def generate_fractions(self):
        print("Generating fractional datasets...")
        mimic_train = self.sequence_dict["mimic"]["train"]
        mimic_test = self.sequence_dict["mimic"]["test"]
        tudd_train = self.sequence_dict["tudd"]["train"]
        tudd_test = self.sequence_dict["tudd"]["test"]

        n_tudd_train = len(tudd_train) - 2000
        step_size = self.parameters["fractional_steps"]
        fractional_datasets = {}
        n_sampled_tudd_train = 0

        while n_sampled_tudd_train + step_size < n_tudd_train:
            n_sampled_tudd_train += step_size

            # getg next tudd batch
            tudd_samples = tudd_train[:n_sampled_tudd_train]
            combined_train_set = mimic_train + tudd_samples
            if self.shuffle == True:
                random.shuffle(combined_train_set)

            fractional_datasets[n_sampled_tudd_train] = combined_train_set

            print(f"fraction {n_sampled_tudd_train} added")

        self.sequence_dict["fractional_mimic_tudd"] = fractional_datasets

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
            print(f"Mean {feature} MIMIC: {mimic_df[feature].mean()}")
            print(f"Mean {feature} TUDD: {tudd_df[feature].mean()}")

    def process_tudd(self):
        measurements = self.data["tudd"]["measurements"]
        mortality_info = self.data["tudd"]["mortality_info"]

        self.SEQUENCE_FEATURES = [
            "hr_value",
            "mbp_value",
            "gcs_total_value",
            "glc_value",
            "creatinine_value",
            "potassium_value",
            "wbc_value",
            "platelets_value",
            "inr_value",
            "anion_gap_value",
            "lactate_value",
            "temperature_value",
            "weight_value",
        ]
        self.NUMERICAL_FEATURES = self.SEQUENCE_FEATURES + ["age"]  # , 'height_value']
        self.CAT_FEATURES = ["gender"]
        self.ALL_FEATURES = self.NUMERICAL_FEATURES + self.CAT_FEATURES

        measurements["measurement_offset"] = pd.to_numeric(
            measurements["measurement_offset"], errors="coerce"
        )

        measurements = pd.merge(
            measurements,
            mortality_info[
                ["caseid", "stay_duration", "age", "gender", "bodyweight", "exitus"]
            ],  #'bodyheight',
            on="caseid",
            how="left",
        )
        measurements.rename(columns={"caseid": "stay_id"}, inplace=True)
        mortality_info.rename(columns={"caseid": "stay_id"}, inplace=True)
        measurements["stay_duration_hours"] = measurements["stay_duration"] * 24
        measurements["measurement_time_from_admission"] = (
            measurements["stay_duration_hours"] + measurements["measurement_offset"]
        )

        # clean negative vals
        measurements = measurements[
            measurements["measurement_time_from_admission"] > -1
        ]
        measurements.loc[
            measurements["measurement_time_from_admission"] <= -1,
            "measurement_time_from_admission",
        ] = 0

        # filter first 24 h
        measurements = measurements[
            (measurements["measurement_time_from_admission"] >= 0)
            & (measurements["measurement_time_from_admission"] <= 24)
        ]
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

        # pivot
        measurements_pivot = measurements_agg.pivot_table(
            index=["stay_id", "measurement_time_from_admission"],
            columns="treatmentname",
            values="value",
        ).reset_index()

        # make time grid
        def create_time_grid(mortality_info):
            df_list = []
            for _, row in mortality_info.iterrows():
                stay_id = row["stay_id"]
                time_range = np.arange(0, 25)  # hour 0 to 24 inclusive
                time_df = pd.DataFrame(
                    {"stay_id": stay_id, "measurement_time_from_admission": time_range}
                )
                df_list.append(time_df)
            return pd.concat(df_list, ignore_index=True)

        time_grid = create_time_grid(mortality_info)

        # merg on time grid
        merged_df = pd.merge(
            time_grid,
            measurements_pivot,
            on=["stay_id", "measurement_time_from_admission"],
            how="left",
        )
        merged_df = pd.merge(
            merged_df,
            mortality_info[
                ["stay_id", "age", "gender", "bodyweight", "exitus"]
            ],  # 'bodyheight',
            on="stay_id",
            how="left",
        )

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
            "bodyweight": "weight_value",  # , 'bodyheight': 'height_value'
        }
        merged_df.rename(columns=treatmentnames_mapping, inplace=True)
        print(
            f'number of unique stay_ids before renaming and bounding: {merged_df["stay_id"].nunique()}'
        )
        # bounds
        bounds = {
            "age": (18, 90),
            "weight_value": (20, 500),  #'height_value': (20, 260),
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
            f'number of unique stay_ids after bounding: {merged_df["stay_id"].nunique()}'
        )
        # if self.parameters['small_data']:
        #     fraction = 0.1
        #     patient_sample = merged_df[['stay_id', 'exitus']].drop_duplicates().groupby('exitus', group_keys=False).apply(
        #         lambda x: x.sample(frac=fraction, random_state=42)
        #     )
        #     sampled_df = merged_df[merged_df['stay_id'].isin(patient_sample['stay_id'])]
        #     merged_df = sampled_df

        # mean before conversion
        print("Mean before conversion:")
        print(f"Glucose (mmol/L): {merged_df['glc_value'].mean()}")
        print(f"Creatinine (micro_mol/L): {merged_df['creatinine_value'].mean()}")
        print(f"INR (Quick): {merged_df['inr_value'].mean()}")
        print(f"Lactate (mmol/L): {merged_df['lactate_value'].mean()}")

        # convert units
        # glucose mmol/L to mg/dL
        merged_df["glc_value"] = merged_df["glc_value"] * 18.0182
        print(f"Glucose conversion done: {merged_df['glc_value'].mean()} mg/dL")

        # creatinine micro_mol/L to mg/dL
        merged_df["creatinine_value"] = merged_df["creatinine_value"] * 0.0113
        print(
            f"Creatinine conversion done: {merged_df['creatinine_value'].mean()} mg/dL"
        )

        # convert quick to inr
        merged_df["inr_value"] = merged_df["inr_value"] / 100
        print(f"INR conversion done: {merged_df['inr_value'].mean()}")

        # convert lactate mmol/L to mg/dL
        # merged_df['lactate_value'] = merged_df['lactate_value'] * 9.01
        # print(f"Lactate conversion done: {merged_df['lactate_value'].mean()} mg/dL")
        print(
            f'number of unique stay_ids before filtering: {merged_df["stay_id"].nunique()}'
        )
        # filter
        merged_df = merged_df[merged_df["age"] >= 18]
        merged_df["age"] = merged_df["age"].apply(lambda x: min(x, 90))

        for feature, (lower, upper) in bounds.items():
            if feature in merged_df.columns:
                merged_df.loc[merged_df[feature] < lower, feature] = np.nan
                merged_df.loc[merged_df[feature] > upper, feature] = np.nan
        merged_df["gender"] = merged_df["gender"].map({"m": 0, "w": 1})
        if self.compare_distributions:
            self.plot_dict["tudd"] = merged_df

        if self.parameters["golden_tudd"]:
            target_proportion_before = merged_df["exitus"].mean()
            print(f"Proportion of target before dropping: {target_proportion_before}")

            missing_counts = merged_df.groupby("stay_id")[self.ALL_FEATURES].apply(
                lambda x: x.isnull().sum().sum()
            )  # total missing across all vars
            missing_counts = missing_counts.sort_values()  # sort from least to most
            top_1000_stay_ids = missing_counts.index[:1000]
            merged_df = merged_df[merged_df["stay_id"].isin(top_1000_stay_ids)]

            target_proportion_after = merged_df["exitus"].mean()
            print(f"Proportion of target after dropping: {target_proportion_after}")

            if merged_df["stay_id"].nunique() != 1000:
                raise ValueError(
                    f"Expected 1000 unique stay_ids, but got {merged_df['stay_id'].nunique()}."
                )

        print(
            f'number of unique stay_ids before iumputing: {merged_df["stay_id"].nunique()}'
        )
        # imputation
        # merged_df.sort_values(['stay_id', 'measurement_time_from_admission'], inplace=True)
        merged_df = self.impute(merged_df)
        # if self.imputation['method'] == 'ffill_bfill':
        #     merged_df = merged_df.groupby('caseid').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

        #     self.find_variables_with_only_nan(merged_df)

        #     stay_ids_to_drop = merged_df.groupby('caseid')[self.NUMERICAL_FEATURES].apply(
        #     lambda group: group.isnull().all(axis=0).any()
        # )
        #     stay_ids_to_drop = stay_ids_to_drop[stay_ids_to_drop].index

        #     print(f"Number of TUDD observations before dropping: {len(merged_df)}")
        #     #merged_df = merged_df[~merged_df['caseid'].isin(stay_ids_to_drop)]
        #     print(f"Number of TUDD observations left after dropping: {len(merged_df)}")

        # elif self.imputation['method'] == 'knn':
        #     merged_df = self.impute_with_knn(merged_df)

        if self.compare_distributions:
            self.plot_dict["tudd_imputed"] = merged_df.copy()
        # categorical features
        merged_df["gender"].fillna(merged_df["gender"].mode()[0], inplace=True)
        merged_df["exitus"].fillna(0, inplace=True)

        if self.parameters["scaling"] == "standard":
            # scaler = StandardScaler()
            merged_df[self.MIMIC_NUMERICAL_FEATURES] = self.mimic_scaler.transform(
                merged_df[self.MIMIC_NUMERICAL_FEATURES]
            )
        elif self.parameters["scaling"] == "MinMax":
            # scaler = MinMaxScaler(feature_range=(self.parameters['scaling_range'][0], self.parameters['scaling_range'][1]))
            merged_df[self.MIMIC_NUMERICAL_FEATURES] = self.mimic_scaler.transform(
                merged_df[self.MIMIC_NUMERICAL_FEATURES]
            )

        # column_order = [
        #     'caseid', 'measurement_time_from_admission', 'mbp_value', 'gcs_total_value', 'glc_value',
        #     'creatinine_value', 'potassium_value', 'hr_value', 'wbc_value', 'platelets_value',
        #     'lactate_value','temperature_value','weight_value', 'inr_value', 'anion_gap_value', 'exitus',
        #     'age', 'gender', 'height_value'
        # ]
        column_order = [
            "stay_id",
            "measurement_time_from_admission",
            "mbp_value",
            "gcs_total_value",
            "glc_value",
            "creatinine_value",
            "potassium_value",
            "hr_value",
            "wbc_value",
            "platelets_value",
            "temperature_value",
            "weight_value",
            "exitus",
            "age",
            "gender",
        ]
        sorted_merged_df = merged_df[column_order]

        # # drop 'anion_gap_value'
        # sorted_merged_df.drop(columns=['anion_gap_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('anion_gap_value')
        # self.NUMERICAL_FEATURES.remove('anion_gap_value')
        # self.ALL_FEATURES.remove('anion_gap_value')

        # # drop 'inr_value'
        # sorted_merged_df.drop(columns=['inr_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('inr_value')
        # self.NUMERICAL_FEATURES.remove('inr_value')
        # self.ALL_FEATURES.remove('inr_value')

        # # drop lactate_value
        # sorted_merged_df.drop(columns=['lactate_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('lactate_value')
        # self.NUMERICAL_FEATURES.remove('lactate_value')
        # self.ALL_FEATURES.remove('lactate_value')

        print(f"Number of unique stay_ids: {sorted_merged_df['stay_id'].nunique()}")

        sequences = []
        for stay_id, group in sorted_merged_df.groupby("stay_id"):
            if len(group) == 25:
                features = group[self.ALL_FEATURES_MIMIC].values
                label = group["exitus"].iloc[0]
                sequences.append((features, label))

        self.sequence_dict["tudd"] = {}
        if self.parameters["golden_tudd"]:
            self.sequence_dict["tudd"]["test"] = sequences
            self.sequence_dict["tudd"]["train"] = []
        else:
            labels = [seq[1] for seq in sequences]
            self.sequence_dict["tudd"]["train"], self.sequence_dict["tudd"]["test"] = (
                train_test_split(
                    sequences, test_size=0.2, stratify=labels, random_state=42
                )
            )
