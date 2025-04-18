import os
import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import set_seed
from classes.preprocessor import Preprocessor

set_seed(42)
random.seed(42)

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


class DatasetManager:
    def __init__(self, variables: dict, parameters={}):
        # dataset types: mimic_mimic, mimic_tudd, tudd_tudd,
        # tudd_mimic, tudd_fract, etc.
        self.mimic_datapath = os.path.join(project_root, "data/raw/mimiciv/first_24h/")
        self.tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
        self.variables = variables
        self.dataset_type = parameters["dataset_type"]
        if isinstance(self.dataset_type, list):
            self.dataset_type = self.dataset_type[0]
        self.parameters = parameters
        self.data = {}
        random.seed(42)

    def load_all_mimic_and_tudd_data(self):
        # --- Load and preprocess MIMIC data ---
        self.data["mimic"] = {}
        self.load_mimic()
        # Uncomment if you want to reduce data size (for debugging)
        # if self.parameters.get("small_data", False):
        #     self.reduce_data()
        self.preprocess("mimic")
        
        # --- Load and preprocess TUDD data ---
        self.data["tudd"] = {}
        self.load_tudd()
        self.preprocess("tudd")
        
        # (Optional) Once both are processed, you can compute aggregate stats and plot missingness:
        print("\nMIMIC Dataset Patient Statistics:")
        self.compute_patient_statistics("mimic")
        print("\nTUDD Dataset Patient Statistics:")
        self.compute_patient_statistics("tudd")
        
        # print("\nMIMIC Time-dependent Missingness Heatmap:")
        # self.plot_time_missingness_heatmap("mimic")
        # print("\nTUDD Time-dependent Missingness Heatmap:")
        # self.plot_time_missingness_heatmap("tudd")
    
    def load_data(self):
        if self.dataset_type in ['mimic_mimic', 'mimic_tudd', 'mimic_combined', 'combined_mimic', 'combined_tudd', 'combined_combined', 'mimic_tudd_fract', 'mimic_mimic_fract']:
            self.data["mimic"] = {}
            self.load_mimic()
            if self.parameters.get("small_data", False):
                self.reduce_data()
            self.preprocess("mimic")
            self.compute_patient_statistics("mimic")
            # self.plot_time_missingness_heatmap("mimic")
            if self.dataset_type in ['mimic_tudd','mimic_combined', 'combined_mimic','combined_tudd', 'combined_combined', 'mimic_tudd_fract']:
                print('Loading TUDD data nested within the dataset type...')
                self.data["tudd"] = {}
                self.load_tudd()
                self.preprocess("tudd")
                self.compute_patient_statistics("tudd")
                # self.plot_time_missingness_heatmap("tudd")
        if self.dataset_type in ['tudd_mimic', 'tudd_tudd', 'tudd_combined']:
            print("Loading TUDD data...")
            self.data["tudd"] = {}
            self.load_tudd()
            self.preprocess("tudd")
            self.compute_patient_statistics("tudd")
            if self.dataset_type in ['tudd_mimic','tudd_combined']:
                self.data["mimic"] = {}
                self.load_mimic()
                if self.parameters.get("small_data", False):
                    self.reduce_data()
                self.preprocess("mimic")
        if "combined" in self.dataset_type:
            self.apply_combined_scaling_and_create_sequences()
        if "fract" in self.dataset_type:
            self.generate_fractions()

    def compute_patient_statistics(self, dataset_name: str):
  
        print(f"\nComputing patient statistics for dataset: {dataset_name}")
        df = self.data[dataset_name].get("merged")
        if df is None:
            print(f"No data available for statistics for {dataset_name}")
            return None

        time_col = "charttime" if dataset_name == "mimic" else "measurement_time_from_admission"
        time_vars = [f"{var}_value" for var in self.variables if var != "static_data"]

        for var in time_vars:
            df[var] = pd.to_numeric(df[var].astype(str).str.replace(',', ''), errors='coerce')

        patient_groups = df.groupby("stay_id")
        total_patients = df["stay_id"].nunique()
        total_time_steps = df.shape[0]

        results = []
        time_vars += ["age_value"] 
        for var in time_vars:
            per_patient_avg = patient_groups[var].mean()
            overall_mean = per_patient_avg.mean()
            overall_std = per_patient_avg.std()

            patients_without_record = patient_groups[var].apply(lambda x: x.isna().all()).sum()
            pct_patients_without = (patients_without_record / total_patients) * 100

            pct_time_missing = (df[var].isna().sum() / total_time_steps) * 100

            mean_measurements = (
                patient_groups[var]
                .apply(lambda x: x.notna().sum())
                .loc[lambda x: x > 0] 
                .mean()
            )

            results.append({
                "variable": var,
                "per_patient_mean": overall_mean,
                "per_patient_std": overall_std,
                "pct_patients_without_record": pct_patients_without,
                "pct_time_steps_missing": pct_time_missing,
                "mean_measurements_per_patient": mean_measurements
            })
        if "gender_value" in df.columns:
            df["gender_value"] = pd.to_numeric(df["gender_value"].astype(str).str.replace(',', ''), errors='coerce')
            gender_df = df.groupby("stay_id")["gender_value"].first()
            pct_female = gender_df.mean() * 100
            results.append({
                "variable": "gender_value",
                "per_patient_mean": pct_female,
                "per_patient_std": None,
                "pct_patients_without_record": gender_df.isna().mean() * 100,
                "pct_time_steps_missing": None,
                "mean_measurements_per_patient": "--"
            })
        if "mortality_value" in df.columns:
            df["mortality_value"] = pd.to_numeric(df["mortality_value"].astype(str).str.replace(',', ''), errors='coerce')
            mortality_df = df.groupby("stay_id")["mortality_value"].first()
            pct_mortality = mortality_df.mean() * 100
            results.append({
                "variable": "mortality_value",
                "per_patient_mean": pct_mortality,
                "per_patient_std": None,
                "pct_patients_without_record": mortality_df.isna().mean() * 100,
                "pct_time_steps_missing": None,
                "mean_measurements_per_patient": "--"
            })

        if "weight_value" in df.columns:
            df["weight_value"] = pd.to_numeric(df["weight_value"].astype(str).str.replace(',', ''), errors='coerce')
            weight_df = df.groupby("stay_id")["weight_value"].first()
            weight_mean = weight_df.mean()
            weight_std = weight_df.std()
            results.append({
                "variable": "weight_value",
                "per_patient_mean": weight_mean,
                "per_patient_std": weight_std,
                "pct_patients_without_record": weight_df.isna().mean() * 100,
                "pct_time_steps_missing": None,
                "mean_measurements_per_patient": "--"
            })

        results_df = pd.DataFrame(results)
        print(f"\nPatient statistics for dataset '{dataset_name}':")
        print(results_df)
        return results_df

    
    def apply_combined_scaling_and_create_sequences(self):
        target_col = f'{self.parameters["target"]}_value'
        mimic_imputed = self.data["mimic"]["imputed"].copy()
        tudd_imputed = self.data["tudd"]["imputed"].copy()
        mimic_labels = mimic_imputed.groupby("stay_id")[target_col].first()
        mimic_train_ids, mimic_test_ids = train_test_split(
            mimic_labels.index, test_size=0.2, stratify=mimic_labels.values, random_state=42
        )
        mimic_train = mimic_imputed[mimic_imputed["stay_id"].isin(mimic_train_ids)].copy()
        mimic_test = mimic_imputed[mimic_imputed["stay_id"].isin(mimic_test_ids)].copy()
        tudd_labels = tudd_imputed.groupby("stay_id")['exitus'].first()
        tudd_train_ids, tudd_test_ids = train_test_split(
            tudd_labels.index, test_size=0.2, stratify=tudd_labels.values, random_state=42
        )
        tudd_train = tudd_imputed[tudd_imputed["stay_id"].isin(tudd_train_ids)].copy()
        tudd_test = tudd_imputed[tudd_imputed["stay_id"].isin(tudd_test_ids)].copy()
        combined_train_df = pd.concat([mimic_train, tudd_train], ignore_index=True)
        if self.dataset_type.startswith("tudd"):
            scaler = StandardScaler()
            scaler.fit(tudd_train[self.numerical_features])
        elif self.dataset_type.startswith("mimic"):
            scaler = StandardScaler()
            scaler.fit(mimic_train[self.numerical_features])
        else:
            scaler = StandardScaler()
            scaler.fit(combined_train_df[self.numerical_features])
        mimic_train[self.numerical_features] = scaler.transform(mimic_train[self.numerical_features])
        mimic_test[self.numerical_features] = scaler.transform(mimic_test[self.numerical_features])
        tudd_train[self.numerical_features] = scaler.transform(tudd_train[self.numerical_features])
        tudd_test[self.numerical_features] = scaler.transform(tudd_test[self.numerical_features])
        self.scaler = scaler
        seq_feature_names = [f"{var}_value" for var in self.variables if var != "static_data"]
        exclude_static = {"mortality", "intime", "first_day_end", "stay_id"}
        static_feature_names = [
            f"{key}_value" for key, attr in self.variables["static_data"].items() 
            if key not in exclude_static and attr.get("training", False)
        ]
        def build_sequences(df, time_col, target):
            sequences = []
            for stay_id, group in df.groupby("stay_id"):
                group = group.sort_values(time_col)
                static_features = group.iloc[0][static_feature_names].values.astype(np.float32)
                sequential_features = group[seq_feature_names].values.astype(np.float32)
                label = group[target].iloc[0]
                sequences.append((sequential_features, static_features, label))
            return sequences
        mimic_sequences_train = build_sequences(mimic_train, "charttime", target_col)
        mimic_sequences_test = build_sequences(mimic_test, "charttime", target_col)
        tudd_sequences_train = build_sequences(tudd_train, "measurement_time_from_admission", 'exitus')
        tudd_sequences_test = build_sequences(tudd_test, "measurement_time_from_admission", 'exitus')
        n_train = min(len(mimic_sequences_train), len(tudd_sequences_train))
        mimic_train_sample = self.stratified_sample(mimic_sequences_train, n_train)
        tudd_train_sample = self.stratified_sample(tudd_sequences_train, n_train)
        combined_train = mimic_train_sample + tudd_train_sample
        random.shuffle(combined_train)
        n_test = min(len(mimic_sequences_test), len(tudd_sequences_test))
        mimic_test_sample = self.stratified_sample(mimic_sequences_test, n_test)
        tudd_test_sample = self.stratified_sample(tudd_sequences_test, n_test)
        combined_test = mimic_test_sample + tudd_test_sample
        random.shuffle(combined_test)
        self.data["combined"] = {
            "sequences_train": combined_train,
            "sequences_test": combined_test,
            "scaler": scaler
        }
        self.data["mimic"] = {
            "sequences_train": mimic_sequences_train,
            "sequences_test": mimic_sequences_test,
            "scaler": scaler
        }
        self.data["tudd"] = {
            "sequences_train": tudd_sequences_train,
            "sequences_test": tudd_sequences_test,
            "scaler": scaler
        }

    def stratified_sample(self, sequences, sample_count):
        if sample_count == len(sequences):
            return sequences
        labels = [seq[2] for seq in sequences]
        sample, _, _, _ = train_test_split(
            sequences, labels, train_size=sample_count, stratify=labels, random_state=42
        )
        return sample

    def generate_fractions(self):
        if "mimic_tudd_fract" in self.dataset_type:
            train_data_full = (
                self.data["mimic"]["sequences_train"] + self.data["mimic"]["sequences_test"]
            )
            random.shuffle(train_data_full)
            train_data_for_fractions = self.data["tudd"]["sequences_train"]
            self.data["train_fractions"] = train_data_for_fractions
        elif "tudd_fract" in self.dataset_type:
            train_data_for_fractions = self.data["tudd"]["sequences_train"]
            random.shuffle(train_data_for_fractions)
            self.data["train_fractions"] = train_data_for_fractions
        elif "mimic_fract" in self.dataset_type:
            train_data_for_fractions = self.data["mimic"]["sequences_train"]
            random.shuffle(train_data_for_fractions)
            self.data["train_fractions"] = train_data_for_fractions
        if "mimic_tudd_fract" in self.dataset_type:
            self.data["mimic_train_all"] = train_data_full
        n_train = len(train_data_for_fractions)
        step_size = self.parameters["fractional_steps"]
        fractional_indices = {}
        n_sampled = 0
        while n_sampled + step_size < n_train:
            n_sampled += step_size
            fractional_indices[n_sampled] = list(range(n_sampled))
        self.data["fractional_indices"] = fractional_indices

    def preprocess(self, data_type: str):
        if data_type == "mimic":
            preprocessor_args = {
                "data_type": data_type,
                "data": self.data["mimic"],
                "variables": self.variables,
                "parameters": self.parameters,
            }
            if hasattr(self, "scaler"):
                preprocessor_args["scaler"] = self.scaler
                print("Using preassigned scaler for MIMIC")
            preprocessor_mimic = Preprocessor(**preprocessor_args)
            preprocessor_mimic.process()
            self.data["mimic"] = preprocessor_mimic.data_process
            self.feature_names = preprocessor_mimic.ALL_FEATURES
            self.numerical_features = preprocessor_mimic.NUMERICAL_FEATURES
            self.scaler = preprocessor_mimic.scaler
            print('Saved MIMIC scaler.')
        elif "tudd" in self.data:
            preprocessor_args = {
                "data_type": data_type,
                "data": self.data["tudd"],
                "variables": self.variables,
                "parameters": self.parameters,
            }
            if hasattr(self, "scaler"):
                preprocessor_args["scaler"] = self.scaler
                print("Using preassigned scaler for TUDD")
            preprocessor_tudd = Preprocessor(**preprocessor_args)
            preprocessor_tudd.process()
            self.data["tudd"] = preprocessor_tudd.data_process
            self.feature_names = preprocessor_tudd.ALL_FEATURES
            self.numerical_features = preprocessor_tudd.NUMERICAL_FEATURES
            self.scaler = preprocessor_tudd.scaler
            print('Saved TUDD scaler.')

    def load_mimic(self):
        for variable, _ in self.variables.items():
            file_path = os.path.join(self.mimic_datapath, f"{variable}.csv")
            if os.path.exists(file_path):
                self.data["mimic"][variable] = pd.read_csv(file_path)
                if variable == "static_data":
                    no_val_keys = {"intime", "first_day_end", "stay_id"}
                    static_data_keys = [
                        f"{key}_value" if key not in no_val_keys else key
                        for key in self.variables["static_data"].keys()
                    ]
                    self.data["mimic"]["static_data"] = self.data["mimic"][variable][list(static_data_keys)]
            # else:
            #     print(f"Warning: {variable}.csv does not exist in {self.mimic_datapath}")

    def load_tudd(self):
        file_path = os.path.join(self.tudd_datapath, "measurement.csv")
        if os.path.exists(file_path):
            self.data["tudd"]["measurements"] = pd.read_csv(file_path, sep="|", index_col=False)
        else:
            raise FileNotFoundError(f"{file_path} does not exist.")
        mortality_info_path = os.path.join(self.tudd_datapath, "stays.csv")
        if os.path.exists(mortality_info_path):
            self.data["tudd"]["mortality_info"] = pd.read_csv(mortality_info_path, sep="|", index_col=False)
        else:
            raise FileNotFoundError(f"{mortality_info_path} does not exist.")

    def reduce_data(self):
        if self.dataset_type == "mimic_mimic":
            static = self.data["mimic"]["static_data"]
            static_small = train_test_split(
                static,
                test_size=0.9,
                stratify=static[f'{self.parameters["target"]}_value'],
            )[0]
            stay_ids = static_small["stay_id"]
            for variable in self.data["mimic"].keys():
                if variable != "static_data" and "stay_id" in self.data["mimic"][variable].columns:
                    self.data["mimic"][variable] = self.data["mimic"][variable][
                        self.data["mimic"][variable]["stay_id"].isin(stay_ids)
                    ]
            self.data["mimic"]["static_data"] = static_small
        elif self.dataset_type == "tudd_tudd":
            raise NotImplementedError("Method for reducing TUDD data is not implemented yet.")
