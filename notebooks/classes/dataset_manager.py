import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from classes.preprocessor import Preprocessor
from utils import set_seed
import random

set_seed(42)
random.seed(42)

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


class DatasetManager:
    def __init__(self, variables: list, parameters={}):
        # dataset types: mimic_mimic, mimic_tudd, tudd_tudd,
        # tudd_mimic, tudd_fract
        self.mimic_datapath = os.path.join(project_root, "data/raw/mimiciv/first_24h/")
        self.tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
        self.variables = variables
        self.dataset_type = parameters["dataset_type"]
        if isinstance(self.dataset_type, list):
            self.dataset_type = self.dataset_type[0]

        self.parameters = parameters
        self.data = {}
        random.seed(42)

    def load_data(self):
        if "mimic" in self.dataset_type or "combined" in self.dataset_type:
            print("Loading MIMIC data...")
            self.data["mimic"] = {}
            self.load_mimic()
            if self.parameters.get("small_data", False):
                self.reduce_data()
            self.preprocess("mimic")
            print(
                f"MIMIC - Training sequences: {len(self.data['mimic']['sequences_train'])}, Test sequences: {len(self.data['mimic']['sequences_test'])}"
            )

        if "tudd" in self.dataset_type or "combined" in self.dataset_type:
            print("Loading TUDD data...")
            self.data["tudd"] = {}
            self.load_tudd()
            self.preprocess("tudd")

            print(
                f"TUDD - Training sequences: {len(self.data['tudd']['sequences_train'])}, Test sequences: {len(self.data['tudd']['sequences_test'])}"
            )

        if "combined" in self.dataset_type:
            print("Creating combined splits...")
            self.create_combined_splits_50_50()
            # self.create_combined_splits_full()
        if "fract" in self.dataset_type:
            print("Creating fractional splits...")
            self.generate_fractions()

    def create_combined_splits_full(self):
        """
        Creates full combined training and test splits from the full MIMIC and TUDD datasets.
        Instead of taking balanced samples, this function concatenates all available training
        and test sequences from MIMIC and TUDD.
        """

        # --- Create full combined training set ---
        mimic_train = self.data["mimic"].get("sequences_train")
        tudd_train = self.data["tudd"].get("sequences_train")
        if mimic_train is None or tudd_train is None:
            raise ValueError(
                "Both MIMIC and TUDD training sequences must be available."
            )

        # Concatenate full training sets from both datasets.
        combined_train = mimic_train + tudd_train
        random.shuffle(combined_train)

        # --- Create full combined test set ---
        mimic_test = self.data["mimic"].get("sequences_test")
        tudd_test = self.data["tudd"].get("sequences_test")
        if mimic_test is None or tudd_test is None:
            raise ValueError("Both MIMIC and TUDD test sequences must be available.")

        # Concatenate full test sets from both datasets.
        combined_test = mimic_test + tudd_test
        random.shuffle(combined_test)

        # Save the full combined splits under a new key.
        self.data["combined"] = {
            "sequences_train": combined_train,
            "sequences_test": combined_test,
        }
        print(
            f"Full combined splits created: {len(combined_train)} training and {len(combined_test)} test sequences."
        )

    def stratified_sample(self, sequences, sample_count):
        if sample_count == len(sequences):
            return sequences
        # Updated: use index 2 of the sequence tuple (label) for stratification
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
            # Store the indices only to avoid memory issues
            fractional_indices[n_sampled] = list(range(n_sampled))

        self.data["fractional_indices"] = fractional_indices

    def create_combined_splits_50_50(self):
        # --- Create combined training set ---
        mimic_train = self.data["mimic"].get("sequences_train")
        tudd_train = self.data["tudd"].get("sequences_train")
        if mimic_train is None or tudd_train is None:
            raise ValueError("Both mimic and tudd training sequences must be available.")

        n_train = min(len(mimic_train), len(tudd_train))
        print(f"n_train from each: {n_train}")
        mimic_train_sample = self.stratified_sample(mimic_train, n_train)
        tudd_train_sample = self.stratified_sample(tudd_train, n_train)
        combined_train = mimic_train_sample + tudd_train_sample
        random.shuffle(combined_train)

        # --- Create combined test set ---
        mimic_test = self.data["mimic"].get("sequences_test")
        tudd_test = self.data["tudd"].get("sequences_test")
        if mimic_test is None or tudd_test is None:
            raise ValueError("Both mimic and tudd test sequences must be available.")

        n_test = min(len(mimic_test), len(tudd_test))
        mimic_test_sample = self.stratified_sample(mimic_test, n_test)
        tudd_test_sample = self.stratified_sample(tudd_test, n_test)
        combined_test = mimic_test_sample + tudd_test_sample
        random.shuffle(combined_test)

        self.data["combined"] = {
            "sequences_train": combined_train,
            "sequences_test": combined_test,
        }
        print(f"Combined splits created: {len(combined_train)} training and {len(combined_test)} test sequences.")

    def preprocess(self, data_type: str):
        sequences_dict = {}
        feature_index_mapping = {}

        if data_type == "mimic":
            preprocessor_mimic = Preprocessor(
                data_type,
                self.data["mimic"],
                self.variables,
                self.parameters,
            )
            preprocessor_mimic.process()
            # has the attributes: data_process (dict with:
            # pre_processing, aggregated, merged, imputed, scaled, sequences,
            # sequences_train, sequences_test
            # ),
            # feature_index_mapping_sequences, scaler,
            self.data["mimic"] = preprocessor_mimic.data_process
            self.feature_names = preprocessor_mimic.ALL_FEATURES
            self.numerical_features = preprocessor_mimic.NUMERICAL_FEATURES

            self.scaler = preprocessor_mimic.scaler

        if "tudd" in self.data:
            preprocessor_args = {
                "data_type": data_type,
                "data": self.data["tudd"],
                "variables": self.variables,
                "parameters": self.parameters,
            }

            # include mimic scaler if it exists
            if hasattr(self, "scaler"):
                preprocessor_args["scaler"] = self.scaler

            preprocessor_tudd = Preprocessor(**preprocessor_args)

            preprocessor_tudd.process()
            self.data["tudd"] = preprocessor_tudd.data_process
            self.feature_names = preprocessor_tudd.ALL_FEATURES
            self.numerical_features = preprocessor_tudd.NUMERICAL_FEATURES
            self.scaler = preprocessor_tudd.scaler

    def load_mimic(self):
        print("Loading MIMIC data...")
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
                    # make sure that only the statric data that is in the variables dict in high level_script gets loaded
                    self.data["mimic"]["static_data"] = self.data["mimic"][variable][
                        list(static_data_keys)
                    ]

            else:
                print(
                    f"Warning: {variable}.csv does not exist in {self.mimic_datapath}"
                )

    def load_tudd(self):
        file_path = os.path.join(self.tudd_datapath, "measurement.csv")
        if os.path.exists(file_path):
            self.data["tudd"]["measurements"] = pd.read_csv(
                file_path, sep="|", index_col=False
            )
        else:
            raise FileNotFoundError(f"{file_path} does not exist.")
        print(self.data["tudd"]["measurements"].head())
        mortality_info_path = os.path.join(self.tudd_datapath, "stays.csv")
        if os.path.exists(mortality_info_path):
            self.data["tudd"]["mortality_info"] = pd.read_csv(
                mortality_info_path, sep="|", index_col=False
            )
        else:
            raise FileNotFoundError(f"{mortality_info_path} does not exist.")

        # # mortality info
        # mortality_info_x_path = os.path.join(self.tudd_datapath, "stays_ane.csv")
        # mortality_info_y_path = os.path.join(
        #     self.tudd_datapath, "stays_others2_ane.csv"
        # )

        # mortality_info_list = []
        # for path in [mortality_info_x_path, mortality_info_y_path]:
        #     if os.path.exists(path):
        #         mortality_info_list.append(pd.read_csv(path, sep="|"))
        #     else:
        #         raise FileNotFoundError(f"{path} does not exist.")

        # self.data["tudd"]["mortality_info"] = pd.concat(
        #     mortality_info_list, ignore_index=True
        # )

    def reduce_data(self):  # TODO not implemented for tudd yet and also fot mimic_tudd
        if self.dataset_type == "mimic_mimic":
            static = self.data["mimic"]["static_data"]
            static_small = train_test_split(
                static,
                test_size=0.9,
                stratify=static[f'{self.parameters["target"]}_value'],
            )[0]
            stay_ids = static_small["stay_id"]
            for variable in self.data["mimic"].keys():
                if variable != "static_data":
                    if "stay_id" in self.data["mimic"][variable].columns:
                        self.data["mimic"][variable] = self.data["mimic"][variable][
                            self.data["mimic"][variable]["stay_id"].isin(stay_ids)
                        ]
            self.data["mimic"]["static_data"] = static_small
        elif self.dataset_type == "tudd_tudd":
            raise NotImplementedError(
                "Method for reducing TUDD data is not implemented yet."
            )
