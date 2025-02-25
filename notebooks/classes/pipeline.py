import os
import pickle
import copy
import hashlib
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
from classes.explainer import SHAPExplainer
import sys
import json

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


class Pipeline(object):
    def __init__(self, variables, parameters, show=False, new_data=True):
        self.variables = variables
        self.parameters = parameters
        self.show = show
        self.force_preprocess = new_data
        # Folder to save preprocessed splits
        self.preprocessed_dir = os.path.join(project_root, "data/preprocessed")
        os.makedirs(self.preprocessed_dir, exist_ok=True)

    def prepare_data(self):
        dt = self.parameters["dataset_type"]
        if isinstance(dt, list):
            dt = dt[0]

        # Determine which files to load based on dataset type
        if dt == "mimic_mimic":
            train_file = os.path.join(self.preprocessed_dir, "mimic_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "mimic_test.pkl")
        elif dt == "tudd_tudd":
            train_file = os.path.join(self.preprocessed_dir, "tudd_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "tudd_test.pkl")
        elif dt == "mimic_tudd":
            train_file = os.path.join(self.preprocessed_dir, "mimic_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "tudd_test.pkl")
        elif dt == "tudd_mimic":
            train_file = os.path.join(self.preprocessed_dir, "tudd_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "mimic_test.pkl")
        elif dt == "mimic_combined":  # Train on mimic only; test on combined
            train_file = os.path.join(self.preprocessed_dir, "mimic_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "combined_test.pkl")
        elif dt == "tudd_combined":  # Train on tudd only; test on combined
            train_file = os.path.join(self.preprocessed_dir, "tudd_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "combined_test.pkl")
        elif dt == "combined_mimic":  # Train on combined; test on mimic
            train_file = os.path.join(self.preprocessed_dir, "combined_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "mimic_test.pkl")
        elif dt == "combined_tudd":  # Train on combined; test on tudd
            train_file = os.path.join(self.preprocessed_dir, "combined_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "tudd_test.pkl")
        elif dt == "combined_combined":  # Both train and test on combined splits
            train_file = os.path.join(self.preprocessed_dir, "combined_train.pkl")
            test_file = os.path.join(self.preprocessed_dir, "combined_test.pkl")
        else:
            raise ValueError(f"Dataset type {dt} is not supported.")

        # If both files exist and we're not forcing a re-preprocess, load them.
        if (
            os.path.exists(train_file) and os.path.exists(test_file)
        ) and not self.force_preprocess:
            print("Loading preprocessed train and test splits from cache...")
            with open(train_file, "rb") as f:
                train_data = pickle.load(f)
            with open(test_file, "rb") as f:
                test_data = pickle.load(f)
            # Additionally, load feature metadata from one of the files (or a separate file)
            # Here, we assume train_data contains metadata as well.
            self.feature_names = train_data["feature_names"]
            self.scaler = train_data["scaler"]
            self.numerical_features = train_data["numerical_features"]

            # Reconstruct a minimal DataManager-like object.
            self.DataManager = type("SimpleDataManager", (), {})()
            # Save the loaded splits in a dictionary for consistency.
            self.DataManager.data = {
                "train": train_data["data"],
                "test": test_data["data"],
            }
        else:
            print("Preprocessing data from scratch...")
            self.DataManager = DatasetManager(
                variables=self.variables, parameters=self.parameters
            )
            self.DataManager.load_data()  # Fills self.DataManager.data

            # Extract metadata from the full dataset manager.
            self.feature_names = self.DataManager.feature_names
            self.scaler = self.DataManager.scaler
            self.numerical_features = self.DataManager.numerical_features

            # For consistency, build a dictionary of preprocessed splits.
            # Here we assume that self.DataManager.data has keys like "mimic", "tudd", "combined"
            # each containing "sequences_train" and "sequences_test".
            if dt in ["mimic_mimic", "mimic_tudd", "mimic_combined"]:
                train_split = self.DataManager.data["mimic"]["sequences_train"]
            elif dt in ["tudd_tudd", "tudd_mimic", "tudd_combined"]:
                train_split = self.DataManager.data["tudd"]["sequences_train"]
            elif dt.startswith("combined"):
                train_split = self.DataManager.data["combined"]["sequences_train"]
            else:
                raise ValueError(f"Unexpected dataset type for train split: {dt}")

            if dt in ["mimic_mimic", "tudd_mimic", "combined_mimic"]:
                test_split = self.DataManager.data["mimic"]["sequences_test"]
            elif dt in ["tudd_tudd", "mimic_tudd", "combined_tudd"]:
                test_split = self.DataManager.data["tudd"]["sequences_test"]
            elif dt.endswith("combined"):
                test_split = self.DataManager.data["combined"]["sequences_test"]
            else:
                raise ValueError(f"Unexpected dataset type for test split: {dt}")

            # Save the splits along with the metadata.
            train_cache = {
                "data": train_split,
                "feature_names": self.feature_names,
                "scaler": self.scaler,
                "numerical_features": self.numerical_features,
            }
            test_cache = {"data": test_split}

            with open(train_file, "wb") as f:
                pickle.dump(train_cache, f)
            with open(test_file, "wb") as f:
                pickle.dump(test_cache, f)

            # For internal consistency, store the splits in a minimal DataManager.
            self.DataManager.data = {"train": train_split, "test": test_split}

            print("Preprocessed splits saved to cache.")

    def train(self):
        dt = self.parameters["dataset_type"]
        if isinstance(dt, list):
            dt = dt[0]
        trainer = Trainer(self.parameters)

        # Retrieve the proper splits from the minimal DataManager.
        if dt == "mimic_mimic":
            train_data = self.DataManager.data["train"]
            test_data = self.DataManager.data["test"]
        elif dt == "tudd_tudd":
            train_data = self.DataManager.data["train"]
            test_data = self.DataManager.data["test"]
        elif dt == "mimic_tudd":
            train_data = self.DataManager.data["train"]
            test_data = self.DataManager.data["test"]
        elif dt == "tudd_mimic":
            train_data = self.DataManager.data["train"]
            test_data = self.DataManager.data["test"]
        elif dt in [
            "mimic_combined",
            "tudd_combined",
            "combined_mimic",
            "combined_tudd",
            "combined_combined",
        ]:
            train_data = self.DataManager.data["train"]
            test_data = self.DataManager.data["test"]
        else:
            raise ValueError(f"Dataset type {dt} is not supported.")

        self.result_dict, self.trained_models = trainer.train(train_data, test_data)
        self.test_sequences = test_data

    def explain(self, model_name, method, num_samples=1000):
        explainer = SHAPExplainer(
            model=self.trained_models[model_name],
        )
        explainer.explain(
            self.test_sequences,
            self.feature_names,
            method,
            num_samples,
            self.scaler,
            self.numerical_features,
        )

    def memorize(self, file_path="parameters_results.csv"):
        if self.parameters.get("fractional_steps"):
            entry = {**self.parameters, **self.result_dict}
        else:
            entry = {**self.parameters, **self.result_dict[0]}

        params_hash = hashlib.md5(str(sorted(entry.items())).encode()).hexdigest()
        entry["parameters_hash"] = params_hash

        entry_exists = False
        if os.path.exists(file_path):
            with open(file_path, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get("parameters_hash") == params_hash:
                        entry_exists = True
                        break

        if not entry_exists:
            fieldnames = list(entry.keys())
            with open(file_path, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(entry)

    def visualize_sequences(self):
        # This method remains unchanged.
        if (
            "tudd" in self.parameters["dataset_type"]
            and "mimic" in self.parameters["dataset_type"]
        ):
            mimic_sequences = self.sequences["mimic"]["train"]
            tudd_sequences = self.sequences["tudd"]["train"]

            num_features = mimic_sequences[0][0].shape[1]

            mimic_feature_values = [[] for _ in range(num_features)]
            tudd_feature_values = [[] for _ in range(num_features)]

            for seq in mimic_sequences:
                for feature_idx in range(num_features):
                    mimic_feature_values[feature_idx].extend(seq[0][:, feature_idx])

            for seq in tudd_sequences:
                for feature_idx in range(num_features):
                    tudd_feature_values[feature_idx].extend(seq[0][:, feature_idx])

            for feature_idx in range(num_features):
                plt.figure(figsize=(10, 6))
                sns.kdeplot(
                    mimic_feature_values[feature_idx],
                    label="MIMIC",
                    fill=True,
                    alpha=0.5,
                )
                sns.kdeplot(
                    tudd_feature_values[feature_idx], label="TUDD", fill=True, alpha=0.5
                )
                plt.title(f"Density Plot for {self.feature_index_mapping[feature_idx]}")
                plt.xlabel(f"Feature {feature_idx}")
                plt.ylabel("Density")
                plt.legend()
                plt.show()


class MultiDatasetPipeline(Pipeline):
    def __init__(self, variables, parameters, dataset_types, show=False, new_data=True):
        super().__init__(variables, parameters, show, new_data)
        self.dataset_types = dataset_types
        self.data_managers = (
            {}
        )  # Will hold preprocessed DatasetManager objects per dataset_type

    def run_all(self, model_list, memorize=False, explain_method=None, num_samples=10):
        all_results = {}
        for ds in self.dataset_types:
            # Update parameters for current dataset type.
            self.parameters["dataset_type"] = ds
            print(f"\n=== Running experiments for dataset '{ds}' ===")
            # Prepare data for this dataset type.
            self.prepare_data()
            for model in model_list:
                self.parameters["models"] = [model]
                print(f"\n--- Training with model '{model}' on dataset '{ds}' ---")
                self.train()
                if explain_method:
                    self.explain(
                        model_name=model, method=explain_method, num_samples=num_samples
                    )
                if memorize:
                    self.memorize()
                all_results[f"{ds}"] = {
                    "result_dict": self.result_dict,
                    # "trained_model": self.trained_models,
                }
                print("Result:", self.result_dict)
                # print("Trained Models:", self.trained_models)
        print("\n=== All Results ===")
        for key, value in all_results.items():
            print(f"{key}: {value}")

        results_path = os.path.join(project_root, "results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"All results saved to {results_path}")
        return all_results
