import os
import pickle
import hashlib
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
from classes.explainer import SHAPExplainer
import sys
import json
import random
import gc
import torch
from copy import deepcopy

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
        self.results_path = os.path.join(project_root, "results.json")

    def prepare_data(self):
        dt = self.parameters["dataset_type"]
        if isinstance(dt, list) and len(dt) == 1:
            dt = dt[0]

        if "fract" not in dt:
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
        else:
            train_file = "None"
            test_file = "None"
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
            self.DataManager.load_data()
            # Extract metadata from the full dataset manager.
            self.feature_names = self.DataManager.feature_names
            self.scaler = self.DataManager.scaler
            self.numerical_features = self.DataManager.numerical_features

            if "fract" not in dt:
                # For consistency, build a dictionary of preprocessed splits.
                # Here we assume that self.DataManager.data has keys like "mimic", "tudd", "combined"
                # each containing "sequences_train" and "sequences_test".
                if dt in [
                    "mimic_mimic",
                    "mimic_tudd",
                    "mimic_combined",
                    "mimic_fract",
                    "mimic_tudd_fract",
                ]:
                    train_split = self.DataManager.data["mimic"]["sequences_train"]
                elif dt in [
                    "tudd_tudd",
                    "tudd_mimic",
                    "tudd_combined",
                ]:
                    train_split = self.DataManager.data["tudd"]["sequences_train"]
                elif dt.startswith("combined"):
                    train_split = self.DataManager.data["combined"]["sequences_train"]
                else:
                    raise ValueError(f"Unexpected dataset type for train split: {dt}")

                if dt in ["mimic_mimic", "tudd_mimic", "combined_mimic", "mimic_fract"]:
                    test_split = self.DataManager.data["mimic"]["sequences_test"]
                elif dt in [
                    "tudd_tudd",
                    "mimic_tudd",
                    "combined_tudd",
                ]:
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

    def train_fractional_experiments(self):
        dt = self.parameters["dataset_type"]
        if isinstance(dt, list):
            dt = dt[0]
        if "fractional_indices" not in self.DataManager.data:
            raise ValueError(
                "No fractional indices found. Make sure 'fract' is in your dataset_type."
            )

        if dt == "mimic_tudd_fract":
            test_data = self.DataManager.data["tudd"]["sequences_test"]
            tudd_train = self.DataManager.data["train_fractions"]
            mimic_all = self.DataManager.data["mimic_train_all"]
        elif dt == "mimic_fract":
            test_data = self.DataManager.data["mimic"]["sequences_test"]
            mimic_train = self.DataManager.data["train_fractions"]
        elif dt == "tudd_fract":
            test_data = self.DataManager.data["tudd"]["sequences_test"]
            tudd_train = self.DataManager.data["train_fractions"]

        self.fraction_results = {}
        # self.fraction_models = {}

        fractional_indices = self.DataManager.data["fractional_indices"]

        for fraction_size in sorted(fractional_indices.keys()):
            print(
                f"\nTraining with fraction_size = {fraction_size} training samples..."
            )

            # build tudd from indices
            idx_list = fractional_indices[fraction_size]
            if dt == "mimic_fract":
                fraction_data = [mimic_train[i] for i in idx_list]
            else:
                fraction_data = [tudd_train[i] for i in idx_list]

            if "mimic_tudd_fract" in self.parameters["dataset_type"]:
                fraction_data = mimic_all + fraction_data
                if self.parameters.get("shuffle_mimic_tudd_fract"):
                    random.shuffle(fraction_data)

            print(f"Length of fraction_data: {len(fraction_data)}")

            local_trainer = Trainer(self.parameters)
            result, model = local_trainer.train(fraction_data, test_data)

            self.fraction_results[fraction_size] = deepcopy(result)
            # self.fraction_models[fraction_size] = deepcopy(model)

            del fraction_data
            # gc.collect()
            # torch.mps.empty_cache()

    def visualize_fraction_results(self, save_path="fraction_results.png"):
        if not hasattr(self, "fraction_results") or not self.fraction_results:
            raise ValueError("No fractional experiment results to visualize.")

        print(f"fraction results: {self.fraction_results}")

        fractions = sorted(self.fraction_results.keys())

        # Get the list of models dynamically from the first available fraction entry
        first_fraction = fractions[0]
        models = list(self.fraction_results[first_fraction].keys())

        plt.figure(figsize=(8, 6))

        for model_name in models:
            aurocs = [
                self.fraction_results[f][model_name][0]["test_auroc"] for f in fractions
            ]
            plt.plot(fractions, aurocs, marker="o", label=model_name)

        plt.xlabel("Number of Training Samples")
        plt.ylabel("AUROC")
        plt.title("Model Performance vs. Number of Training Samples")

        if len(models) > 1:
            plt.legend(title="Models")

        plt.grid(True)
        plt.savefig(save_path)

        if self.show:
            plt.show()

    def run_experiment(self):
        self.prepare_data()

        dt = self.parameters["dataset_type"]
        if isinstance(dt, list) and len(dt) == 1:
            dt = dt[0]

        if isinstance(dt, str) and "fract" in dt:
            # FRACTIONAL TRAINING
            self.train_fractional_experiments()
            self.visualize_fraction_results()

            with open(self.results_path, "w") as f:
                json.dump(self.fraction_results, f, indent=4)
            print(f"Fraction experiment results saved to {self.results_path}")

        else:
            # NORMAL TRAINING
            self.train()
            with open(self.results_path, "w") as f:
                json.dump(self.result_dict, f, indent=4)
            print(f"Results saved to {self.results_path}")

    def explain(self, model_name, method, num_samples=1000, feature_to_explain = None):
        explainer = SHAPExplainer(
            model=self.trained_models[model_name],
            feature_names=self.feature_names,
        )
        explainer.explain(
            self.test_sequences,
            method,
            num_samples,
            self.scaler,
            self.numerical_features,
            feature_to_explain,
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

        self.results_path = os.path.join(project_root, "results.json")
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"All results saved to {self.results_path}")
        return all_results
