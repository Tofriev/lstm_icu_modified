from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
import hashlib
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.explainer import SHAPExplainer
import pickle


class Pipeline(object):
    def __init__(self, variables, parameters, show=False, new_data=True):
        self.variables = variables
        self.parameters = parameters
        self.show = show
        self.cache_file = self.parameters.get("cache_file", "preprocessed_data.pkl")
        self.force_preprocess = new_data

    def prepare_data(self):
        if os.path.exists(self.cache_file) and not self.force_preprocess:
            print("Loading preprocessed data from cache...")
            with open(self.cache_file, "rb") as f:
                self.DataManager = pickle.load(f)
        else:
            print("Preprocessing data from scratch...")
            self.DataManager = DatasetManager(
                variables=self.variables, parameters=self.parameters
            )
            self.DataManager.load_data()  # Fills self.DataManager.data
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.DataManager, f)
            print("Preprocessed data saved to cache.")

        self.feature_names = self.DataManager.feature_names
        self.scaler = self.DataManager.scaler
        self.numerical_features = self.DataManager.numerical_features

    def train(self):
        trainer = Trainer(self.parameters)
        dt = self.parameters["dataset_type"]
        # Choose training and testing sets based on the dataset type.
        if dt == "mimic_mimic":
            train_data = self.DataManager.data["mimic"]["sequences_train"]
            test_data = self.DataManager.data["mimic"]["sequences_test"]
        elif dt == "tudd_tudd":
            train_data = self.DataManager.data["tudd"]["sequences_train"]
            test_data = self.DataManager.data["tudd"]["sequences_test"]
        elif dt == "mimic_tudd":
            train_data = self.DataManager.data["mimic"]["sequences_train"]
            test_data = self.DataManager.data["tudd"]["sequences_test"]
        elif dt == "tudd_mimic":
            train_data = self.DataManager.data["tudd"]["sequences_train"]
            test_data = self.DataManager.data["mimic"]["sequences_test"]
        elif dt == "mimic_combined":  # Train on mimic only; test on combined
            train_data = self.DataManager.data["mimic"]["sequences_train"]
            test_data = self.DataManager.data["combined"]["sequences_test"]
        elif dt == "tudd_combined":  # Train on tudd only; test on combined
            train_data = self.DataManager.data["tudd"]["sequences_train"]
            test_data = self.DataManager.data["combined"]["sequences_test"]
        elif dt == "combined_mimic":  # Train on combined; test on mimic
            train_data = self.DataManager.data["combined"]["sequences_train"]
            test_data = self.DataManager.data["mimic"]["sequences_test"]
        elif dt == "combined_tudd":  # Train on combined; test on tudd
            train_data = self.DataManager.data["combined"]["sequences_train"]
            test_data = self.DataManager.data["tudd"]["sequences_test"]
        elif dt == "combined_combined":  # Both train and test on combined splits
            train_data = self.DataManager.data["combined"]["sequences_train"]
            test_data = self.DataManager.data["combined"]["sequences_test"]
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
