from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
import hashlib
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.explainer import SHAPExplainer
import pickle
import matplotlib.pyplot as plt
import hashlib
from copy import deepcopy
import random
import gc
import torch
from classes.data_module import HybridDataModule


class Pipeline(object):
    def __init__(self, variables, parameters, show=False, new_data=True):
        self.variables = variables
        self.parameters = parameters
        self.show = show
        # self.cache_file = self.parameters.get("cache_file", "preprocessed_data.pkl")
        self.force_preprocess = new_data

    def prepare_data(self):
        # if os.path.exists(self.cache_file) and not self.force_preprocess:
        #     print("Loading preprocessed data from cache...")
        #     with open(self.cache_file, "rb") as f:
        #         self.DataManager = pickle.load(f)

        print("Preprocessing data from scratch...")
        self.DataManager = DatasetManager(
            variables=self.variables, parameters=self.parameters
        )
        self.DataManager.load_data()

        self.feature_names = self.DataManager.feature_names
        self.scaler = self.DataManager.scaler
        self.numerical_features = self.DataManager.numerical_features

    def train(self):
        self.trainer = Trainer(self.parameters)
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
        # elif "fract" in dt:
        #     print("Training fractional experiments...")
        #     self.train_fractional_experiments()
        else:
            raise ValueError(f"Dataset type {dt} is not supported.")
        print("Training...")
        self.result_dict, self.trained_models = self.trainer.train(
            train_data, test_data
        )
        self.test_sequences = test_data

    def train_fractional_experiments(self):
        if "fractional_indices" not in self.DataManager.data:
            raise ValueError(
                "No fractional indices found. Make sure 'fract' is in your dataset_type."
            )

        # Always test on TUDD test
        test_data = self.DataManager.data["tudd"]["sequences_test"]
        tudd_all = self.DataManager.data["tudd_train_all"]

        mimic_all = []
        if "mimic_tudd_fract" in self.parameters["dataset_type"]:
            mimic_all = self.DataManager.data["mimic_train_all"]

        self.fraction_results = {}
        fractional_indices = self.DataManager.data["fractional_indices"]

        for fraction_size in sorted(fractional_indices.keys()):
            print(
                f"\nTraining with fraction_size = {fraction_size} training samples..."
            )

            # build fraction from tudd
            idx_list = fractional_indices[fraction_size]
            fraction_data = [tudd_all[i] for i in idx_list]

            if "mimic_tudd_fract" in self.parameters["dataset_type"]:
                fraction_data = mimic_all + fraction_data
                random.shuffle(fraction_data)

            print(f"Length of fraction_data: {len(fraction_data)}")

            # ---------------------------------------------------------------------
            # (A) Write the fraction_data to disk so we don't hold it all in memory
            # For example:
            fraction_dir = "data/fractional"
            os.makedirs(fraction_dir, exist_ok=True)

            # Store the file in this directory
            fraction_file = os.path.join(
                fraction_dir, f"fraction_data_{fraction_size}.pkl"
            )
            write_sequences_to_file(fraction_data, fraction_file)
            # ---------------------------------------------------------------------

            local_trainer = Trainer(self.parameters)

            # ---------------------------------------------------------------------
            # (B) Build a HybridDataModule with the fraction_data on disk,
            #     but normal in-memory test_data

            batch_size = self.parameters["model_parameters"][
                self.parameters["models"][0]
            ]["batch_size"]
            data_module = HybridDataModule(
                train_file=fraction_file,
                test_sequences=test_data,
                batch_size=batch_size,
            )

            # (C) Train using local_trainer.train_with_datamodule(...) or similar
            # or if your Trainer class expects train/test sequences directly,
            # you might adapt it to accept a DataModule. For example:
            result, model = local_trainer.train_with_datamodule(data_module)
            # ---------------------------------------------------------------------

            self.fraction_results[fraction_size] = deepcopy(result)

            # cleanup
            del fraction_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    # TODO: needs to be fixed for dict access
    # def memorize_fraction_results(self, file_path="parameters_fraction_results.csv"):
    #     fieldnames = ["fraction", "auroc"] + list(self.parameters.keys())
    #     with open(file_path, mode="a", newline="") as file:
    #         writer = csv.DictWriter(file, fieldnames=fieldnames)
    #         if file.tell() == 0:
    #             writer.writeheader()
    #         for fraction, result in self.fraction_results.items():
    #             row = {"fraction": fraction, "auroc": result["auroc"]}
    #             row.update(self.parameters)
    #             writer.writerow(row)

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


def write_sequences_to_file(sequence_list, file_path):
    with open(file_path, "wb") as f:
        for seq in sequence_list:
            pickle.dump(seq, f)  # seq is (features, label)
