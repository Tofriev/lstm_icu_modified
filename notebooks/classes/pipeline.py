import os
import pickle
import hashlib
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
from classes.explainer import ExtractSHAPExplainer
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

    def make_pkl_datasets(self):
        """
        Creates pickle files for individual dataset splits using two different scalers.
        
        First, it creates mimic and tudd splits (and combined splits) using a DatasetManager instance
        that loads mimic first (thus obtaining the mimic scaler). These are saved with a "_mimic_scaler" suffix.
        
        Then, it creates a new DatasetManager instance and manually loads TUDD first (so that the TUDD scaler is used)
        followed by mimic. The resulting splits are saved with a "_tudd_scaler" suffix.
        """
        # ----------------------------
        # Instance 1: Using mimic scaler
        # ----------------------------
        print("Generating pickle datasets using mimic scaler (instance 1)...")
        print("MIMIC")
        dm1 = DatasetManager(variables=self.variables, parameters=self.parameters)
        dm1.data["mimic"] = {}
        dm1.load_mimic()
        dm1.preprocess("mimic")
        dm1.compute_patient_statistics("mimic")
        #dm1.plot_time_missingness_heatmap("mimic")
        feature_names = dm1.feature_names
        numerical_features = dm1.numerical_features
        print('TUDD')
        dm1.data["tudd"] = {}
        dm1.load_tudd()
        dm1.preprocess("tudd")
        dm1.compute_patient_statistics("tudd")
        #dm1.plot_time_missingness_heatmap("tudd")
        
        # Retrieve splits from dm1
        mimic_train_1 = dm1.data["mimic"]["sequences_train"]
        mimic_test_1 = dm1.data["mimic"]["sequences_test"]
        tudd_train_1 = dm1.data["tudd"]["sequences_train"]
        tudd_test_1 = dm1.data["tudd"]["sequences_test"]

        # Define file paths (with a mimic scaler suffix)
        mimic_train_file_1 = os.path.join(self.preprocessed_dir, "mimic_train_mimic_scaler.pkl")
        mimic_test_file_1 = os.path.join(self.preprocessed_dir, "mimic_test_mimic_scaler.pkl")
        tudd_train_file_1 = os.path.join(self.preprocessed_dir, "tudd_train_mimic_scaler.pkl")
        tudd_test_file_1 = os.path.join(self.preprocessed_dir, "tudd_test_mimic_scaler.pkl")

        # Save individual splits with mimic scaler
        with open(mimic_train_file_1, "wb") as f:
            pickle.dump({"data": mimic_train_1, "scaler": dm1.scaler, "feature_names":feature_names, "numerical_features": numerical_features}, f)
        with open(mimic_test_file_1, "wb") as f:
            pickle.dump({"data": mimic_test_1}, f)
        with open(tudd_train_file_1, "wb") as f:
            pickle.dump({"data": tudd_train_1}, f)
        with open(tudd_test_file_1, "wb") as f:
            pickle.dump({"data": tudd_test_1}, f)

        # ----------------------------
        # Instance 2: Using tudd scaler
        # ----------------------------
        print("Generating pickle datasets using tudd scaler (instance 2)...")
        print("TUDD")
        dm2 = DatasetManager(variables=self.variables, parameters=self.parameters)
        # Instead of using load_data(), manually load TUDD first to obtain the tudd scaler.
        dm2.data["tudd"] = {}
        dm2.load_tudd()
        dm2.preprocess("tudd")
        dm2.compute_patient_statistics("tudd")
        dm2.plot_time_missingness_heatmap("tudd")

        feature_names = dm2.feature_names
        numerical_features = dm2.numerical_features
        # print("MIMIC")
        # # Now load mimic after TUDD so that the scaler remains from TUDD.
        # dm2.data["mimic"] = {}
        # dm2.load_mimic()
        # dm2.preprocess("mimic")

        # # Retrieve splits from dm2
        # mimic_train_2 = dm2.data["mimic"]["sequences_train"]
        # mimic_test_2 = dm2.data["mimic"]["sequences_test"]
        tudd_train_2 = dm2.data["tudd"]["sequences_train"]
        tudd_test_2 = dm2.data["tudd"]["sequences_test"]

        # # Define file paths (with a tudd scaler suffix)
        # mimic_train_file_2 = os.path.join(self.preprocessed_dir, "mimic_train_tudd_scaler.pkl")
        # mimic_test_file_2 = os.path.join(self.preprocessed_dir, "mimic_test_tudd_scaler.pkl")
        tudd_train_file_2 = os.path.join(self.preprocessed_dir, "tudd_train_tudd_scaler.pkl")
        tudd_test_file_2 = os.path.join(self.preprocessed_dir, "tudd_test_tudd_scaler.pkl")

        # # Save individual splits with tudd scaler
        # with open(mimic_train_file_2, "wb") as f:
        #     pickle.dump({"data": mimic_train_2, "scaler": dm2.scaler, "feature_names":feature_names, "numerical_features": numerical_features}, f)
        # with open(mimic_test_file_2, "wb") as f:
        #     pickle.dump({"data": mimic_test_2}, f)
        with open(tudd_train_file_2, "wb") as f:
            pickle.dump({"data": tudd_train_2, "scaler": dm2.scaler, "feature_names":feature_names, "numerical_features": numerical_features}, f)
        with open(tudd_test_file_2, "wb") as f:
            pickle.dump({"data": tudd_test_2}, f)

    def prepare_data(self):
        dt = self.parameters["dataset_type"]
        if isinstance(dt, list) and len(dt) == 1:
            dt = dt[0]

        if "fract" not in dt:
            # Determine which files to load based on dataset type
            if dt == "mimic_mimic":
                train_file = os.path.join(self.preprocessed_dir, "mimic_train_mimic_scaler.pkl")
                test_file = os.path.join(self.preprocessed_dir, "mimic_test_mimic_scaler.pkl")
            elif dt == "tudd_tudd":
                train_file = os.path.join(self.preprocessed_dir, "tudd_train_tudd_scaler.pkl")
                test_file = os.path.join(self.preprocessed_dir, "tudd_test_tudd_scaler.pkl")
            elif dt == "mimic_tudd":
                train_file = os.path.join(self.preprocessed_dir, "mimic_train_mimic_scaler.pkl")
                test_file = os.path.join(self.preprocessed_dir, "tudd_test_mimic_scaler.pkl")
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
            #print("Loading preprocessed train and test splits from cache...")
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
            print(f'SCALER FROM DATAMANAGER {self.scaler}')
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
            raise ValueError("No fractional indices found. Make sure 'fract' is in your dataset_type.")

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
        fractional_indices = self.DataManager.data["fractional_indices"]

        for fraction_size in sorted(fractional_indices.keys()):
            print(f"\nTraining with fraction_size = {fraction_size} training samples...")
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
            del fraction_data
            
    def visualize_fraction_results(self, save_path="fraction_results.png"):
        if not hasattr(self, "fraction_results") or not self.fraction_results:
            raise ValueError("No fractional experiment results to visualize.")

        print(f"fraction results: {self.fraction_results}")

        fractions = sorted(self.fraction_results.keys())

        # Get the list of models dynamically from the first available fraction entry
        first_fraction = fractions[0]
        models = list(self.fraction_results[first_fraction].keys())

        custom_colors = {
        "lstm_static": "red",
        "mutli_channel_lstm_static": "blue"
        }

        custom_labels = {
            "lstm_static": "Lstm",
            "mutli_channel_lstm_static": "MultiChannelLstm"
        }

        plt.figure(figsize=(8, 6))

        for model_name in models:
            aurocs = [
                self.fraction_results[f][model_name][0]["test_auroc"] for f in fractions
            ]
            color = custom_colors.get(model_name, None)
            label = custom_labels.get(model_name, model_name)
            plt.plot(fractions, aurocs, label=label, color=color)

        plt.xlabel("Number of Training Samples")
        plt.ylabel("AUROC")
        plt.title("Model Performance vs. Number of Training Samples")
        plt.ylim(0.6, 0.9)
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

            model_names = "_".join(self.parameters["models"])
            filename = f"results_{dt}_{model_names}.json"
            self.results_path = os.path.join(project_root, filename)

            # Save results
            with open(self.results_path, "w") as f:
                json.dump(self.fraction_results if "fract" in dt else self.result_dict, f, indent=4)

        else:
            # NORMAL TRAINING
            self.train()
            model_names = "_".join(self.parameters["models"])
            filename = f"results_{dt}_{model_names}.json"
            self.results_path = os.path.join(project_root, filename)

            # Save results
            with open(self.results_path, "w") as f:
                json.dump(self.fraction_results if "fract" in dt else self.result_dict, f, indent=4)

    def explain(self, model_name, method, num_samples=1000, feature_to_explain=None, feature_type=None, feature_idx=None, sample_idx=None):
        model = self.trained_models[model_name]
        feature_names = self.feature_names

        explainer = ExtractSHAPExplainer(model=model, feature_names=feature_names)
        if self.scaler is not None:
                print("SCALER FOUND") 
        else:
            raise ValueError("No Scaler")

        explainer.explain(
            sample_idx = sample_idx,
            sequences=self.test_sequences,  
            model_name=model_name,
            num_samples=num_samples,
            method=method,
            feature_idx=feature_idx,
            feature_to_explain=feature_to_explain,
            scaler=self.scaler,
            batch_size=10,
            dataset_type = self.parameters['dataset_type'] 
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

    def run_all(self, model_list, memorize=False, explain_method=None, num_samples=1000):
        all_results = {}
        # Loop over each dataset type.
        for ds in self.dataset_types:
            # Update parameters for the current dataset.
            self.parameters["dataset_type"] = ds
            for model in model_list:
                # Update model list for the current experiment.
                self.parameters["models"] = [model]
                print(f"\n=== Running experiment for dataset '{ds}' with model '{model}' ===")
                # Run the complete experiment (this will prepare data, train the model, and save results).
                self.run_experiment()
                # Prepare the result dictionary for this combination.
                combination_key = f"{ds}_{model}"
                combination_result = {
                    "result_dict": self.result_dict
                }
                # If an explanation method is provided, run explain and store the returned SHAP scores.
                if explain_method:
                    # It is assumed that the explain() method has been modified to return the SHAP scores.
                    shap_scores = self.explain(model_name=model, method=explain_method, num_samples=num_samples)
                    combination_result["shap_scores"] = shap_scores
                # Optionally memorize the parameters and results.
                if memorize:
                    self.memorize()
                # Save the result for the current combination.
                all_results[combination_key] = combination_result
                print("Result for", combination_key, ":", combination_result["result_dict"])
        # Generate model name part for the output filename.
        model_part = "_".join(model_list)
        filename = f"results_multiple_{model_part}.json"
        self.results_path = os.path.join(project_root, filename)
        # Save the complete results to file.
        with open(self.results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        return all_results
