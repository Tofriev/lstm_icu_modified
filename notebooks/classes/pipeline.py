from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
import hashlib
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns


class Pipeline(object):
    def __init__(self, variables, parameters, show=False):
        self.variables = variables
        self.parameters = parameters
        self.show = show

    def prepare_data(self):
        self.DataManager = DatasetManager(
            variables=self.variables, parameters=self.parameters
        )
        self.DataManager.load_data()  # DataManager.data holds all data afterwards

        # if self.show:
        #     seq, label = self.sequences['train'][0]
        #     print(seq.shape)
        #     print(label.shape)
        #     print(dict(
        #         sequence=torch.tensor(seq, dtype=torch.float32),
        #         label=torch.tensor(label).long(),
        #         ))

    def train(self):
        trainer = Trainer(self.parameters)
        if self.parameters[
            "dataset_type"
        ] == "mimic_tudd_fract" and self.parameters.get("fractional_steps"):
            print("Training fractional")
            self.result_dict = trainer.train_fractional(self.sequences)

        elif self.parameters["dataset_type"] == "mimic_mimic":
            self.result_dict = trainer.train(
                self.DataManager.data["mimic"]["sequences_train"],
                self.DataManager.data["mimic"]["sequences_test"],
            )

        elif self.parameters["dataset_type"] == "tudd_tudd":
            self.result_dict = trainer.train(
                self.sequences["tudd"]["train"], self.sequences["tudd"]["test"]
            )
        elif self.parameters["dataset_type"] == "mimic_tudd":
            self.result_dict = trainer.train(
                self.sequences["mimic"]["train"], self.sequences["tudd"]["test"]
            )
        elif self.parameters["dataset_type"] == "tudd_mimic":
            self.result_dict = trainer.train(
                self.sequences["tudd"]["train"], self.sequences["mimic"]["test"]
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
