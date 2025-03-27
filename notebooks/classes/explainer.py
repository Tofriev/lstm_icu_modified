import os
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import shap

###############################################################################
# Base class with common functionality
###############################################################################
class SHAPExplainerBase:
    def __init__(self, model, feature_names=None):
        self.orig_model = model       # original (trained) model
        self.model = model            # may be wrapped if needed
        self.feature_names = feature_names

        # Variables that will be set after extraction or loading:
        self.shap_values = None       # structure: [[seq_class0, static_class0], [seq_class1, static_class1]]
        self.test_seq_data_np = None  # test sequential data (num_samples, time_steps, n_seq_features)
        self.test_static_data_np = None  # test static data (num_samples, n_static_features)
        self.test_labels_np = None    # test labels (num_samples,)

    def pad_sequences(self, sequences, max_len=None):
        if max_len is None:
            max_len = max(seq.shape[0] for seq in sequences)
        n_features = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
        for i, seq in enumerate(sequences):
            length = seq.shape[0]
            padded[i, :length, :] = seq
        return padded

    def replicate_static(self, shap_array, time_steps):
        """
        Replicate static SHAP values along the time dimension.
        shap_array: (num_samples, n_static_features)
        Return: (num_samples, time_steps, n_static_features)
        """
        return np.repeat(shap_array[:, np.newaxis, :], time_steps, axis=1)

    def is_probability_output(self, sample_input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(*sample_input).cpu().numpy().flatten()
            return np.all((output >= 0.0) & (output <= 1.0))

    ############################################################################
    # PLOTTING FUNCTIONS (using only class 1 i.e. "Non Survival")
    ############################################################################
    def plot_shap_heatmap_mean_abs(self):
        """
        Plot mean(|SHAP|) across time steps and features for Non Survival (class=1).
        """
        if self.shap_values is None:
            raise ValueError("SHAP values have not been set. Run extraction or load from file first.")

        shap_seq = self.shap_values[1][0]    # sequential branch
        shap_static = self.shap_values[1][1]   # static branch
        ts_length = self.test_seq_data_np.shape[1]
        shap_static_repl = self.replicate_static(shap_static, ts_length)
        shap_combined = np.concatenate([shap_seq, shap_static_repl], axis=2)
        
        shap_abs = np.abs(shap_combined)
        mean_abs = shap_abs.mean(axis=0)  # shape: (time_steps, total_features)
        overall_mean = mean_abs.mean(axis=0)  # shape: (total_features,)
        sorted_indices = np.argsort(overall_mean)[::-1]
        sorted_feature_names = [self.feature_names[i] for i in sorted_indices]
        sorted_mean_abs = mean_abs[:, sorted_indices].T  # shape: (total_features, time_steps)

        time_steps = sorted_mean_abs.shape[1]
        df = pd.DataFrame(
            sorted_mean_abs,
            index=sorted_feature_names,
            columns=[f"Time {i}" for i in range(time_steps)]
        )
        df_clipped = df.clip(lower=1e-6)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_clipped,
            cmap="Reds",
            annot=False,
            fmt=".2f",
            cbar_kws={"label": "Mean Absolute SHAP Score (Log Scale)"},
            norm=LogNorm(vmin=df_clipped.min().min(), vmax=df_clipped.max().max()),
        )
        plt.title("SHAP Heatmap (Mean Absolute SHAP) [Non Survival] [Log Scale]")
        plt.xlabel("Time Steps")
        plt.ylabel("Features (Sorted by Overall Importance)")
        plt.tight_layout()
        plt.show()

    def plot_single_feature_time_shap(self, sample_idx, feature_to_explain, scaler=None, input_type='sequential', feature_idx=None):
        """
        Plot a single feature's values over time together with the corresponding Non-Survival SHAP values.
        Also annotate the plot with predicted and actual class.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values have not been set. Run extraction or load from file first.")
        
        # Select data based on input_type.
        if input_type == 'sequential':
            branch_idx = 0
            data_array = self.test_seq_data_np
            ts_length = data_array.shape[1]
            raw_values = data_array[sample_idx, :, feature_idx]
        elif input_type == 'static':
            branch_idx = 1
            data_array = self.test_static_data_np
            ts_length = self.test_seq_data_np.shape[1]
            raw_value = data_array[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")

        if scaler is not None:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        else:
            feature_values = raw_values

        time_steps = np.arange(ts_length)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y", labelcolor="black")

        shap_vals = self.shap_values[1][branch_idx]
        if input_type == 'sequential':
            shap_vals_non_surv = shap_vals[sample_idx, :, feature_idx]
        else:
            shap_val_single = shap_vals[sample_idx, feature_idx]
            shap_vals_non_surv = np.repeat(shap_val_single, ts_length)

        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non-Survival]")
        ax2.plot(time_steps, shap_vals_non_surv, color="red", linestyle="--", linewidth=1.5,
                 alpha=0.6, label="SHAP (Non Survival)")
        min_val = np.min(shap_vals_non_surv)
        max_val = np.max(shap_vals_non_surv)
        margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        ax2.set_ylim(min_val - margin, max_val + margin)
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.subplots_adjust(bottom=0.25)
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])  # [left, bottom, width, height] in figure coords
        ax_annot.axis("off")

        actual_label_val = self.test_labels_np[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"

        self.orig_model.eval()
        with torch.no_grad():
            seq_t = torch.tensor(self.test_seq_data_np[sample_idx : sample_idx+1]).float()
            stat_t = torch.tensor(self.test_static_data_np[sample_idx : sample_idx+1]).float()
            out = self.orig_model(seq_t, stat_t).squeeze()
            predicted_class_str = ""
            predicted_prob = 0.0
            if out.dim() == 0:
                p_non_surv = float(out.item())
                p_surv = 1.0 - p_non_surv
                if p_non_surv >= 0.5:
                    predicted_class_str = "Non Survival"
                    predicted_prob = p_non_surv
                else:
                    predicted_class_str = "Survival"
                    predicted_prob = p_surv
            elif out.shape[0] == 2:
                probs = torch.softmax(out, dim=0)
                p_surv = probs[0].item()
                p_non_surv = probs[1].item()
                if p_surv >= p_non_surv:
                    predicted_class_str = "Survival"
                    predicted_prob = p_surv
                else:
                    predicted_class_str = "Non Survival"
                    predicted_prob = p_non_surv

        text_str = (
            f"Predicted: {predicted_class_str} (P={predicted_prob:.2f})\n"
            f"Actual: {actual_label_str}"
        )
        # fig.text(
        #     0.5, 0.01, text_str,
        #     ha='center', va='bottom', fontsize=10,
        #     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        # )
        ax_annot.text(0.5, 0.5, text_str,
                  ha='center', va='center', fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        #plt.tight_layout()
        plt.show()


###############################################################################
# Child class that extracts new SHAP values and saves them (with test data)
###############################################################################
class ExtractSHAPExplainer(SHAPExplainerBase):
    def extract_shap_values(self, sequences, num_samples, batch_size=10, background_pct=0.1, random_seed=42):
        """
        Extract SHAP values from the provided sequences. The input `sequences` is a list of tuples:
            (sequential_features, static_features, label)
        This method also stores the test data arrays.
        """
        print("Extracting SHAP values...")

        seq_list = [s[0] for s in sequences]
        all_seq_data_np = self.pad_sequences(seq_list)
        all_static_data_np = np.array([s[1] for s in sequences])
        all_labels_np = np.array([s[2] for s in sequences])
        total_samples = all_seq_data_np.shape[0]

        np.random.seed(random_seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        num_background = int(background_pct * total_samples)
        background_idx = indices[:num_background]
        test_idx = indices[num_background:]

        background_seq_np = all_seq_data_np[background_idx]
        background_static_np = all_static_data_np[background_idx]
        test_seq_np = all_seq_data_np[test_idx]
        test_static_np = all_static_data_np[test_idx]
        test_labels_np = all_labels_np[test_idx]

        if num_samples > len(test_seq_np):
            num_samples = len(test_seq_np)
        test_seq_np = test_seq_np[:num_samples]
        test_static_np = test_static_np[:num_samples]
        test_labels_np = test_labels_np[:num_samples]

        self.test_seq_data_np = test_seq_np
        self.test_static_data_np = test_static_np
        self.test_labels_np = test_labels_np

        background_seq_tensor = torch.tensor(background_seq_np).float()
        background_static_tensor = torch.tensor(background_static_np).float()
        explainer = shap.DeepExplainer(self.model, [background_seq_tensor, background_static_tensor])

        shap_values_batches = []
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch_seq = torch.tensor(test_seq_np[i : i + batch_size]).float()
            batch_static = torch.tensor(test_static_np[i : i + batch_size]).float()
            shap_values_batch = explainer.shap_values([batch_seq, batch_static])
            shap_values_batches.append(shap_values_batch)

        aggregated_shap_values = []
        num_outputs = len(shap_values_batches[0])
        for out_idx in range(num_outputs):
            aggregated_for_output = []
            num_branches = len(shap_values_batches[0][out_idx])
            for branch_idx in range(num_branches):
                arrays = [batch[out_idx][branch_idx] for batch in shap_values_batches]
                if arrays[0].ndim == 3:
                    max_time = max(a.shape[1] for a in arrays)
                    padded_arrays = []
                    for arr in arrays:
                        if arr.shape[1] < max_time:
                            pad_width = ((0,0),(0,max_time - arr.shape[1]),(0,0))
                            arr_padded = np.pad(arr, pad_width, mode='constant')
                            padded_arrays.append(arr_padded)
                        else:
                            padded_arrays.append(arr)
                    aggregated = np.concatenate(padded_arrays, axis=0)
                elif arrays[0].ndim == 2:
                    aggregated = np.concatenate(arrays, axis=0)
                else:
                    raise ValueError("Unexpected SHAP array dimension.")
                aggregated_for_output.append(aggregated)
            aggregated_shap_values.append(aggregated_for_output)

        self.shap_values = aggregated_shap_values

    def save_shap_values(self, model_name, num_samples, shap_dir="SHAP_scores"):
        """
        Save the SHAP values and test data to a JSON file.
        The file name will be "{model_name}_{num_samples}.json" in folder shap_dir.
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values to save. Run extract_shap_values() first.")
        os.makedirs(shap_dir, exist_ok=True)
        filename = f"{model_name}_{num_samples}.json"
        full_path = os.path.join(shap_dir, filename)
        data_to_save = {
            "shap_values": {
                "label_0": {
                    "sequential": self.shap_values[0][0].tolist(),
                    "static": self.shap_values[0][1].tolist()
                },
                "label_1": {
                    "sequential": self.shap_values[1][0].tolist(),
                    "static": self.shap_values[1][1].tolist()
                }
            },
            "test_data": {
                "seq": self.test_seq_data_np.tolist(),
                "static": self.test_static_data_np.tolist(),
                "labels": self.test_labels_np.tolist()
            }
        }
        with open(full_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"SHAP values and test data saved to {full_path}")

    def explain(self,sequences, model_name, num_samples=1000, batch_size=10, save_shap=True, method="plot_single_feature_time_shap", **kwargs):
        """
        Main method: extract new SHAP values from the given sequences, save them if desired,
        and then run a plotting method.
        """
        self.extract_shap_values(sequences, num_samples, batch_size)
        if save_shap:
            self.save_shap_values(model_name, num_samples)
        if method == "plot_shap_heatmap_mean_abs":
            self.plot_shap_heatmap_mean_abs()
        elif method == "plot_single_feature_time_shap":
            # Expecting kwargs: feature_idx, feature_to_explain, scaler, input_type
            self.plot_single_feature_time_shap(**kwargs)
        else:
            raise ValueError(f"Unknown plotting method: {method}")


###############################################################################
# Child class that loads existing SHAP values and test data from file.
###############################################################################
class LoadSHAPExplainer(SHAPExplainerBase):
    def load_shap_values(self, model_name, num_samples, shap_dir="SHAP_scores"):
        """
        Load the SHAP values and test data from a JSON file.
        """
        filename = f"{model_name}_{num_samples}.json"
        full_path = os.path.join(shap_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        with open(full_path, "r") as f:
            data = json.load(f)
        shap_data = data["shap_values"]
        self.shap_values = [
            [np.array(shap_data["label_0"]["sequential"]), np.array(shap_data["label_0"]["static"])],
            [np.array(shap_data["label_1"]["sequential"]), np.array(shap_data["label_1"]["static"])]
        ]
        test_data = data["test_data"]
        self.test_seq_data_np = np.array(test_data["seq"])
        self.test_static_data_np = np.array(test_data["static"])
        self.test_labels_np = np.array(test_data["labels"])
        print(f"Loaded SHAP values and test data from {full_path}")
