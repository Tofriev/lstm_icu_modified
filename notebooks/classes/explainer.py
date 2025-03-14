import shap
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class SHAPExplainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.shap_values = None
        self.test_seq_data_np = None   # (num_samples, time_steps, n_seq_features)
        self.test_static_data_np = None  # (num_samples, n_static_features)
        self.scale_factors = None
        self.feature_names = feature_names

    def pad_sequences(self, sequences, max_len=None):
        """
        Pad a list of 2D arrays (time_steps, n_features) with zeros so all have the same number of time steps.
        """
        if max_len is None:
            max_len = max(seq.shape[0] for seq in sequences)
        n_features = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
        for i, seq in enumerate(sequences):
            length = seq.shape[0]
            padded[i, :length, :] = seq
        return padded

    def extract_shap_values(self, sequences, num_samples, batch_size=10, background_pct=0.1, random_seed=42):
       
        print("Extracting SHAP values...")

        #  sequential and static parts
        seq_list = [s[0] for s in sequences]
        all_seq_data_np = self.pad_sequences(seq_list)  # shape: (total_samples, max_time_steps, n_seq_features)
        all_static_data_np = np.array([s[1] for s in sequences])  # shape: (total_samples, n_static_features)
        total_samples = all_seq_data_np.shape[0]
        
        print(f"Sequential data shape (after padding): {all_seq_data_np.shape}")
        print(f"Static data shape: {all_static_data_np.shape}")
        print("First sequential sample:\n", all_seq_data_np[0])
        print("First static sample:\n", all_static_data_np[0])
        
        # Split indices
        np.random.seed(random_seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        num_background = int(background_pct * total_samples)
        background_idx = indices[:num_background]
        test_idx = indices[num_background:]
        
        print(f"N background samples: {num_background}")
        print(f"N test samples before limiting: {len(test_idx)}")
        
        background_seq_np = all_seq_data_np[background_idx]
        background_static_np = all_static_data_np[background_idx]
        test_seq_np = all_seq_data_np[test_idx]
        test_static_np = all_static_data_np[test_idx]
        
        if num_samples > len(test_seq_np):
            num_samples = len(test_seq_np)
        test_seq_np = test_seq_np[:num_samples]
        test_static_np = test_static_np[:num_samples]
        
        # for later visualization.
        self.test_seq_data_np = test_seq_np
        self.test_static_data_np = test_static_np
        
        print(f"Background sequential shape: {background_seq_np.shape}")
        print(f"Background static shape: {background_static_np.shape}")
        print(f"Test sequential shape: {test_seq_np.shape}")
        print(f"Test static shape: {test_static_np.shape}")

        background_seq_tensor = torch.tensor(background_seq_np).float()
        background_static_tensor = torch.tensor(background_static_np).float()
        explainer = shap.DeepExplainer(self.model, [background_seq_tensor, background_static_tensor])
        
        shap_values_batches = []
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch_seq = torch.tensor(test_seq_np[i : i + batch_size]).float()
            batch_static = torch.tensor(test_static_np[i : i + batch_size]).float()
           
            shap_values_batch = explainer.shap_values([batch_seq, batch_static])
            shap_values_batches.append(shap_values_batch)
            print(f"Processed batch {i // batch_size + 1}")
        
        if len(shap_values_batches) == 1:
            self.shap_values = shap_values_batches[0]
            np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
            print(f'SHAP VALS: {self.shap_values}')
        #     for i, shap_val in enumerate(self.shap_values):
        #         print(f"Input {i}: {shap_val[0]}")
        #         shap_df = pd.DataFrame(shap_val[0][0])
        #         print(shap_df)
        #         continue
        else:
            num_inputs = len(shap_values_batches[0])
            self.shap_values = [
                np.concatenate([np.array(batch[i]) for batch in shap_values_batches], axis=0)
                for i in range(num_inputs)
            ]
        
        # for idx, sv in enumerate(self.shap_values):
        #     print(f"Aggregated SHAP values for input {idx} shape: {sv.shape}"





    def explain_with_ordinary_SHAP(self, feature_names):
        """
        Visualize the SHAP summary plot using the sequential input.
        """
        print("Visualizing ordinary SHAP summary plot...")
        num_features = self.test_seq_data_np.shape[2]
        aggregated_shap_values = self.shap_values[0].mean(axis=1)   # (num_samples, n_seq_features)
        aggregated_test_data = self.test_seq_data_np.mean(axis=1)      # (num_samples, n_seq_features)
        
        shap.summary_plot(
            aggregated_shap_values,
            aggregated_test_data,
            feature_names or [f"Feature {i}" for i in range(num_features)],
            show=True,
        )
    
    def plot_shap_heatmap_feature_rank(self, feature_names):
        """
        Plot a heatmap showing the mean feature rank over time (sequential input).
        """
        print("Plotting SHAP heatmap (mean feature rank)...")
        if self.shap_values is None:
            raise ValueError("Run extract_shap_values first.")
        shap_values = np.abs(self.shap_values[0])
        ranks = np.argsort(np.argsort(-shap_values, axis=2), axis=2) + 1
        mean_ranks = ranks.mean(axis=0)  # (time_steps, n_seq_features)
        overall_mean = mean_ranks.mean(axis=0)  # (n_seq_features,)
        sorted_indices = np.argsort(overall_mean)
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        sorted_mean_ranks = mean_ranks[:, sorted_indices].T  # (n_seq_features, time_steps)
        
        time_steps = sorted_mean_ranks.shape[1]
        df = pd.DataFrame(
            sorted_mean_ranks,
            index=sorted_feature_names,
            columns=[f"Time {i}" for i in range(time_steps)]
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="Reds_r", annot=False, fmt=".2f", cbar_kws={"label": "Mean Rank"})
        plt.title("SHAP Heatmap (Mean Feature Rank)")
        plt.xlabel("Time Steps")
        plt.ylabel("Features (Sorted by Overall Importance)")
        plt.show()
    
    def plot_shap_heatmap_mean_abs(self, feature_names):
        """
        Plot a heatmap showing the mean absolute SHAP score per feature at each timestep (sequential input).
        """
        print("Plotting SHAP heatmap (mean absolute SHAP) for sequential input...")
        if self.shap_values is None:
            raise ValueError("Run extract_shap_values first.")
        for output_idx, class_shap in enumerate(self.shap_values):
            print(f"Processing SHAP values for input {output_idx}...")
            shap_abs = np.abs(class_shap)
            mean_abs = shap_abs.mean(axis=0)  # (time_steps, n_seq_features)
            overall_mean = mean_abs.mean(axis=0)  # (n_seq_features,)
            sorted_indices = np.argsort(overall_mean)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_mean_abs = mean_abs[:, sorted_indices].T  # (n_seq_features, time_steps)
            time_steps = sorted_mean_abs.shape[1]
            df = pd.DataFrame(
                sorted_mean_abs,
                index=sorted_feature_names,
                columns=[f"Time {i}" for i in range(time_steps)]
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(df, cmap="Reds", annot=False, fmt=".2f", cbar_kws={"label": "Mean Absolute SHAP Score"})
            plt.title(f"SHAP Heatmap (Mean Absolute SHAP) – Input {output_idx}")
            plt.xlabel("Time Steps")
            plt.ylabel("Features (Sorted by Overall Importance)")
            plt.show()
    
    def plot_single_feature_time_shap(self, sample_idx, variable_name, scaler=None):
        """
        Plot sequential feature values and corresponding SHAP values over time for a single feature.
        """
        if self.shap_values is None:
            raise ValueError("Run extract_shap_values first.")
        if not self.feature_names:
            raise ValueError("Set feature_names in the explainer.")
        if variable_name not in self.feature_names:
            raise ValueError(f"Variable '{variable_name}' not found in feature names.")
        
        feature_idx = self.feature_names.index(variable_name)
        raw_values = self.test_seq_data_np[sample_idx, :, feature_idx]
        if scaler is not None:
            feature_values = raw_values * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]
        else:
            feature_values = raw_values
        time_steps = np.arange(len(feature_values))
        
        for output_idx, shap_vals_input in enumerate(self.shap_values):
            shap_vals = shap_vals_input[sample_idx, :, feature_idx]
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Feature Value", color="tab:blue")
            ax1.plot(time_steps, feature_values, color="tab:blue", label="Feature Value")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            
            ax2 = ax1.twinx()
            ax2.set_ylabel("SHAP Value", color="tab:red")
            ax2.plot(time_steps, shap_vals, color="tab:red", label="SHAP Value")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            
            plt.title(f"Feature: {variable_name} (Sample {sample_idx}, Input {output_idx})")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            plt.tight_layout()
            plt.show()
    
    def is_probability_output(self, sample_input):
        """
        Check if model output is a probability.
        sample_input should be a tuple: (seq_tensor, static_tensor)
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(*sample_input)
            print(f"Model output: {output}")
            output_np = output.cpu().numpy().flatten()
            return np.all((output_np >= 0.0) & (output_np <= 1.0))
    
    def explain(self, sequences, method, num_samples=1000, scaler=None, numerical_features=None,
                feature_to_explain=None, batch_size=10):
        """
        Main method to extract and visualize SHAP values.
        sequences: list of tuples (sequential_features, static_features, label)
        """
        sample_np_seq = np.array([sequences[0][0]])
        sample_np_static = np.array([sequences[0][1]])
        sample_torch = (torch.tensor(sample_np_seq).float(), torch.tensor(sample_np_static).float())
        
        if self.is_probability_output(sample_torch):
            print("Model outputs probabilities: wrapping to logits...")
            self.model = LogitWrapper(self.model)
        else:
            print("Model outputs logits.")
        
        self.extract_shap_values(sequences, num_samples, batch_size)
        print("Feature names:", self.feature_names)
        if method == "ordinary_SHAP":
            self.explain_with_ordinary_SHAP(self.feature_names)
        elif method == "heatmap_SHAP":
            self.plot_shap_heatmap_mean_abs(self.feature_names)
        elif method == "feature_rank_heatmap_SHAP":
            self.plot_shap_heatmap_feature_rank(self.feature_names)
        elif method == "plot_single_feature_time_shap":
            self.plot_single_feature_time_shap(sample_idx=1, variable_name=feature_to_explain, scaler=scaler)
        else:
            raise ValueError(f"Unknown method: {method}")


class LogitWrapper(torch.nn.Module):
    """
    Wraps a model that returns probabilities so that it returns logits.
    Assumes a binary classification scenario.
    """
    def __init__(self, model):
        super(LogitWrapper, self).__init__()
        self.model = model

    def forward(self, *args):
        # If a single argument is passed and it is a tuple or list, unpack it.
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        p = self.model(*args)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        logits = torch.log(p / (1 - p))
        return logits
