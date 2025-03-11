import shap
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# idx 1 is non-survival, idx 0 is survival
class SHAPExplainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.shap_values = None
        self.test_data_np = None
        self.scale_factors = None
        self.feature_names = feature_names

    def extract_shap_values(
        self, sequences, num_samples, batch_size=10, background_pct=0.1, random_seed=42
    ):
        """
        Extract SHAP values for the model and store them in the class instance.

        Parameters:
        - sequences: Input sequences for SHAP computation.
        - num_samples: Number of samples to explain.
        - batch_size: Batch size for SHAP computation.
        """
        print("Extracting SHAP values...")

        all_data_np = np.array([seq[0] for seq in sequences])
        total_samples, time_steps, num_features = all_data_np.shape

        print(f"Dataset Shape: {all_data_np.shape}")
        print(f"First 10 rows of dataset:\n{all_data_np[:10]}")

        # decide on backgound data
        np.random.seed(random_seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        num_background = int(background_pct * total_samples)
        background_idx = indices[:num_background]
        test_idx = indices[num_background:]

        print(f"N background samples: {num_background}")
        print(f"N remaining samples: {len(test_idx)}")

        background_data_np = all_data_np[background_idx]
        # torch.Tensor for shap.DeepExplainer
        background_data = torch.tensor(background_data_np).float()

        print(f"Background data shape: {background_data.shape}")
        print("background data row:\n", background_data_np[0])

        # exclude backgrond from test
        test_data_np = all_data_np[test_idx]
        if num_samples > len(test_data_np):
            num_samples = len(test_data_np)

        # remaining samples in background? nicht rausnehmen

        # take the samples from test data
        test_data_np = test_data_np[:num_samples]
        self.test_data_np = test_data_np

        print(f"Background data shape: {background_data.shape}")
        print("background data row:\n", background_data_np[0])

        print(f"test data shape: {test_data_np.shape}")
        print("test data row:\n", test_data_np[0])

        explainer = shap.DeepExplainer(self.model, background_data)

        # test data to tensor
        test_data_tensor = torch.tensor(test_data_np).float()

        # run shap batches
        shap_values_batches = []
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch = test_data_tensor[i : i + batch_size]
            shap_values_batch = explainer.shap_values(batch)
            print(shap_values_batch)
            shap_values_batches.append(shap_values_batch)
            print(f"Processed batch {i // batch_size + 1}")

        # concat SHAP values across batches
        self.shap_values = [
            np.concatenate([batch[i] for batch in shap_values_batches], axis=0)
            for i in range(len(shap_values_batches[0]))
        ]
        print(f"SHAP values shape: {self.shap_values[0][0].shape}")
        for i in range(len(self.shap_values[0])):
            vals = self.shap_values[0][i]
            sum1 = np.sum(vals, axis=1)
            sum11 = np.sum(sum1)
            print(f"sum axis 1 {sum11}")

        # inspect
        for output_index, sv in enumerate(self.shap_values):
            print(f"SHAP values output {output_index} shape: {sv.shape}")
            print(f"Example rows of SHAP values output {output_index}:\n{sv[:10]}")

    def explain_with_ordinary_SHAP(self, feature_names):
        """
        Visualize SHAP values using the ordinary SHAP summary plot.
        """
        print("Visualizing SHAP summary plot...")

        num_features = self.test_data_np.shape[2]

        aggregated_shap_values = self.shap_values[1].mean(
            axis=1
        )  # Shape: (num_samples, num_features)
        aggregated_test_data = self.test_data_np.mean(
            axis=1
        )  # Shape: (num_samples, num_features)

        print(f"Aggregated SHAP values shape: {aggregated_shap_values.shape}")
        print(f"Aggregated test data shape: {aggregated_test_data.shape}")

        shap.summary_plot(
            aggregated_shap_values,
            aggregated_test_data,
            feature_names or [f"Feature {i}" for i in range(num_features)],
            show=True,
        )

    def plot_shap_heatmap_feature_rank(self, feature_names):
        """
        Plot a heatmap showing the mean feature rank in importance for each timestep,
        with features sorted by overall importance (first rank across all timesteps).
        """
        print("Plotting SHAP heatmap (mean feature rank)...")

        if self.shap_values is None:
            raise ValueError(
                "SHAP values have not been extracted. Run extract_shap_values first."
            )

        # shape: (num_samples, time_steps, num_features)
        shap_values = np.abs(self.shap_values[1])  # absolute SHAP values

        #  ranks across features for each sample and timestep
        ranks = (
            np.argsort(np.argsort(-shap_values, axis=2), axis=2) + 1
        )  # Rank starts from 1
        mean_ranks = ranks.mean(
            axis=0
        )  # mean rank across samples, shape: (time_steps, num_features)

        #  overall mean rank across all timesteps
        overall_mean_ranks = mean_ranks.mean(axis=0)  # Shape: (num_features,)

        sorted_indices = np.argsort(overall_mean_ranks)
        sorted_feature_names = [feature_names[i] for i in sorted_indices]
        sorted_mean_ranks = mean_ranks[
            :, sorted_indices
        ].T  # Transpose for heatmap (features x time_steps)

        time_steps = sorted_mean_ranks.shape[1]
        df = pd.DataFrame(
            sorted_mean_ranks,
            index=sorted_feature_names,
            columns=[f"Time {i}" for i in range(time_steps)],
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df,
            cmap="Reds_r",
            annot=False,
            fmt=".2f",
            cbar_kws={"label": "Mean Rank"},
        )
        plt.title("SHAP Heatmap (Mean Feature Rank in Importance)")
        plt.xlabel("Time Steps")
        plt.ylabel("Features (Sorted by Overall Importance)")
        plt.show()

    def plot_shap_heatmap_mean_abs(self, feature_names):
        """
        Plot a heatmap showing the mean absolute SHAP score for each feature at each timestep,
        with features sorted by overall mean absolute SHAP importance, **for all classes**.
        """
        print("Plotting SHAP heatmap (mean absolute SHAP) for ALL classes...")

        if self.shap_values is None:
            raise ValueError(
                "SHAP values have not been extracted. Run extract_shap_values first."
            )

        # Loop over each class (or each output index)
        for output_idx, class_shap_values in enumerate(self.shap_values):
            print(f"\n--- Class (output_idx) = {output_idx} ---")

            # shape of class_shap_values: (num_samples, time_steps, num_features)
            # Take absolute value
            shap_values_abs = np.abs(class_shap_values)

            # Compute mean absolute SHAP scores per time step (average over samples)
            mean_abs_shap = shap_values_abs.mean(axis=0)  # shape: (time_steps, num_features)

            # Compute overall mean absolute SHAP per feature (average over time)
            overall_mean_abs = mean_abs_shap.mean(axis=0)  # shape: (num_features,)

            # Sort features by overall mean absolute SHAP (highest importance first)
            sorted_indices = np.argsort(overall_mean_abs)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]

            # Rearrange the mean absolute SHAP matrix accordingly and transpose it
            # so that rows represent features and columns represent time steps
            sorted_mean_abs_shap = mean_abs_shap[:, sorted_indices].T
            time_steps = sorted_mean_abs_shap.shape[1]

            df = pd.DataFrame(
                sorted_mean_abs_shap,
                index=sorted_feature_names,
                columns=[f"Time {i}" for i in range(time_steps)],
            )

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df,
                cmap="Reds",
                annot=False,
                fmt=".2f",
                cbar_kws={"label": "Mean Absolute SHAP Score"},
            )
            plt.title(f"SHAP Heatmap (Mean Absolute SHAP Score) – Class {output_idx}")
            plt.xlabel("Time Steps")
            plt.ylabel("Features (Sorted by Overall Mean Absolute SHAP)")
            plt.show()

    def plot_single_feature_time_shap(self, sample_idx, variable_name, scaler=None):
        """
        Plot the feature values and corresponding SHAP values over time for a single feature.
        Plot *for every class* in the model's output.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values have not been extracted. Call extract_shap_values first.")
        
        if not hasattr(self, "feature_names") or self.feature_names is None:
            raise ValueError("Feature names are not set. Please pass feature_names to the explain method.")

        if variable_name not in self.feature_names:
            raise ValueError(f"Variable name '{variable_name}' not found in feature names: {self.feature_names}")
        
        # get the feature index
        feature_idx = self.feature_names.index(variable_name)

        # get the raw (optionally unscaled) feature values for the chosen sample
        raw_values = self.test_data_np[sample_idx, :, feature_idx]

        if scaler is not None:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        else:
            feature_values = raw_values

        time_steps = np.arange(len(feature_values))

        # Now loop over each class's SHAP values
        for output_idx, shap_vals_for_class in enumerate(self.shap_values):
            shap_vals = shap_vals_for_class[sample_idx, :, feature_idx]

            fig, ax1 = plt.subplots(figsize=(10, 5))

            color_feature = "tab:blue"
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Feature Value", color=color_feature)
            ax1.plot(time_steps, feature_values, color=color_feature, label="Feature Value")
            ax1.tick_params(axis="y", labelcolor=color_feature)

            ax2 = ax1.twinx()
            color_shap = "tab:red"
            ax2.set_ylabel("SHAP Value", color=color_shap)
            ax2.plot(time_steps, shap_vals, color=color_shap, label="SHAP Value")
            ax2.tick_params(axis="y", labelcolor=color_shap)

            plt.title(f"Feature: {variable_name} (Sample {sample_idx}, Class={output_idx})")

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            plt.tight_layout()
            plt.show()

    def is_probability_output(self, sample_input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
            print(f"output: {output}")
            # Assuming binary classification with output shape (batch_size, 1) or (batch_size,)
            output_np = output.numpy().flatten()
            return np.all((output_np >= 0.0) & (output_np <= 1.0))

    def explain(
        self,
        sequences,
        method,
        num_samples=1000,
        scaler=None,
        numerical_features=None,
        feature_to_explain=None,
        batch_size=10,
        #feature_idx=None,
        
    ):
        """
        Main method to extract SHAP values and visualize based on the chosen method.

        Parameters:
        - sequences: Input sequences for SHAP computation.
        - feature_names: List of feature names.
        - method: Visualization method ("ordinary_SHAP" or "scatter_SHAP").
        - num_samples: Number of samples to explain.
        - batch_size: Batch size for SHAP computation.
        """
        sample_np = np.array([sequences[0][0]])  # shape (1, time_steps, features)
        sample_torch = torch.tensor(sample_np).float()

        if self.is_probability_output(sample_torch):
            print("model output probabilities: wrapping to return logits..")
            self.model = LogitWrapper(self.model)
        else:
            print("Model outputslogits")

        self.extract_shap_values(sequences, num_samples, batch_size)
        print(self.feature_names)
        if (
            method == "scatter_SHAP"
            and scaler is not None
            and numerical_features is not None
        ):
            numerical_indices = [self.feature_names.index(f) for f in numerical_features]
            print(f"Numerical feature indices for SHAP adjustment: {numerical_indices}")
            print(f"Scale factors for SHAP adjustment: {self.scale_factors}")
            # Correctly adjust SHAP values by dividing by scale_factors
            self.shap_values[1][:, :, numerical_indices] /= self.scale_factors
            print("Adjusted SHAP values by dividing with scale factors.")

        if method == "ordinary_SHAP":
            self.explain_with_ordinary_SHAP(self.feature_names)
        elif method == "heatmap_SHAP":
            self.plot_shap_heatmap_mean_abs(self.feature_names)
        elif method == "feature_rank_heatmap_SHAP":
            self.plot_shap_heatmap_feature_rank(self.feature_names)
        elif method == "plot_single_feature_time_shap":
            self.plot_single_feature_time_shap(
                sample_idx=1, variable_name=feature_to_explain, scaler=scaler
            )

        else:
            raise ValueError(f"Unknown method: {method}")
        # ['mbp_value', 'gcs_total_value', 'glc_value', 'creatinine_value', 'potassium_value', 'hr_value', 'wbc_value', 'platelets_value', 'inr_value', 'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value', 'age_value', 'gender_value']


class LogitWrapper(torch.nn.Module):
    """
    Wraps a model that returns probabilities so that it instead returns logits.
    Assumes a *binary* classification scenario with a single probability output (p).
    """

    def __init__(self, model):
        super(LogitWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        p = self.model(x)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        logits = torch.log(p / (1 - p))
        return logits
