import shap
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.shap_values = None
        self.test_data_np = None
        self.scale_factors = None

    def extract_shap_values(self, sequences, num_samples, batch_size=10):
        """
        Extract SHAP values for the model and store them in the class instance.

        Parameters:
        - sequences: Input sequences for SHAP computation.
        - num_samples: Number of samples to explain.
        - batch_size: Batch size for SHAP computation.
        """
        print("Extracting SHAP values...")

        self.test_data_np = np.array([seq[0] for seq in sequences[:num_samples]])
        print(f"Test data shape: {self.test_data_np.shape}")

        background_data = torch.tensor(
            [seq[0] for seq in sequences[:batch_size]]
        ).float()
        print(f"Background data shape: {background_data.shape}")

        explainer = shap.DeepExplainer(self.model, background_data)

        test_data_tensor = torch.tensor(
            [seq[0] for seq in sequences[:num_samples]]
        ).float()
        print(f"Test data tensor shape: {test_data_tensor.shape}")

        shap_values_batches = []

        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch = test_data_tensor[i : i + batch_size]
            shap_values_batch = explainer.shap_values(batch)
            shap_values_batches.append(shap_values_batch)
            print(f"Processed batch {i//batch_size + 1}")

        # concat SHAP values across batches
        self.shap_values = [
            np.concatenate([batch[i] for batch in shap_values_batches], axis=0)
            for i in range(len(shap_values_batches[0]))
        ]
        print(f"SHAP values shape: {[sv.shape for sv in self.shap_values]}")

    def explain_with_ordinary_SHAP(self, feature_names):
        """
        Visualize SHAP values using the ordinary SHAP summary plot.
        """
        print("Visualizing SHAP summary plot...")

        num_features = self.test_data_np.shape[2]

        aggregated_shap_values = self.shap_values[0].mean(
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

    def plot_shap_heatmap(self, feature_names):
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
        shap_values = np.abs(self.shap_values[0])  # absolute SHAP values

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
            cmap="RdBu",
            annot=False,
            fmt=".2f",
            cbar_kws={"label": "Mean Rank"},
        )
        plt.title("SHAP Heatmap (Mean Feature Rank in Importance)")
        plt.xlabel("Time Steps")
        plt.ylabel("Features (Sorted by Overall Importance)")
        plt.show()

    def explain(
        self,
        sequences,
        feature_names,
        method,
        num_samples=1000,
        scaler=None,
        numerical_features=None,
        batch_size=10,
        feature_idx=None,
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

        self.extract_shap_values(sequences, num_samples, batch_size)

        if (
            method == "scatter_SHAP"
            and scaler is not None
            and numerical_features is not None
        ):
            numerical_indices = [feature_names.index(f) for f in numerical_features]
            print(f"Numerical feature indices for SHAP adjustment: {numerical_indices}")
            print(f"Scale factors for SHAP adjustment: {self.scale_factors}")
            # Correctly adjust SHAP values by dividing by scale_factors
            self.shap_values[0][:, :, numerical_indices] /= self.scale_factors
            print("Adjusted SHAP values by dividing with scale factors.")

        if method == "ordinary_SHAP":
            self.explain_with_ordinary_SHAP(feature_names)
        elif method == "heatmap_SHAP":
            self.plot_shap_heatmap(feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")