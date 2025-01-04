import shap
import numpy as np
import torch
from tqdm import tqdm


class SHAPExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, sequences, feature_names, num_samples=100, batch_size=10):
        print("Explaining model with SHAP")
        test_data_np = np.array([seq[0] for seq in sequences[:num_samples]])
        num_features = test_data_np.shape[2]

        background_data = torch.tensor(
            [seq[0] for seq in sequences[:batch_size]]
        ).float()
        explainer = shap.DeepExplainer(self.model, background_data)

        print("Generating SHAP values ...")
        test_data_tensor = torch.tensor(
            [seq[0] for seq in sequences[:num_samples]]
        ).float()
        shap_values_batches = []

        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch = test_data_tensor[i : i + batch_size]
            shap_values_batch = explainer.shap_values(batch)
            shap_values_batches.append(shap_values_batch)

        shap_values = [
            np.concatenate([batch[i] for batch in shap_values_batches], axis=0)
            for i in range(len(shap_values_batches[0]))
        ]

        aggregated_shap_values = shap_values[0].mean(
            axis=1
        )  # Shape: (num_samples, num_features)
        aggregated_test_data = test_data_np.mean(
            axis=1
        )  # Shape: (num_samples, num_features)

        shap.summary_plot(
            aggregated_shap_values,
            aggregated_test_data,
            feature_names or [f"Feature {i}" for i in range(num_features)],
            show=True,  # Ensure the plot is displayed
        )
