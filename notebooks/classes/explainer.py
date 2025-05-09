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
import matplotlib.gridspec as gridspec

###############################################################################
# Base class with common functionality
###############################################################################
class SHAPExplainerBase:
    def __init__(self, model, feature_names=None):
        self.orig_model = model       # original (trained) model
        self.model = model            # may be wrapped if needed
        self.feature_names = feature_names

        # Variables that will be set after extraction or loading:
        self.shap_values = None       # structure: [[seq_class0, static_class0], [seq_class1, static_cl ass1]]
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
        if self.shap_values is None:
            raise ValueError("SHAP values not set.")
        # choose 0 for survival and 1 for non survival
        label_expl = 1
        print(f'shap set to explainatino for label {label_expl}')
        shap_seq = self.shap_values[label_expl][0]    # (batch, time, #dynamic)
        shap_static = self.shap_values[1][1] # (batch, #static)
        time_steps = self.test_seq_data_np.shape[1]

        static_feat_names = ["age_value", "gender_value"]  # or as appropriate
        n_dynamic = shap_seq.shape[2]
        n_static  = shap_static.shape[1]

        mean_abs_seq    = np.abs(shap_seq).mean(axis=0)  # (time, n_dynamic)
        mean_abs_static = np.abs(shap_static).mean(axis=0)  # (n_static,)

        mean_abs_static_corrected = mean_abs_static / time_steps

        dyn_importance = mean_abs_seq.mean(axis=0)
        stat_importance = mean_abs_static_corrected

        all_feat_names = self.feature_names
        overall_importances = np.concatenate([dyn_importance, stat_importance])
        sorted_indices = np.argsort(overall_importances)[::-1]

        dyn_sorted = [i for i in sorted_indices if i < n_dynamic]
        stat_sorted = [i for i in sorted_indices if i >= n_dynamic]

        mean_abs_seq_sorted = mean_abs_seq[:, dyn_sorted].T
        dyn_feat_names_sorted = [all_feat_names[i] for i in dyn_sorted]
        dyn_df = pd.DataFrame(
            mean_abs_seq_sorted,
            index=dyn_feat_names_sorted,
            columns=[f"Time {t}" for t in range(time_steps)]
        ).clip(lower=1e-6)

        repeated_static = []
        stat_feat_names_sorted = [all_feat_names[i] for i in stat_sorted]
        for idx in stat_sorted:
            corrected_val = mean_abs_static_corrected[idx - n_dynamic]
            repeated_static.append(np.repeat(corrected_val, time_steps))
        stat_arr_sorted = np.vstack(repeated_static)
        stat_df = pd.DataFrame(
            stat_arr_sorted,
            index=stat_feat_names_sorted,
            columns=[f"Time {t}" for t in range(time_steps)]
        ).clip(lower=1e-6)


        nrows_dyn = dyn_df.shape[0]
        nrows_stat = stat_df.shape[0]

        fig = plt.figure(figsize=(10, 0.7*(nrows_dyn + nrows_stat)))
        gs = gridspec.GridSpec(
            nrows=2, ncols=2,
            width_ratios=[30, 1],  # left column  heatmaps, right column  colorbar
            height_ratios=[nrows_dyn * 1.5, nrows_stat],  
            hspace=0.4, wspace=0.05
        )

        ax_top    = fig.add_subplot(gs[0, 0])
        ax_bottom = fig.add_subplot(gs[1, 0])
        ax_cbar   = fig.add_subplot(gs[:, 1])  # colorbar spans both rows

        vmin = min(dyn_df.values.min(), stat_df.values.min())
        vmax = max(dyn_df.values.max(), stat_df.values.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)

        sns.heatmap(dyn_df, ax=ax_top,
                    cmap="Reds", norm=norm,
                    cbar=False,
                    xticklabels=True, yticklabels=True)

        ax_top.set_title("Mean Absolute SHAP Score")
        ax_top.set_xlabel("Time Steps")
        ax_top.set_ylabel("Sequential", labelpad=10)

        sns.heatmap(stat_df, ax=ax_bottom,
                    cmap="Reds", norm=norm,
                    cbar=False,
                    xticklabels=False, yticklabels=True)

        ax_bottom.set_title("")
        ax_bottom.set_xlabel("")
        ax_bottom.set_ylabel("Static", labelpad=30)


        sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
        cbar.set_label("Mean Absolute SHAP (Log Scale)")

        plt.show()
        
    def plot_single_feature_time_shap_three_train_datasets(self, file_path1, file_path2, file_path3, sample_idx, feature_to_explain, model1_name, model2_name, model3_name, input_type='sequential', feature_idx=None):
        """
        Plot a single feature's values over time together with the corresponding SHAP values 
        for non-survival (label 1) from three different models. These models are of the same type,
        but trained on different training datasets. The test data is assumed to be the same across files.
        
        Parameters:
            file_path1 (str): JSON file path for the first training dataset.
            file_path2 (str): JSON file path for the second training dataset.
            file_path3 (str): JSON file path for the third training dataset.
            sample_idx (int): The sample index to plot.
            feature_to_explain (str): The feature name to annotate.
            model1_name (str): Label/name for model 1.
            model2_name (str): Label/name for model 2.
            model3_name (str): Label/name for model 3.
            input_type (str): Either 'sequential' or 'static'.
            feature_idx (int): The index of the feature to plot.
        """
     
        # Load JSON files.
        with open(file_path1, "r") as f:
            data1 = json.load(f)
        with open(file_path2, "r") as f:
            data2 = json.load(f)
        with open(file_path3, "r") as f:
            data3 = json.load(f)
            
        # Use test data from the first file (assuming it's identical across all files).
        test_seq_data = np.array(data1["test_data"]["seq"])
        test_static_data = np.array(data1["test_data"]["static"])
        test_labels = np.array(data1["test_data"]["labels"])
        
        # Get the scaler from file1 metadata.
        if "metadata" in data1 and "scaler" in data1["metadata"]:
            scaler_info = data1["metadata"]["scaler"]
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            scaler = dummy
        else:
            raise ValueError("No scaler found in the JSON metadata of the first file.")
        
        # Extract SHAP values for non-survival (label 1) from each JSON.
        label_selection  = "label_1"
        shap_seq1 = np.array(data1["shap_values"][label_selection]["sequential"])
        shap_static1 = np.array(data1["shap_values"][label_selection]["static"])
        shap_seq2 = np.array(data2["shap_values"][label_selection]["sequential"])
        shap_static2 = np.array(data2["shap_values"][label_selection]["static"])
        shap_seq3 = np.array(data3["shap_values"][label_selection]["sequential"])
        shap_static3 = np.array(data3["shap_values"][label_selection]["static"])
        
        # Depending on the input type, extract raw feature values and corresponding SHAP values.
        if input_type.lower() == 'sequential':
            ts_length = test_seq_data.shape[1]
            raw_values = test_seq_data[sample_idx, :, feature_idx]
            shap_vals_model1 = shap_seq1[sample_idx, :, feature_idx]
            shap_vals_model2 = shap_seq2[sample_idx, :, feature_idx]
            shap_vals_model3 = shap_seq3[sample_idx, :, feature_idx]
        elif input_type.lower() == 'static':
            ts_length = test_seq_data.shape[1]  # use the time steps from the sequential branch
            raw_value = test_static_data[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
            shap_val1 = shap_static1[sample_idx, feature_idx]
            shap_val2 = shap_static2[sample_idx, feature_idx]
            shap_val3 = shap_static3[sample_idx, feature_idx]
            shap_vals_model1 = np.repeat(shap_val1, ts_length)
            shap_vals_model2 = np.repeat(shap_val2, ts_length)
            shap_vals_model3 = np.repeat(shap_val3, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")
        
        # Descale the raw feature values.
        try:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        except Exception as e:
            print("Error in descaling:", e)
            feature_values = raw_values
        
        time_steps = np.arange(ts_length)
        
        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y")
        
        # Plot SHAP values from each model.
        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non-Survival]")
        # Assign distinct colors for clarity.
        ax2.plot(time_steps, shap_vals_model1, color='blue', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model1_name}")
        ax2.plot(time_steps, shap_vals_model2, color='red', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model2_name}")
        ax2.plot(time_steps, shap_vals_model3, color='green', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model3_name}")
        
        # Adjust y-axis limits based on combined SHAP values.
        combined_shap = np.concatenate([shap_vals_model1, shap_vals_model2, shap_vals_model3])
        min_val = np.min(combined_shap)
        max_val = np.max(combined_shap)
        margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        ax2.set_ylim(min_val - margin, max_val + margin)
        
        ax1.set_title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        
        # Combine legends.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # Build annotation text for predicted and actual classes.
        pred_text = ""
        actual_label_val = test_labels[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        
        # Model 1 predictions.
        if "predictions" in data1:
            pred1 = data1["predictions"][sample_idx]
            p_surv1 = pred1.get("p_surv", None)
            p_non_surv1 = pred1.get("p_non_surv", None)
            if p_surv1 is not None and p_non_surv1 is not None:
                if p_surv1 >= p_non_surv1:
                    predicted_class_str1 = "Survival"
                    predicted_prob1 = p_surv1
                else:
                    predicted_class_str1 = "Non Survival"
                    predicted_prob1 = p_non_surv1
                pred_text += f"{model1_name} Prediction: {predicted_class_str1} (P={predicted_prob1:.2f})\n"
        
        # Model 2 predictions.
        if "predictions" in data2:
            pred2 = data2["predictions"][sample_idx]
            p_surv2 = pred2.get("p_surv", None)
            p_non_surv2 = pred2.get("p_non_surv", None)
            if p_surv2 is not None and p_non_surv2 is not None:
                if p_surv2 >= p_non_surv2:
                    predicted_class_str2 = "Survival"
                    predicted_prob2 = p_surv2
                else:
                    predicted_class_str2 = "Non Survival"
                    predicted_prob2 = p_non_surv2
                pred_text += f"{model2_name} Prediction: {predicted_class_str2} (P={predicted_prob2:.2f})\n"
        
        # Model 3 predictions.
        if "predictions" in data3:
            pred3 = data3["predictions"][sample_idx]
            p_surv3 = pred3.get("p_surv", None)
            p_non_surv3 = pred3.get("p_non_surv", None)
            if p_surv3 is not None and p_non_surv3 is not None:
                if p_surv3 >= p_non_surv3:
                    predicted_class_str3 = "Survival"
                    predicted_prob3 = p_surv3
                else:
                    predicted_class_str3 = "Non Survival"
                    predicted_prob3 = p_non_surv3
                pred_text += f"{model3_name} Prediction: {predicted_class_str3} (P={predicted_prob3:.2f})\n"
        
        pred_text += f"Actual: {actual_label_str}"
        
        # Add annotation box.
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])
        ax_annot.axis("off")
        ax_annot.text(0.5, 0.2, pred_text,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.25)
        plt.show()
        
    def plot_single_feature_time_shap_three_train_datasets(self, file_path1, file_path2, file_path3, sample_idx, feature_to_explain, model1_name, model2_name, model3_name, input_type='sequential', feature_idx=None):
        """
        Plot a single feature's values over time together with the corresponding SHAP values 
        for non-survival (label 1) from three different models. These models are of the same type,
        but trained on different training datasets. The test data is assumed to be the same across files.
        
        Parameters:
            file_path1 (str): JSON file path for the first training dataset.
            file_path2 (str): JSON file path for the second training dataset.
            file_path3 (str): JSON file path for the third training dataset.
            sample_idx (int): The sample index to plot.
            feature_to_explain (str): The feature name to annotate.
            model1_name (str): Label/name for model 1.
            model2_name (str): Label/name for model 2.
            model3_name (str): Label/name for model 3.
            input_type (str): Either 'sequential' or 'static'.
            feature_idx (int): The index of the feature to plot.
        """
     
        # Load JSON files.
        with open(file_path1, "r") as f:
            data1 = json.load(f)
        with open(file_path2, "r") as f:
            data2 = json.load(f)
        with open(file_path3, "r") as f:
            data3 = json.load(f)
            
        # Use test data from the first file (assuming it's identical across all files).
        test_seq_data = np.array(data1["test_data"]["seq"])
        test_static_data = np.array(data1["test_data"]["static"])
        test_labels = np.array(data1["test_data"]["labels"])
        
        # Get the scaler from file1 metadata.
        if "metadata" in data1 and "scaler" in data1["metadata"]:
            scaler_info = data1["metadata"]["scaler"]
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            scaler = dummy
        else:
            raise ValueError("No scaler found in the JSON metadata of the first file.")
        
        # Extract SHAP values for non-survival (label 1) from each JSON.
        label_selection  = "label_1"
        shap_seq1 = np.array(data1["shap_values"][label_selection]["sequential"])
        shap_static1 = np.array(data1["shap_values"][label_selection]["static"])
        shap_seq2 = np.array(data2["shap_values"][label_selection]["sequential"])
        shap_static2 = np.array(data2["shap_values"][label_selection]["static"])
        shap_seq3 = np.array(data3["shap_values"][label_selection]["sequential"])
        shap_static3 = np.array(data3["shap_values"][label_selection]["static"])
        
        # Depending on the input type, extract raw feature values and corresponding SHAP values.
        if input_type.lower() == 'sequential':
            ts_length = test_seq_data.shape[1]
            raw_values = test_seq_data[sample_idx, :, feature_idx]
            shap_vals_model1 = shap_seq1[sample_idx, :, feature_idx]
            shap_vals_model2 = shap_seq2[sample_idx, :, feature_idx]
            shap_vals_model3 = shap_seq3[sample_idx, :, feature_idx]
        elif input_type.lower() == 'static':
            ts_length = test_seq_data.shape[1]  # use the time steps from the sequential branch
            raw_value = test_static_data[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
            shap_val1 = shap_static1[sample_idx, feature_idx]
            shap_val2 = shap_static2[sample_idx, feature_idx]
            shap_val3 = shap_static3[sample_idx, feature_idx]
            shap_vals_model1 = np.repeat(shap_val1, ts_length)
            shap_vals_model2 = np.repeat(shap_val2, ts_length)
            shap_vals_model3 = np.repeat(shap_val3, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")
        
        # Descale the raw feature values.
        try:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        except Exception as e:
            print("Error in descaling:", e)
            feature_values = raw_values
        
        time_steps = np.arange(ts_length)
        
        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y")
        
        # Plot SHAP values from each model.
        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non-Survival]")
        # Assign distinct colors for clarity.
        ax2.plot(time_steps, shap_vals_model1, color='blue', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model1_name}")
        ax2.plot(time_steps, shap_vals_model2, color='red', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model2_name}")
        ax2.plot(time_steps, shap_vals_model3, color='green', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model3_name}")
        
        # Adjust y-axis limits based on combined SHAP values.
        combined_shap = np.concatenate([shap_vals_model1, shap_vals_model2, shap_vals_model3])
        min_val = np.min(combined_shap)
        max_val = np.max(combined_shap)
        margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        ax2.set_ylim(min_val - margin, max_val + margin)
        
        ax1.set_title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        
        # Combine legends.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # Build annotation text for predicted and actual classes.
        pred_text = ""
        actual_label_val = test_labels[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        
        # Model 1 predictions.
        if "predictions" in data1:
            pred1 = data1["predictions"][sample_idx]
            p_surv1 = pred1.get("p_surv", None)
            p_non_surv1 = pred1.get("p_non_surv", None)
            if p_surv1 is not None and p_non_surv1 is not None:
                if p_surv1 >= p_non_surv1:
                    predicted_class_str1 = "Survival"
                    predicted_prob1 = p_surv1
                else:
                    predicted_class_str1 = "Non Survival"
                    predicted_prob1 = p_non_surv1
                pred_text += f"{model1_name} Predicted: {predicted_class_str1} (P={predicted_prob1:.2f})\n"
        
        # Model 2 predictions.
        if "predictions" in data2:
            pred2 = data2["predictions"][sample_idx]
            p_surv2 = pred2.get("p_surv", None)
            p_non_surv2 = pred2.get("p_non_surv", None)
            if p_surv2 is not None and p_non_surv2 is not None:
                if p_surv2 >= p_non_surv2:
                    predicted_class_str2 = "Survival"
                    predicted_prob2 = p_surv2
                else:
                    predicted_class_str2 = "Non Survival"
                    predicted_prob2 = p_non_surv2
                pred_text += f"{model2_name} Predicted: {predicted_class_str2} (P={predicted_prob2:.2f})\n"
        
        # Model 3 predictions.
        if "predictions" in data3:
            pred3 = data3["predictions"][sample_idx]
            p_surv3 = pred3.get("p_surv", None)
            p_non_surv3 = pred3.get("p_non_surv", None)
            if p_surv3 is not None and p_non_surv3 is not None:
                if p_surv3 >= p_non_surv3:
                    predicted_class_str3 = "Survival"
                    predicted_prob3 = p_surv3
                else:
                    predicted_class_str3 = "Non Survival"
                    predicted_prob3 = p_non_surv3
                pred_text += f"{model3_name} Predicted: {predicted_class_str3} (P={predicted_prob3:.2f})\n"
        
        pred_text += f"Actual: {actual_label_str}"
        
        # Add annotation box.
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])
        ax_annot.axis("off")
        ax_annot.text(0.5, 0.2, pred_text,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    


    def plot_single_feature_time_shap_three_train_datasets(self, file_path1, file_path2, file_path3, sample_idx, feature_to_explain, model1_name, model2_name, model3_name, input_type='sequential', feature_idx=None):
        """
        Plot a single feature's values over time together with the corresponding SHAP values 
        for non-survival (label 1) from three different models. These models are of the same type,
        but trained on different training datasets. The test data is assumed to be the same across files.
        
        Parameters:
            file_path1 (str): JSON file path for the first training dataset.
            file_path2 (str): JSON file path for the second training dataset.
            file_path3 (str): JSON file path for the third training dataset.
            sample_idx (int): The sample index to plot.
            feature_to_explain (str): The feature name to annotate.
            model1_name (str): Label/name for model 1.
            model2_name (str): Label/name for model 2.
            model3_name (str): Label/name for model 3.
            input_type (str): Either 'sequential' or 'static'.
            feature_idx (int): The index of the feature to plot.
        """
     
        # Load JSON files.
        with open(file_path1, "r") as f:
            data1 = json.load(f)
        with open(file_path2, "r") as f:
            data2 = json.load(f)
        with open(file_path3, "r") as f:
            data3 = json.load(f)
            
        # Use test data from the first file (assuming it's identical across all files).
        test_seq_data = np.array(data1["test_data"]["seq"])
        test_static_data = np.array(data1["test_data"]["static"])
        test_labels = np.array(data1["test_data"]["labels"])
        
        # Get the scaler from file1 metadata.
        if "metadata" in data1 and "scaler" in data1["metadata"]:
            scaler_info = data1["metadata"]["scaler"]
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            scaler = dummy
        else:
            raise ValueError("No scaler found in the JSON metadata of the first file.")
        
        # Extract SHAP values for non-survival (label 1) from each JSON.
        label_selection  = "label_1"
        shap_seq1 = np.array(data1["shap_values"][label_selection]["sequential"])
        shap_static1 = np.array(data1["shap_values"][label_selection]["static"])
        shap_seq2 = np.array(data2["shap_values"][label_selection]["sequential"])
        shap_static2 = np.array(data2["shap_values"][label_selection]["static"])
        shap_seq3 = np.array(data3["shap_values"][label_selection]["sequential"])
        shap_static3 = np.array(data3["shap_values"][label_selection]["static"])
        
        # Depending on the input type, extract raw feature values and corresponding SHAP values.
        if input_type.lower() == 'sequential':
            ts_length = test_seq_data.shape[1]
            raw_values = test_seq_data[sample_idx, :, feature_idx]
            shap_vals_model1 = shap_seq1[sample_idx, :, feature_idx]
            shap_vals_model2 = shap_seq2[sample_idx, :, feature_idx]
            shap_vals_model3 = shap_seq3[sample_idx, :, feature_idx]
        elif input_type.lower() == 'static':
            ts_length = test_seq_data.shape[1]  # use the time steps from the sequential branch
            raw_value = test_static_data[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
            shap_val1 = shap_static1[sample_idx, feature_idx]
            shap_val2 = shap_static2[sample_idx, feature_idx]
            shap_val3 = shap_static3[sample_idx, feature_idx]
            shap_vals_model1 = np.repeat(shap_val1, ts_length)
            shap_vals_model2 = np.repeat(shap_val2, ts_length)
            shap_vals_model3 = np.repeat(shap_val3, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")
        
        # Descale the raw feature values.
        try:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        except Exception as e:
            print("Error in descaling:", e)
            feature_values = raw_values
        
        time_steps = np.arange(ts_length)
        
        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y")
        
        # Plot SHAP values from each model.
        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non-Survival]")
        # Assign distinct colors for clarity.
        ax2.plot(time_steps, shap_vals_model1, color='blue', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model1_name}")
        ax2.plot(time_steps, shap_vals_model2, color='red', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model2_name}")
        ax2.plot(time_steps, shap_vals_model3, color='green', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model3_name}")
        
        # Adjust y-axis limits based on combined SHAP values.
        combined_shap = np.concatenate([shap_vals_model1, shap_vals_model2, shap_vals_model3])
        min_val = np.min(combined_shap)
        max_val = np.max(combined_shap)
        margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        ax2.set_ylim(min_val - margin, max_val + margin)
        
        ax1.set_title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        
        # Combine legends.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # Build annotation text for predicted and actual classes.
        pred_text = ""
        actual_label_val = test_labels[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        
        # Model 1 predictions.
        if "predictions" in data1:
            pred1 = data1["predictions"][sample_idx]
            p_surv1 = pred1.get("p_surv", None)
            p_non_surv1 = pred1.get("p_non_surv", None)
            if p_surv1 is not None and p_non_surv1 is not None:
                if p_surv1 >= p_non_surv1:
                    predicted_class_str1 = "Survival"
                    predicted_prob1 = p_surv1
                else:
                    predicted_class_str1 = "Non Survival"
                    predicted_prob1 = p_non_surv1
                pred_text += f"{model1_name} Predicted: {predicted_class_str1} (P={predicted_prob1:.2f})\n"
        
        # Model 2 predictions.
        if "predictions" in data2:
            pred2 = data2["predictions"][sample_idx]
            p_surv2 = pred2.get("p_surv", None)
            p_non_surv2 = pred2.get("p_non_surv", None)
            if p_surv2 is not None and p_non_surv2 is not None:
                if p_surv2 >= p_non_surv2:
                    predicted_class_str2 = "Survival"
                    predicted_prob2 = p_surv2
                else:
                    predicted_class_str2 = "Non Survival"
                    predicted_prob2 = p_non_surv2
                pred_text += f"{model2_name} Predicted: {predicted_class_str2} (P={predicted_prob2:.2f})\n"
        
        # Model 3 predictions.
        if "predictions" in data3:
            pred3 = data3["predictions"][sample_idx]
            p_surv3 = pred3.get("p_surv", None)
            p_non_surv3 = pred3.get("p_non_surv", None)
            if p_surv3 is not None and p_non_surv3 is not None:
                if p_surv3 >= p_non_surv3:
                    predicted_class_str3 = "Survival"
                    predicted_prob3 = p_surv3
                else:
                    predicted_class_str3 = "Non Survival"
                    predicted_prob3 = p_non_surv3
                pred_text += f"{model3_name} Predicted: {predicted_class_str3} (P={predicted_prob3:.2f})\n"
        
        pred_text += f"Actual: {actual_label_str}"
        
        # Add annotation box.
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])
        ax_annot.axis("off")
        ax_annot.text(0.5, 0.2, pred_text,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.25)
        plt.show()


    def plot_single_feature_time_shap_two_models(self, file_path1, file_path2, sample_idx, feature_to_explain,model1_name, model2_name, input_type='sequential', feature_idx=None):
        
        with open(file_path1, "r") as f:
            data1 = json.load(f)
        with open(file_path2, "r") as f:
            data2 = json.load(f)
        
        test_seq_data = np.array(data1["test_data"]["seq"])
        test_static_data = np.array(data1["test_data"]["static"])
        test_labels = np.array(data1["test_data"]["labels"])

        scaler = None
        if "metadata" in data1 and "scaler" in data1["metadata"]:
            scaler_info = data1["metadata"]["scaler"]
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            scaler = dummy
        else:
            raise ValueError("No scaler found in the JSON metadata of the first file.")

    
        shap_seq1 = np.array(data1["shap_values"]["label_1"]["sequential"])
        shap_static1 = np.array(data1["shap_values"]["label_1"]["static"])
        shap_seq2 = np.array(data2["shap_values"]["label_1"]["sequential"])
        shap_static2 = np.array(data2["shap_values"]["label_1"]["static"])

        if input_type.lower() == 'sequential':
            ts_length = test_seq_data.shape[1]
            raw_values = test_seq_data[sample_idx, :, feature_idx]
            shap_vals_model1 = shap_seq1[sample_idx, :, feature_idx]
            shap_vals_model2 = shap_seq2[sample_idx, :, feature_idx]
        elif input_type.lower() == 'static':
            ts_length = test_seq_data.shape[1]  
            raw_value = test_static_data[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
            shap_val1 = shap_static1[sample_idx, feature_idx]
            shap_val2 = shap_static2[sample_idx, feature_idx]
            shap_vals_model1 = np.repeat(shap_val1, ts_length)
            shap_vals_model2 = np.repeat(shap_val2, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")

        try:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        except Exception as e:
            print("Error in descaling:", e)
            feature_values = raw_values

        time_steps = np.arange(ts_length)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values,color='black', linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y")
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non-Survival]")
        ax2.plot(time_steps, shap_vals_model1, color='red', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model1_name}")
        ax2.plot(time_steps, shap_vals_model2,color='blue', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model2_name}")
        
        combined_shap = np.concatenate([shap_vals_model1, shap_vals_model2])
        min_val = np.min(combined_shap)
        max_val = np.max(combined_shap)
        margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        ax2.set_ylim(min_val - margin, max_val + margin)
        
        ax1.set_title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        pred_text = ""
        actual_label_val = test_labels[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        
        if "predictions" in data1:
            pred1 = data1["predictions"][sample_idx]
            p_surv1 = pred1.get("p_surv", None)
            p_non_surv1 = pred1.get("p_non_surv", None)
            if p_surv1 is not None and p_non_surv1 is not None:
                if p_surv1 >= p_non_surv1:
                    predicted_class_str1 = "Survival"
                    predicted_prob1 = p_surv1
                else:
                    predicted_class_str1 = "Non Survival"
                    predicted_prob1 = p_non_surv1
                pred_text += f"{model1_name} Predicted: {predicted_class_str1} (P={predicted_prob1:.2f})\n"
        
        if "predictions" in data2:
            pred2 = data2["predictions"][sample_idx]
            p_surv2 = pred2.get("p_surv", None)
            p_non_surv2 = pred2.get("p_non_surv", None)
            if p_surv2 is not None and p_non_surv2 is not None:
                if p_surv2 >= p_non_surv2:
                    predicted_class_str2 = "Survival"
                    predicted_prob2 = p_surv2
                else:
                    predicted_class_str2 = "Non Survival"
                    predicted_prob2 = p_non_surv2
                pred_text += f"{model2_name} Predicted: {predicted_class_str2} (P={predicted_prob2:.2f})\n"
        
        pred_text += f"Label: {actual_label_str}"
        
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])
        ax_annot.axis("off")
        ax_annot.text(0.5, 0.5, pred_text,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    def plot_single_feature_time_shap(self, sample_idx, feature_to_explain, input_type='sequential', feature_idx=None):
        """
        Plot a single feature's values over time together with the corresponding Non-Survival SHAP values.
        Also annotate the plot with predicted and actual class.
        Uses saved predictions (if available) for the annotation.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values have not been set. Run extraction or load from file first.")
        

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

        # Apply descaling if scaler is provided.
        if self.scaler is not None:
            # Works for both a fitted scaler object or a dummy scaler loaded from JSON.
            try:
                feature_scale = self.scaler.scale_[feature_idx]
                feature_mean = self.scaler.mean_[feature_idx]
                feature_values = raw_values * feature_scale + feature_mean
            except Exception as e:
                print("Error in descaling:", e)
                feature_values = raw_values
        else:
            raise ValueError('No Scaler found.')

        time_steps = np.arange(ts_length)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y", labelcolor="black")

        # choose survival or no survival shap 
        shap_vals = self.shap_values[1][branch_idx]
        if input_type == 'sequential':
            shap_vals_non_surv = shap_vals[sample_idx, :, feature_idx]
        else:
            shap_val_single = shap_vals[sample_idx, feature_idx]
            shap_vals_non_surv = np.repeat(shap_val_single, ts_length)

        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value [Non Survival]")
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

        if hasattr(self, "predictions") and (self.predictions is not None):
            pred = self.predictions[sample_idx]
            p_surv = pred["p_surv"]
            p_non_surv = pred["p_non_surv"]
            if p_surv >= p_non_surv:
                predicted_class_str = "Survival"
                predicted_prob = p_surv
            else:
                predicted_class_str = "Non Survival"
                predicted_prob = p_non_surv
        else:
            predicted_class_str = "N/A"
            predicted_prob = 0.0

        text_str = (
            f"Predicted: {predicted_class_str} (P={predicted_prob:.2f})\n"
            f"Actual: {actual_label_str}"
        )
        ax_annot.text(0.5, 0.5, text_str,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
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

    def save_shap_values(self, model_name,dataset_type, num_samples, shap_dir="SHAP_scores"):
        """
        Save the SHAP values, test data, predictions, and metadata (feature names and scaler parameters)
        to a JSON file.
        The file name will be "{model_name}_{num_samples}.json" in folder shap_dir.
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values to save. Run extract_shap_values() first.")
        
        os.makedirs(shap_dir, exist_ok=True)
        print(f'model name: {model_name}')
        print(f'dataset type: {dataset_type}')
        print(f'num samples: {num_samples}')
        json_filename = f"{model_name}_{dataset_type}_{num_samples}.json"
        json_full_path = os.path.join(shap_dir, json_filename)
        
        predictions = []
        self.orig_model.eval()
        with torch.no_grad():
            for i in range(len(self.test_seq_data_np)):
                seq_t = torch.tensor(self.test_seq_data_np[i : i+1]).float()
                stat_t = torch.tensor(self.test_static_data_np[i : i+1]).float()
                out = self.orig_model(seq_t, stat_t).squeeze()
                if out.dim() == 0:
                    p_non_surv = float(out.item())
                    p_surv = 1.0 - p_non_surv
                elif out.shape[0] == 2:
                    probs = torch.softmax(out, dim=0)
                    p_surv = probs[0].item()
                    p_non_surv = probs[1].item()
                else:
                    raise ValueError("Unexpected output shape from the model.")
                predictions.append({"p_surv": p_surv, "p_non_surv": p_non_surv})
        
        metadata = {"feature_names": self.feature_names}
        if hasattr(self, "scaler") and self.scaler is not None:
            print('access Scaler saving')
            print(self.scaler.mean_)

            metadata["scaler"] = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist()
            }
        
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
            },
            "predictions": predictions,
            "metadata": metadata
        }
        
        with open(json_full_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        
        print(f"SHAP values, test data, predictions, and metadata saved to {json_full_path}")




    def explain(self,sequences, model_name,num_samples=1000, batch_size=10, save_shap=True, method="plot_single_feature_time_shap",scaler=None,dataset_type=None, **kwargs):
        """
        Main method: extract new SHAP values from the given sequences, save them if desired,
        and then run a plotting method.
        """
        self.scaler=scaler
        self.extract_shap_values(sequences, num_samples, batch_size)
        if save_shap:
            self.save_shap_values(model_name, dataset_type, num_samples)
        if method == "plot_shap_heatmap_mean_abs":
            self.plot_shap_heatmap_mean_abs()
        elif method == "plot_single_feature_time_shap":
            #  kwargs: feature_idx, feature_to_explain, input_type
            self.plot_single_feature_time_shap(**kwargs)
        else:
            raise ValueError(f"Unknown plotting method: {method}")


###############################################################################
# Child class that loads existing SHAP values and test data from file.
###############################################################################
class LoadSHAPExplainer(SHAPExplainerBase):
    def load_shap_values(self, model_name, num_samples, dataset_name=None, shap_dir="SHAP_scores"):
        """
        Load the SHAP values, test data, predictions, and metadata (including feature names and scaler parameters)
        from a JSON file.
        """
        import os, json
        filename = f"{model_name}_{dataset_name}_{num_samples}.json" #if dataset_name is not None else f"{model_name}_{num_samples}.json"
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
        
        # Load predictions if available.
        self.predictions = data.get("predictions", None)
        
        # Load metadata (e.g., feature names and scaler) if available.
        metadata = data.get("metadata", {})
        if "feature_names" in metadata:
            self.feature_names = metadata["feature_names"]
        if "scaler" in metadata:
            scaler_info = metadata["scaler"]
            # Create a simple dummy object to hold these attributes.
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            self.scaler = dummy
        
        print(f"Loaded SHAP values, test data, predictions, and metadata from {full_path}")


    def plot_single_feature_time_shap_three_train_datasets_standardized(self, file_path1, file_path2, file_path3, sample_idx, feature_to_explain, model1_name, model2_name, model3_name, input_type='sequential', feature_idx=None, ranges_json_path="ranges.json"):

        with open(ranges_json_path, "r", encoding="utf‑8") as f:
            ranges_cfg = json.load(f)
        feat_limits = ranges_cfg[feature_to_explain]["feature_ranges"]
        shap_limits = ranges_cfg[feature_to_explain]["shap_range"]

        # Load JSON files.
        with open(file_path1, "r") as f:
            data1 = json.load(f)
        with open(file_path2, "r") as f:
            data2 = json.load(f)
        with open(file_path3, "r") as f:
            data3 = json.load(f)
            
        # Use test data from the first file (assuming it's identical across all files).
        test_seq_data = np.array(data1["test_data"]["seq"])
        test_static_data = np.array(data1["test_data"]["static"])
        test_labels = np.array(data1["test_data"]["labels"])
        
        # Get the scaler from file1 metadata.
        if "metadata" in data1 and "scaler" in data1["metadata"]:
            scaler_info = data1["metadata"]["scaler"]
            class DummyScaler:
                pass
            dummy = DummyScaler()
            dummy.mean_ = np.array(scaler_info["mean"])
            dummy.scale_ = np.array(scaler_info["scale"])
            scaler = dummy
        else:
            raise ValueError("No scaler found in the JSON metadata of the first file.")
        
        # Extract SHAP values for non-survival (label 1) from each JSON.
        label_selection  = "label_1"
        shap_seq1 = np.array(data1["shap_values"][label_selection]["sequential"])
        shap_static1 = np.array(data1["shap_values"][label_selection]["static"])
        shap_seq2 = np.array(data2["shap_values"][label_selection]["sequential"])
        shap_static2 = np.array(data2["shap_values"][label_selection]["static"])
        shap_seq3 = np.array(data3["shap_values"][label_selection]["sequential"])
        shap_static3 = np.array(data3["shap_values"][label_selection]["static"])
        
        # Depending on the input type, extract raw feature values and corresponding SHAP values.
        if input_type.lower() == 'sequential':
            ts_length = test_seq_data.shape[1]
            raw_values = test_seq_data[sample_idx, :, feature_idx]
            shap_vals_model1 = shap_seq1[sample_idx, :, feature_idx]
            shap_vals_model2 = shap_seq2[sample_idx, :, feature_idx]
            shap_vals_model3 = shap_seq3[sample_idx, :, feature_idx]
     
        elif input_type.lower() == 'static':
            ts_length = test_seq_data.shape[1]  # use the time steps from the sequential branch
            raw_value = test_static_data[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
            shap_val1 = shap_static1[sample_idx, feature_idx]
            shap_val2 = shap_static2[sample_idx, feature_idx]
            shap_val3 = shap_static3[sample_idx, feature_idx]
            shap_vals_model1 = np.repeat(shap_val1, ts_length)
            shap_vals_model2 = np.repeat(shap_val2, ts_length)
            shap_vals_model3 = np.repeat(shap_val3, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")
        
        # Descale the raw feature values.
        try:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        except Exception as e:
            print("Error in descaling:", e)
            feature_values = raw_values
        
        time_steps = np.arange(ts_length)
        sum_shap1 = float(np.sum(shap_vals_model1))
        sum_shap2 = float(np.sum(shap_vals_model2))
        sum_shap3 = float(np.sum(shap_vals_model3))

        # Create the plot.
        fig, ax1 = plt.subplots(figsize=(10, 5))
        #ax1.set_xlabel("Time Step")
        #ax1.set_ylabel("Feature Value")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5)#, label="Feature Value")
        ax1.set_ylim(feat_limits["min"], feat_limits["max"])
       # ax1.set_yticks(np.arange(feat_limits["min"], feat_limits["max"], 1))
        
        # Plot SHAP values from each model.
        ax2 = ax1.twinx()
        #ax2.set_ylabel("SHAP Value [Non-Survival]")
        ax2.set_ylim(shap_limits["min"], shap_limits["max"])
        # Assign distinct colors for clarity.
        ax2.plot(time_steps, shap_vals_model1, color='blue', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model1_name} Σ={sum_shap1:.4f}")
        ax2.plot(time_steps, shap_vals_model2, color='red', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model2_name} Σ={sum_shap2:.4f}")
        ax2.plot(time_steps, shap_vals_model3, color='green', linestyle="--", linewidth=1.5, alpha=0.6, label=f"SHAP {model3_name} Σ={sum_shap3:.4f}")
        
        # Adjust y-axis limits based on combined SHAP values.
        # combined_shap = np.concatenate([shap_vals_model1, shap_vals_model2, shap_vals_model3])
        # min_val = np.min(combined_shap)
        # max_val = np.max(combined_shap)
        # margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
        # ax2.set_ylim(min_val - margin, max_val + margin)
        
        #Eax1.set_title(f"Feature: {feature_to_explain} (Sample {sample_idx})")
        
        # Combine legends.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # Build annotation text for predicted and actual classes.
        pred_text = ""
        actual_label_val = test_labels[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        
        # Model 1 predictions.
        if "predictions" in data1:
            pred1 = data1["predictions"][sample_idx]
            p_surv1 = pred1.get("p_surv", None)
            p_non_surv1 = pred1.get("p_non_surv", None)
            if p_surv1 is not None and p_non_surv1 is not None:
                if p_surv1 >= p_non_surv1:
                    predicted_class_str1 = "Survival"
                    predicted_prob1 = p_surv1
                else:
                    predicted_class_str1 = "Non Survival"
                    predicted_prob1 = p_non_surv1
                pred_text += f"{model1_name} Predicted: {predicted_class_str1} (P={predicted_prob1:.2f})\n"
        
        # Model 2 predictions.
        if "predictions" in data2:
            pred2 = data2["predictions"][sample_idx]
            p_surv2 = pred2.get("p_surv", None)
            p_non_surv2 = pred2.get("p_non_surv", None)
            if p_surv2 is not None and p_non_surv2 is not None:
                if p_surv2 >= p_non_surv2:
                    predicted_class_str2 = "Survival"
                    predicted_prob2 = p_surv2
                else:
                    predicted_class_str2 = "Non Survival"
                    predicted_prob2 = p_non_surv2
                pred_text += f"{model2_name} Predicted: {predicted_class_str2} (P={predicted_prob2:.2f})\n"
        
        # Model 3 predictions.
        if "predictions" in data3:
            pred3 = data3["predictions"][sample_idx]
            p_surv3 = pred3.get("p_surv", None)
            p_non_surv3 = pred3.get("p_non_surv", None)
            if p_surv3 is not None and p_non_surv3 is not None:
                if p_surv3 >= p_non_surv3:
                    predicted_class_str3 = "Survival"
                    predicted_prob3 = p_surv3
                else:
                    predicted_class_str3 = "Non Survival"
                    predicted_prob3 = p_non_surv3
                pred_text += f"{model3_name} Predicted: {predicted_class_str3} (P={predicted_prob3:.2f})\n"
        
        pred_text += f"True: {actual_label_str}"
        
        # Add annotation box.
        ax_annot = fig.add_axes([0.1, 0.05, 0.8, 0.1])
        ax_annot.axis("off")
        # ax_annot.text(0.5, 0.2, pred_text,
        #             ha='center', va='center', fontsize=10,
        #             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        # plt.subplots_adjust(bottom=0.25)
        plt.show()


        sum_shap1 = float(np.sum(shap_vals_model1))
        sum_shap2 = float(np.sum(shap_vals_model2))
        sum_shap3 = float(np.sum(shap_vals_model3))

        print(
        f"\nSummed SHAP contribution for sample {sample_idx} and feature '{feature_to_explain}':\n"
        f"  {model1_name}: {sum_shap1:.4f}\n"
        f"  {model2_name}: {sum_shap2:.4f}\n"
        f"  {model3_name}: {sum_shap3:.4f}\n"
        )
        
        mean_sums = {
            "mean_sum1": np.mean(np.sum(shap_seq1[:, :, feature_idx], axis=1)),
            "mean_sum2": np.mean(np.sum(shap_seq2[:, :, feature_idx], axis=1)),
            "mean_sum3": np.mean(np.sum(shap_seq3[:, :, feature_idx], axis=1)),
        }
        print(mean_sums)