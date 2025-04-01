#%% Common plotting script for multiple samples on one plot

import torch
import numpy as np
import matplotlib.pyplot as plt
from classes.explainer import LoadSHAPExplainer

# --- Load saved SHAP values and test data ---
model_name = "lstm_static"  
num_samples = 1000         
explainer = LoadSHAPExplainer(model=None, feature_names=None)
explainer.load_shap_values(model_name, num_samples)

# --- Define thresholds ---
high_threshold = 0.8
low_threshold_lower = 0.4
low_threshold_upper = 0.7

# Create lists for each group:
pred_surv_actual_surv = []      
pred_non_surv_actual_surv = []  
uncertain_actual_surv = []      

# For actual non survival (label == 1)
pred_surv_actual_non_surv = []     
pred_non_surv_actual_non_surv = [] 
uncertain_actual_non_surv = []     

for idx in range(len(explainer.test_seq_data_np)):
    actual = explainer.test_labels_np[idx]
    pred = explainer.predictions[idx]
    p_surv = pred["p_surv"]
    p_non_surv = pred["p_non_surv"]
    
    is_uncertain = (low_threshold_lower <= p_surv <= low_threshold_upper) or (low_threshold_lower <= p_non_surv <= low_threshold_upper)
    
    if actual == 0:  # actual survival
        if p_surv >= high_threshold:
            pred_surv_actual_surv.append(idx)
        elif p_non_surv >= high_threshold:
            pred_non_surv_actual_surv.append(idx)
        elif is_uncertain:
            uncertain_actual_surv.append(idx)
    elif actual == 1:  # actual non survival
        if p_surv >= high_threshold:
            pred_surv_actual_non_surv.append(idx)
        elif p_non_surv >= high_threshold:
            pred_non_surv_actual_non_surv.append(idx)
        elif is_uncertain:
            uncertain_actual_non_surv.append(idx)

# --- Select 5 samples from each group (if available) ---
selected_pred_surv_actual_surv = pred_surv_actual_surv[:5]
selected_pred_non_surv_actual_surv = pred_non_surv_actual_surv[:5]
selected_uncertain_actual_surv = uncertain_actual_surv[:5]

selected_pred_surv_actual_non_surv = pred_surv_actual_non_surv[:5]
selected_pred_non_surv_actual_non_surv = pred_non_surv_actual_non_surv[:5]
selected_uncertain_actual_non_surv = uncertain_actual_non_surv[:5]

print("Selected sample indices for actual survival (label=0):")
print("Predicted Survival:", selected_pred_surv_actual_surv)
print("Predicted Non Survival:", selected_pred_non_surv_actual_surv)
print("Uncertain:", selected_uncertain_actual_surv)

print("\nSelected sample indices for actual non survival (label=1):")
print("Predicted Survival:", selected_pred_surv_actual_non_surv)
print("Predicted Non Survival:", selected_pred_non_surv_actual_non_surv)
print("Uncertain:", selected_uncertain_actual_non_surv)

def common_plot_for_indices(indices, explainer, feature_idx, feature_to_explain, input_type='sequential'):
    """
    Create a common plot for a list of sample indices.
    For the given feature, global y-axis limits (2.5th to 97.5th percentiles)
    are computed using the entire test set so that all common plots for the same
    feature share identical limits. A margin of 10% of the computed range is added
    to ensure that all lines are fully visible.
    Then, each sample's descaled feature values and corresponding SHAP values 
    (from the non-survival branch) are plotted. An annotation box is added with 
    the predicted and actual labels for each sample.
    """
    if input_type == 'sequential':
        branch_idx = 0
        ts_length = explainer.test_seq_data_np.shape[1]
        if explainer.scaler is not None:
            global_feature_values = explainer.test_seq_data_np[:, :, feature_idx] * \
                                    explainer.scaler.scale_[feature_idx] + \
                                    explainer.scaler.mean_[feature_idx]
        else:
            global_feature_values = explainer.test_seq_data_np[:, :, feature_idx]
        global_feature_values = global_feature_values.flatten()
        feature_lower, feature_upper = np.percentile(global_feature_values, [2.5, 97.5])
        
        global_shap_values = explainer.shap_values[1][branch_idx][:, :, feature_idx].flatten()
        shap_lower, shap_upper = np.percentile(global_shap_values, [2.5, 97.5])
        
        data_array = explainer.test_seq_data_np
    elif input_type == 'static':
        branch_idx = 1
        ts_length = explainer.test_seq_data_np.shape[1]
        if explainer.scaler is not None:
            global_feature_values = explainer.test_static_data_np[:, feature_idx] * \
                                    explainer.scaler.scale_[feature_idx] + \
                                    explainer.scaler.mean_[feature_idx]
        else:
            global_feature_values = explainer.test_static_data_np[:, feature_idx]
        global_feature_values = np.repeat(global_feature_values, ts_length)
        feature_lower, feature_upper = np.percentile(global_feature_values, [2.5, 97.5])
        
        global_shap_values = explainer.shap_values[1][branch_idx][:, feature_idx]
        global_shap_values = np.repeat(global_shap_values, ts_length)
        shap_lower, shap_upper = np.percentile(global_shap_values, [2.5, 97.5])
        
        data_array = explainer.test_static_data_np
    else:
        raise ValueError("input_type must be either 'sequential' or 'static'.")
    
    feature_range = feature_upper - feature_lower
    shap_range = shap_upper - shap_lower
    feature_lower_adj = feature_lower - 0.5 * feature_range
    feature_upper_adj = feature_upper + 0.5 * feature_range
    shap_lower_adj = shap_lower - 0.5 * shap_range
    shap_upper_adj = shap_upper + 0.5 * shap_range
    
    time_steps = np.arange(ts_length)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Feature Value")
    ax2 = ax1.twinx()
    ax2.set_ylabel("SHAP Value [Non-Survival]")
    
    annotations = []
    
    for idx in indices:
        if input_type == 'sequential':
            raw_values = data_array[idx, :, feature_idx]
        else:
            raw_value = data_array[idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
        
        if explainer.scaler is not None:
            try:
                feature_scale = explainer.scaler.scale_[feature_idx]
                feature_mean = explainer.scaler.mean_[feature_idx]
                feature_values = raw_values * feature_scale + feature_mean
            except Exception as e:
                print("Error in descaling:", e)
                feature_values = raw_values
        else:
            feature_values = raw_values
        
        ax1.plot(time_steps, feature_values, color="black", linewidth=1.5, alpha=0.7)
        
        shap_vals = explainer.shap_values[1][branch_idx]
        if input_type == 'sequential':
            shap_vals_non_surv = shap_vals[idx, :, feature_idx]
        else:
            shap_val_single = shap_vals[idx, feature_idx]
            shap_vals_non_surv = np.repeat(shap_val_single, ts_length)
        ax2.plot(time_steps, shap_vals_non_surv, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        
        actual_label_val = explainer.test_labels_np[idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"
        if hasattr(explainer, "predictions") and (explainer.predictions is not None):
            pred = explainer.predictions[idx]
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
        annotations.append(f"Sample {idx}: Predicted: {predicted_class_str} (P={predicted_prob:.2f}), Actual: {actual_label_str}")
    
    ax1.set_ylim(feature_lower_adj, feature_upper_adj)
    ax2.set_ylim(shap_lower_adj, shap_upper_adj)
    
    ax1.plot([], [], color="black", linewidth=1.5, label="Feature Value")
    ax2.plot([], [], color="red", linestyle="--", linewidth=1.5, label="SHAP (Non Survival)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.title(f"Common Plot for Feature: {feature_to_explain}\n(Samples: {indices})")
    plt.subplots_adjust(bottom=0.25)
    
    ax_annot = fig.add_axes([0.1, 0.01, 0.8, 0.15])
    ax_annot.axis("off")
    annotation_text = "\n".join(annotations)
    ax_annot.text(0.5, 0.5, annotation_text,
                  ha='center', va='center', fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    plt.show()




# --- Set the feature to plot ---
# 0: mps | 1: gcs | 6: wbc | 7: platelets | 5: hr 

feature_idx = 0
feature_to_explain = explainer.feature_names[feature_idx] if explainer.feature_names is not None else f"Feature {feature_idx}"

# --- Plot common plots for each group ---
print("Common Plot for Actual Survival / Predicted Survival:")
common_plot_for_indices(selected_pred_surv_actual_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

print("Common Plot for Actual Survival / Predicted Non Survival:")
common_plot_for_indices(selected_pred_non_surv_actual_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

print("Common Plot for Actual Survival / Uncertain:")
common_plot_for_indices(selected_uncertain_actual_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

print("Common Plot for Actual Non Survival / Predicted Survival:")
common_plot_for_indices(selected_pred_surv_actual_non_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

print("Common Plot for Actual Non Survival / Predicted Non Survival:")
common_plot_for_indices(selected_pred_non_surv_actual_non_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

print("Common Plot for Actual Non Survival / Uncertain:")
common_plot_for_indices(selected_uncertain_actual_non_surv, explainer, feature_idx, feature_to_explain, input_type='sequential')

# %%
