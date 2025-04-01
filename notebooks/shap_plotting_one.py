#%%
import math
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
low_threshold_upper = 0.6

# --- Group the sample indices by actual label and prediction ---
pred_surv_actual_surv = []     
pred_non_surv_actual_surv = [] 
uncertain_actual_surv = []      

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

# --- Select 5 samples from each group---
sel_pred_surv_actual_surv = pred_surv_actual_surv[:5]
sel_pred_non_surv_actual_surv = pred_non_surv_actual_surv[:5]
sel_uncertain_actual_surv = uncertain_actual_surv[:5]

sel_pred_surv_actual_non_surv = pred_surv_actual_non_surv[:5]
sel_pred_non_surv_actual_non_surv = pred_non_surv_actual_non_surv[:5]
sel_uncertain_actual_non_surv = uncertain_actual_non_surv[:5]

groups = [
    (sel_pred_surv_actual_surv, "Actual Survival / Predicted Survival"),
    (sel_pred_non_surv_actual_surv, "Actual Survival / Predicted Non Survival"),
    (sel_uncertain_actual_surv, "Actual Survival / Uncertain"),
    (sel_pred_surv_actual_non_surv, "Actual Non Survival / Predicted Survival"),
    (sel_pred_non_surv_actual_non_surv, "Actual Non Survival / Predicted Non Survival"),
    (sel_uncertain_actual_non_surv, "Actual Non Survival / Uncertain")
]

selected_indices = []
for (group_indices, _) in groups:
    selected_indices.extend(group_indices)

# --- Set the feature to plot ---
feature_idx = 7
feature_to_explain = explainer.feature_names[feature_idx] if explainer.feature_names is not None else f"Feature {feature_idx}"

input_type = 'sequential'
branch_idx = 0
ts_length = explainer.test_seq_data_np.shape[1]
if explainer.scaler is not None:
    global_feature = explainer.test_seq_data_np[:, :, feature_idx] * explainer.scaler.scale_[feature_idx] + explainer.scaler.mean_[feature_idx]
else:
    global_feature = explainer.test_seq_data_np[:, :, feature_idx]
global_feature = global_feature.flatten()
feat_lower, feat_upper = np.percentile(global_feature, [2.5, 97.5])
feat_margin = 0.1 * (feat_upper - feat_lower)
global_feat_lower = feat_lower - feat_margin
global_feat_upper = feat_upper + feat_margin

global_shap = explainer.shap_values[1][branch_idx][:, :, feature_idx].flatten()
shap_lower, shap_upper = np.percentile(global_shap, [2.5, 97.5])
shap_margin = 0.1 * (shap_upper - shap_lower)
global_shap_lower = shap_lower - shap_margin
global_shap_upper = shap_upper + shap_margin

# --- Create a combined figure with 6 rows (groups) and 5 columns (samples per group) ---
nrows = len(groups)  # 6 groups
ncols = 5            # 5 samples per group
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True)
axes = axes.flatten()

for row, (group_indices, group_label) in enumerate(groups):
    for col, idx in enumerate(group_indices):
        ax = axes[row*ncols + col]
        time_steps = np.arange(ts_length)
        raw_vals = explainer.test_seq_data_np[idx, :, feature_idx]
        if explainer.scaler is not None:
            try:
                scale = explainer.scaler.scale_[feature_idx]
                mean = explainer.scaler.mean_[feature_idx]
                feat_vals = raw_vals * scale + mean
            except Exception as e:
                feat_vals = raw_vals
        else:
            feat_vals = raw_vals
        
        ax.plot(time_steps, feat_vals, color='black', linewidth=1.5)
        ax.set_ylim(global_feat_lower, global_feat_upper)
        ax.set_title(f"S{idx}", fontsize=10)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Feature", fontsize=8)
        
        ax2 = ax.twinx()
        shap_vals = explainer.shap_values[1][branch_idx][idx, :, feature_idx]
        ax2.plot(time_steps, shap_vals, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_ylim(global_shap_lower, global_shap_upper)
        ax2.set_ylabel("SHAP", fontsize=8)
        
        actual = explainer.test_labels_np[idx]
        actual_str = "Surv" if actual == 0 else "NonSurv"
        pred = explainer.predictions[idx]
        p_surv = pred["p_surv"]
        p_non_surv = pred["p_non_surv"]
        if p_surv >= p_non_surv:
            pred_str = "Surv"
            pred_prob = p_surv
        else:
            pred_str = "NonSurv"
            pred_prob = p_non_surv
        ax.text(0.5, 0.05, f"P:{pred_str} (P={pred_prob:.2f})\nA:{actual_str}",
                transform=ax.transAxes, fontsize=7, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
total_subplots = nrows * ncols
for j in range(len(selected_indices), total_subplots):
    axes[j].axis('off')

fig.suptitle(f"Combined Plots for {feature_to_explain}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
