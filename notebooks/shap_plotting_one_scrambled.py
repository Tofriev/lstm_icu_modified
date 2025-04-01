#%%
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
    
    if actual == 0:  
        if p_surv >= high_threshold:
            pred_surv_actual_surv.append(idx)
        elif p_non_surv >= high_threshold:
            pred_non_surv_actual_surv.append(idx)
        elif is_uncertain:
            uncertain_actual_surv.append(idx)
    elif actual == 1:  
        if p_surv >= high_threshold:
            pred_surv_actual_non_surv.append(idx)
        elif p_non_surv >= high_threshold:
            pred_non_surv_actual_non_surv.append(idx)
        elif is_uncertain:
            uncertain_actual_non_surv.append(idx)

# --- Select 5 samples from each group (if available) ---
sel_pred_surv_actual_surv = pred_surv_actual_surv[:5]
sel_pred_non_surv_actual_surv = pred_non_surv_actual_surv[:5]
sel_uncertain_actual_surv = uncertain_actual_surv[:5]

sel_pred_surv_actual_non_surv = pred_surv_actual_non_surv[:5]
sel_pred_non_surv_actual_non_surv = pred_non_surv_actual_non_surv[:5]
sel_uncertain_actual_non_surv = uncertain_actual_non_surv[:5]

# --- Define groups and their labels for the subplots ---
groups = [
    (sel_pred_surv_actual_surv, "Actual Survival / Predicted Survival"),
    (sel_pred_non_surv_actual_surv, "Actual Survival / Predicted Non Survival"),
    (sel_uncertain_actual_surv, "Actual Survival / Uncertain"),
    (sel_pred_surv_actual_non_surv, "Actual Non Survival / Predicted Survival"),
    (sel_pred_non_surv_actual_non_surv, "Actual Non Survival / Predicted Non Survival"),
    (sel_uncertain_actual_non_surv, "Actual Non Survival / Uncertain")
]

print("Selected indices:")
print("Actual Survival / Predicted Survival:", sel_pred_surv_actual_surv)
print("Actual Survival / Predicted Non Survival:", sel_pred_non_surv_actual_surv)
print("Actual Survival / Uncertain:", sel_uncertain_actual_surv)
print("Actual Non Survival / Predicted Survival:", sel_pred_surv_actual_non_surv)
print("Actual Non Survival / Predicted Non Survival:", sel_pred_non_surv_actual_non_surv)
print("Actual Non Survival / Uncertain:", sel_uncertain_actual_non_surv)

# --- Set feature to plot ---
feature_idx = 7
feature_to_explain = explainer.feature_names[feature_idx] if explainer.feature_names is not None else f"Feature {feature_idx}"

ts_length = explainer.test_seq_data_np.shape[1]
branch_idx = 0
if explainer.scaler is not None:
    global_feature_vals = explainer.test_seq_data_np[:, :, feature_idx] * explainer.scaler.scale_[feature_idx] + explainer.scaler.mean_[feature_idx]
else:
    global_feature_vals = explainer.test_seq_data_np[:, :, feature_idx]
global_feature_vals = global_feature_vals.flatten()
feat_lower, feat_upper = np.percentile(global_feature_vals, [2.5, 97.5])

global_shap_vals = explainer.shap_values[1][branch_idx][:, :, feature_idx].flatten()
shap_lower, shap_upper = np.percentile(global_shap_vals, [2.5, 97.5])

feat_range = feat_upper - feat_lower
shap_range = shap_upper - shap_lower
feat_lower_adj = feat_lower - 0.1 * feat_range
feat_upper_adj = feat_upper + 0.1 * feat_range
shap_lower_adj = shap_lower - 0.1 * shap_range
shap_upper_adj = shap_upper + 0.1 * shap_range

# --- Create a combined figure with 6 subplots (3 rows x 2 columns) ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18), sharex=True)
axes = axes.flatten()

for i, (group_indices, group_label) in enumerate(groups):
    ax1 = axes[i]
    ax2 = ax1.twinx()
    ax1.set_title(group_label)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Feature Value")
    ax2.set_ylabel("SHAP Value [Non-Survival]")
    
    annotations = []
    time_steps = np.arange(ts_length)
    
    for idx in group_indices:
        raw_vals = explainer.test_seq_data_np[idx, :, feature_idx]
        if explainer.scaler is not None:
            try:
                scale = explainer.scaler.scale_[feature_idx]
                mean = explainer.scaler.mean_[feature_idx]
                feat_vals = raw_vals * scale + mean
            except Exception as e:
                print("Descaling error:", e)
                feat_vals = raw_vals
        else:
            feat_vals = raw_vals
        
        ax1.plot(time_steps, feat_vals, color="black", linewidth=1.5, alpha=0.7)
        
        shap_vals = explainer.shap_values[1][branch_idx][idx, :, feature_idx]
        ax2.plot(time_steps, shap_vals, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        
        actual = explainer.test_labels_np[idx]
        actual_str = "Survival" if actual == 0 else "Non Survival"
        pred = explainer.predictions[idx]
        p_surv = pred["p_surv"]
        p_non_surv = pred["p_non_surv"]
        if p_surv >= p_non_surv:
            pred_str = "Survival"
            pred_prob = p_surv
        else:
            pred_str = "Non Survival"
            pred_prob = p_non_surv
        annotations.append(f"S{idx}: Pred:{pred_str} (P={pred_prob:.2f}), Act:{actual_str}")
    
    ax1.set_ylim(feat_lower_adj, feat_upper_adj)
    ax2.set_ylim(shap_lower_adj, shap_upper_adj)
    
    ax1.plot([], [], color="black", linewidth=1.5, label="Feature Value")
    ax2.plot([], [], color="red", linestyle="--", linewidth=1.5, label="SHAP (Non Survival)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    ann_text = "\n".join(annotations)
    ax1.text(0.5, 0.05, ann_text, transform=ax1.transAxes, fontsize=8,
             ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
plt.suptitle(f"Combined Common Plot for {feature_to_explain}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
