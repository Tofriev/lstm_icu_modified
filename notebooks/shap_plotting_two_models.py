#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from classes.explainer import LoadSHAPExplainer



model1_name = "Lstm"
model2_name = "MultiChannelLstm"

model_name = "lstm_static"  
num_samples = 1000      
dataset_name = 'combined_tudd' 

explainer = LoadSHAPExplainer(model=None, feature_names=None)
explainer.load_shap_values(model_name, num_samples, dataset_name)

high_threshold = 0.8
low_threshold_lower = 0.4
low_threshold_upper = 0.6

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

selected_pred_surv_actual_surv = pred_surv_actual_surv[:5]
selected_pred_non_surv_actual_surv = pred_non_surv_actual_surv[:5]
selected_uncertain_actual_surv = uncertain_actual_surv[:5]

selected_pred_surv_actual_non_surv = pred_surv_actual_non_surv[:5]
selected_pred_non_surv_actual_non_surv = pred_non_surv_actual_non_surv[:5]
selected_uncertain_actual_non_surv = uncertain_actual_non_surv[:5]

selected_indices = (selected_pred_surv_actual_surv +
                    selected_pred_non_surv_actual_surv +
                    selected_uncertain_actual_surv +
                    selected_pred_surv_actual_non_surv +
                    selected_pred_non_surv_actual_non_surv +
                    selected_uncertain_actual_non_surv)

print("Selected sample indices for actual survival (label=0):")
print("Predicted Survival:", selected_pred_surv_actual_surv)
print("Predicted Non Survival:", selected_pred_non_surv_actual_surv)
print("Uncertain:", selected_uncertain_actual_surv)

print("\nSelected sample indices for actual non survival (label=1):")
print("Predicted Survival:", selected_pred_surv_actual_non_surv)
print("Predicted Non Survival:", selected_pred_non_surv_actual_non_surv)
print("Uncertain:", selected_uncertain_actual_non_surv)

#  feature to explain
feature_idx = 0
feature_to_explain = explainer.feature_names[feature_idx] if explainer.feature_names is not None else f"Feature {feature_idx}"

file_path1 = "SHAP_scores/lstm_static_combined_tudd_1000.json"
file_path2 = "SHAP_scores/multi_channel_lstm_static_combined_tudd_1000.json"  

selected_indices = [idx for idx in range(len(explainer.test_seq_data_np))]
# selected_indices = selected_indices[:100]
selected_indices = [0, 5, 8, 61]

for sample_idx in selected_indices:
    explainer.plot_single_feature_time_shap_two_models(
        file_path1=file_path1,
        file_path2=file_path2,
        sample_idx=sample_idx,
        feature_to_explain=feature_to_explain,
        model1_name=model1_name,
        model2_name=model2_name,
        feature_idx=feature_idx,
        input_type='sequential'
    )



# %%
