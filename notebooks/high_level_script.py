#%%
from classes.dataset_manager import DatasetManager
from classes.pipeline import Pipeline
from dataclasses import dataclass
from utils import set_seed
import json
import os

set_seed(42)

# high level functionalities work maily for mimic (yet)



variables = {
    'mbp': {'type': 'numerical', 'sequence': True, 'training': True},             # Mean Blood Pressure
      'gcs_total': {'type': 'numerical', 'sequence': True, 'training': True},       # Glasgow Coma Scale Total
    'glc': {'type': 'numerical', 'sequence': True, 'training': True},             # Glucose levels
    #'resprate_mortality': {'type': 'numerical', 'sequence': True, 'training': True},  # Respiratory Rate
    'creatinine': {'type': 'numerical', 'sequence': True, 'training': True},      # Creatinine levels
    'potassium': {'type': 'numerical', 'sequence': True, 'training': True},       # Potassium levels
    'hr': {'type': 'numerical', 'sequence': True, 'training': True},              # Heart Rate
    #'sodium': {'type': 'numerical', 'sequence': True, 'training': True},          # Sodium levels
    'wbc': {'type': 'numerical', 'sequence': True, 'training': True},             # White Blood Cells (leukocytes)
    'platelets': {'type': 'numerical', 'sequence': True, 'training': True},       # Platelets (thrombocytes)
    #'inr': {'type': 'numerical', 'sequence': True, 'training': True},             # International Normalized Ratio (Prothrombin Time)
    #'anion_gap': {'type': 'numerical', 'sequence': True, 'training': True},       # Anion Gap
    #'lactate': {'type': 'numerical', 'sequence': True, 'training': True},         # Lactate levels
    #'urea': {'type': 'numerical', 'sequence': True, 'training': True},            # Urea levels
    'temperature': {'type': 'numerical', 'sequence': True, 'training': True},     # Body Temperature
    'weight': {'type': 'numerical', 'sequence': True, 'training': True},          # Weight over time (time series)
    'static_data': {
        'mortality': {'type': 'target', 'sequence': False, 'training': False},     # Mortality outcome
        'age': {'type': 'numerical', 'sequence': False, 'training': True},        # Age
        'gender': {'type': 'categorical', 'sequence': False, 'training': True},   # Gender
        #'height': {'type': 'numerical', 'sequence': False, 'training': True},     # Height
        'intime': {'type': 'datetime', 'sequence': False, 'training': False},
        'first_day_end': {'type': 'datetime', 'sequence': False, 'training': False},
        'stay_id': {'type': 'id', 'sequence': False, 'training': False}
    }
}

def count_features(variables):
    exclude_keys = {'mortality', 'intime', 'first_day_end', 'stay_id', 'static_data'}
    
    top_level_keys = [key for key in variables if key not in exclude_keys]
    count = len(top_level_keys)
    count += len([key for key in variables['static_data'] if key not in exclude_keys])
    
    return count
n_features = count_features(variables)




# model parameters work also for tudd
parameters = {
    'target':'mortality',
    #'dataset_type': 'tudd_tudd',
    #'dataset_type': 'mimic_mimic',
    #'dataset_type': 'tudd_mimic',
    'dataset_type': 'mimic_tudd',
    'golden_tudd': True,
    #'dataset_type': 'mimic_tudd_fract',
    #'fractional_steps': 1000, # example for mimic_tudd: adds 1000 samples from tudd train to the training set of mimic for every fraction
    'small_data': False, # not implemented for tudd yet 
    'aggregation_frequency': 'H',
    'imputation': {'method': 'ffill_bfill'},#, 'n_neighbors': 3}, # ffilll uses mean for features without any values
    'sampling': {'method': 'undersampling', 'sampling_strategy': 0.1}, #minority / majority class = sampling streategy
    'scaling': 'Standard', # Standard and MinMax implemented, also try Robust
    #'scaling_range': [0, 1],
    #'n_features': n_features,
    'n_features': n_features,
    'models': ['lstm'],
    #'models': ['multi_channel_lstm'],
    #'models': ['cnn_lstm'],
    #'models': ['attention_lstm'],
    'compare_distributions': True,
    'shuffle': True,
    

    'model_parameters': {
        'lstm': { # model config will be input to model __init__
            'n_hidden': 100,
            'n_layers': 2, # adjust this for stacked lstm
            'n_classes': 2, # also adjust the steps in lstm model for correct auroc calculation if this changes
            'dropout': 0.75,
            'bidirectional': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 3.0],
            'batch_size': 32,
            'n_epochs': 6,
            'gradient_clip_val': 1,
        }, 
        'multi_channel_lstm': {
            'n_hidden': 100,
            'n_layers': 1,
            'n_classes': 2, # also adjust the steps in lstm model for correct auroc calculation if this changes
            'dropout': 0.75,
            'bidirectional': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 3.0],
            'batch_size': 32,
            'n_epochs': 5,
            'gradient_clip_val': 1,
        },
        'cnn_lstm': {
            'n_hidden': 100,
            'n_layers': 1,
            'n_classes': 2, # also adjust the steps in lstm model for correct auroc calculation if this changes
            'dropout': 0.75,
            'bidirectional': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 3.0],
            'batch_size': 32,
            'n_epochs': 5,
            'gradient_clip_val': 1,
            'architecture': 'parallel',  # 'cnn_lstm', 'lstm_cnn', 'parallel'
            'cnn_out_channels': 64,
            'kernel_size': 3,

        },
        'attention_lstm': {
            'n_hidden': 100,
            'n_layers': 1,
            'n_classes': 2, # also adjust the steps in lstm model for correct auroc calculation if this changes
            'dropout': 0.75,
            'bidirectional': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 3.0],
            'batch_size': 32,
            'n_epochs': 5,
            'gradient_clip_val': 1,
            'attention_type':'dot' # 'additive', 'dot'
        } 
    },
}
   



#%%
pipe = Pipeline(variables=variables, parameters=parameters, show=True)
pipe.prepare_data()
print(pipe.feature_index_mapping)
pipe.visualize_sequences()
pipe.train()
pipe.memorize()

print(pipe.result_dict)



#%%
# only for fractional learning: use cell above otherwise
model_names = ['attention_lstm']
#model_names = ['lstm', 'multi_channel_lstm', 'cnn_lstm', 'attention_lstm']

json_file = 'results.json'
for model_name in model_names:
    parameters['models'] = [model_name]
    pipe = Pipeline(variables=variables, parameters=parameters, show=True)
    
    pipe.prepare_data()
    pipe.train()
    print(pipe.result_dict)
    pipe.result_dict['model_name'] = model_name
    if not os.path.exists(json_file) or os.stat(json_file).st_size == 0:
        with open(json_file, 'w') as file:
            json.dump([], file)

    with open(json_file, 'r') as file:
        data = json.load(file)

    data.append(pipe.result_dict)
    with open(json_file, 'w') as file:
        json.dump(data, file)

  

#%%

# %%
pipe.memorize()



# %%


