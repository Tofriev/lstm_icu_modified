#%%
from classes.dataset_manager import DatasetManager
from classes.pipeline import Pipeline
from dataclasses import dataclass
from utils import set_seed

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
    'inr': {'type': 'numerical', 'sequence': True, 'training': True},             # International Normalized Ratio (Prothrombin Time)
    'anion_gap': {'type': 'numerical', 'sequence': True, 'training': True},       # Anion Gap
    'lactate': {'type': 'numerical', 'sequence': True, 'training': True},         # Lactate levels
    #'urea': {'type': 'numerical', 'sequence': True, 'training': True},            # Urea levels
    'temperature': {'type': 'numerical', 'sequence': True, 'training': True},     # Body Temperature
    'weight': {'type': 'numerical', 'sequence': True, 'training': True},          # Weight over time (time series)
    'static_data': {
        'mortality': {'type': 'target', 'sequence': False, 'training': False},     # Mortality outcome
        'age': {'type': 'numerical', 'sequence': False, 'training': True},        # Age
        'gender': {'type': 'categorical', 'sequence': False, 'training': True},   # Gender
        'height': {'type': 'numerical', 'sequence': False, 'training': True},     # Height
        'intime': {'type': 'datetime', 'sequence': False, 'training': False},
        'first_day_end': {'type': 'datetime', 'sequence': False, 'training': False},
        'stay_id': {'type': 'id', 'sequence': False, 'training': False}
    }
}

n_features = sum(1 for v in variables.values() if isinstance(v, dict) and v.get('sequence', False))
n_features += sum(1 for v in variables['static_data'].values() if isinstance(v, dict) and v.get('sequence', False))
print(f'n features (should be 16): {n_features}') 


# model parameters work also for tudd
parameters = {
    'target':'mortality',
    #'dataset_type': 'tudd_tudd',
    'dataset_type': 'mimic_mimic',
    #'dataset_type': 'tudd_mimic',
    'small_data': True, # not implemented for tudd yet 
    'aggregation_frequency': 'H',
    'imputation': {'method': 'ffill_bfill'}, # uses mean for features without any values
    'sampling': {'method': 'undersampling', 'sampling_strategy': 0.1}, #minority / majority class = sampling streategy
    'scaling': 'standard', # also try MinMax, and Robust
    #'n_features': n_features,
    'n_features': 16,
    'models': ['lstm'],  
    'model_parameters': {
        'lstm': { # model config will be input to model __init__
            'n_hidden': 100,
            'n_layers': 2,
            'n_classes': 2, # also adjust the steps in lstm model for correct auroc calculation if this changes
            'dropout': 0.75,
            'bidirectional': True,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'class_weights': [1.0, 3.0],
            'batch_size': 32,
            'n_epochs': 3,
            'gradient_clip_val': 1,
        },  
    },  
}


pipe = Pipeline(variables=variables, parameters=parameters, show=True)
pipe.prepare_data()
pipe.train()



# %%
 