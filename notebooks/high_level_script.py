# %%
from classes.dataset_manager import DatasetManager
from classes.pipeline import Pipeline
from dataclasses import dataclass
from utils import set_seed
import json
import os
from classes.pipeline import MultiDatasetPipeline
from start_utils import variables, count_features

set_seed(42)
n_features = count_features(variables)


# model parameters work also for tudd
parameters = {
    "target": "mortality",
    "dataset_type": "mimic_mimic",
    # "dataset_type": "tudd_tudd",
    # "dataset_type": ["mimic_mimic", "mimic_tudd"],
    # "dataset_type": "tudd_mimic",
    # "dataset_type": "mimic_tudd",
    # "dataset_type": [
    #     "mimic_combined",
    #     "tudd_combined",
    #     # "combined_combined",
    #     # "combined_mimic",
    #     # "combined_tudd",
    #     #     "tudd_tudd",
    #     #     "mimic_mimic",
    #     #     "mimic_tudd",
    #     #     "tudd_mimic",
    # ],
    # ,
    # "dataset_type": "combined_combined",
    # "dataset_type": "combined_mimic",
    # "dataset_type": "combined_tudd",
    # "dataset_type": [  # fract only works in single run and with new_data = true
    #     # "tudd_fract",
    #     # "mimic_fract",
    #     "mimic_tudd_fract",
    # ],
    "shuffle_mimic_tudd_fract": False,
    "fractional_steps": 200,  # example for mimic_tudd: adds 1000 samples from tudd train to the training set of mimic for every fraction. maybe try with 200.
    "small_data": False,  # not implemented for tudd yet
    "aggregation_frequency": "H",
    "imputation": {
        "method": "ffill_bfill"
    },  # , 'n_neighbors': 3}, # ffilll uses mean for features without any values
    "scaling": "Standard",  # Standard and MinMax implemented, also try Robust
    #'scaling_range': [0, 1],
    #'n_features': n_features,
    "n_features": n_features,
    # "models": ["lstm"],
    "models": [
        "lstm",
        # "multi_channel_lstm",
    ],
    # "models": ["multi_channel_lstm"],
    # "models": ["cnn_lstm"],
    #'models': ['attention_lstm'],
    "compare_distributions": False,
    "shuffle": True,
    "sparsity_check": False,  # prints
    "model_parameters": {
        "lstm": {  # model config will be input to model __init__
            "n_hidden": 100,
            "n_layers": 2,  # adjust this for stacked lstm
            "n_classes": 2,  # also adjust the steps in lstm model for correct auroc calculation if this changes
            "dropout": 0.75,
            "bidirectional": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "class_weights": [1.0, 3.0],
            "batch_size": 128,
            "n_epochs": 5,  # 5
            "gradient_clip_val": 1,
        },
        # "multi_channel_lstm": {
        #     "n_hidden": 100,
        #     "n_layers": 1,
        #     "n_classes": 2,  # also adjust the steps in lstm model for correct auroc calculation if this changes
        #     "dropout": 0.75,
        #     "bidirectional": True,
        #     "learning_rate": 1e-4,
        #     "weight_decay": 1e-5,
        #     "class_weights": [1.0, 3.0],
        #     "batch_size": 128,
        #     "n_epochs": 3,
        #     "gradient_clip_val": 1,
        # },
        "multi_channel_lstm": {
            "n_hidden": 100,
            "n_layers": 2,
            "n_classes": 2,  # also adjust the steps in lstm model for correct auroc calculation if this changes
            "dropout": 0.75,
            "bidirectional": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "class_weights": [1.0, 3.0],
            "batch_size": 128,
            "n_epochs": 6,  # 6
            "gradient_clip_val": 1,
        },
        "cnn_lstm": {
            "n_hidden": 100,
            "n_layers": 1,
            "n_classes": 2,  # also adjust the steps in lst m model for correct auroc calculation if this changes
            "dropout": 0.75,
            "bidirectional": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "class_weights": [1.0, 3.0],
            "batch_size": 32,
            "n_epochs": 5,
            "gradient_clip_val": 1,
            "architecture": "parallel",  # 'cnn_lstm', 'lstm_cnn', 'parallel'
            "cnn_out_channels": 64,
            "kernel_size": 3,
        },
        "attention_lstm": {
            "n_hidden": 100,
            "n_layers": 1,
            "n_classes": 2,  # also adjust the steps in lstm model for correct auroc calculation if this changes
            "dropout": 0.75,
            "bidirectional": True,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "class_weights": [1.0, 3.0],
            "batch_size": 32,
            "n_epochs": 5,
            "gradient_clip_val": 1,
            "attention_type": "dot",  # 'additive', 'dot'
        },
    },
}


if isinstance(parameters["dataset_type"], list) and len(parameters["dataset_type"]) > 1:
    ############################# Mutli Run ##################################################
    print("Multi Run")
    multi_pipe = MultiDatasetPipeline(
        variables=variables,
        parameters=parameters,
        dataset_types=parameters["dataset_type"],
        new_data=True,
    )
    results = multi_pipe.run_all(model_list=parameters["models"])
else:
    ############################# Single Run #################################################
    print("Single Run")
    pipe = Pipeline(variables=variables, parameters=parameters, new_data=True)
    pipe.run_experiment()
    # pipe.memorize()
    pipe.explain(model_name="lstm", method="heatmap_shap", num_samples=10)
    #pipe.explain(model_name="lstm", feature_to_explain = 'age_value', method="plot_single_feature_time_shap", num_samples=100)
    # ['mbp_value', 'gcs_total_value', 'glc_value', 'creatinine_value', 'potassium_value', 'hr_value', 'wbc_value', 'platelets_value', 'inr_value', 'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value', 'age_value', 'gender_value']

# print(pipe.result_dict)

# # %%
# multi_pipe = MultiDatasetPipeline(
#     variables=variables,
#     parameters=parameters,
#     dataset_types=parameters["dataset_type"],
#     new_data=True,
# )

# multi_pipe.prepare_data()

# results = multi_pipe.run_all(model_list=parameters["models"])


# # %%
# # only for fractional learning: use cell above otherwise
# model_names = ["lstm", "multi_channel_lstm"]
# # model_names = ['lstm', 'multi_channel_lstm', 'cnn_lstm', 'attention_lstm']

# json_file = "results.json"
# for model_name in model_names:
#     parameters["models"] = [model_name]
#     pipe = Pipeline(variables=variables, parameters=parameters, show=True)

#     pipe.prepare_data()
#     pipe.train()
#     print(pipe.result_dict)
#     pipe.result_dict["model_name"] = model_name
#     if not os.path.exists(json_file) or os.stat(json_file).st_size == 0:
#         with open(json_file, "w") as file:
#             json.dump([], file)

#     with open(json_file, "r") as file:
#         data = json.load(file)

#     data.append(pipe.result_dict)
#     with open(json_file, "w") as file:
#         json.dump(data, file)


# # %%

# # %%
# pipe.memorize()


# # %%

# %%
