import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from classes.preprocessor import Preprocessor
from utils import set_seed

set_seed(42)


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


class DatasetManager:
    def __init__(self, variables: list, parameters={}):
        # dataset types: mimic_mimic, mimic_tudd, tudd_tudd,
        # tudd_mimic, fractional_mimic_tudd, fractional_tudd_mimic
        self.mimic_datapath = os.path.join(project_root, "data/raw/mimiciv/first_24h/")
        self.tudd_datapath = os.path.join(project_root, "data/raw/tudd/")
        self.variables = variables
        self.dataset_type = parameters["dataset_type"]
        self.parameters = parameters
        self.data = {}

    def load_data(self):
        if "mimic" in self.dataset_type:
            self.data["mimic"] = {}
            self.load_mimic()
            if self.parameters.get("small_data", False):
                self.reduce_data()
            self.preprocess("mimic")

        if "tudd" in self.dataset_type:
            self.data["tudd"] = {}
            self.load_tudd()
            self.preprocess("tudd")

    def preprocess(self, data_type: str):
        sequences_dict = {}
        feature_index_mapping = {}

        if data_type == "mimic":
            preprocessor_mimic = Preprocessor(
                data_type,
                self.data["mimic"],
                self.variables,
                self.parameters,
            )
            preprocessor_mimic.process()
            # has the attributes: data_process (dict with:
            # pre_processing, aggregated, merged, imputed, scaled, sequences,
            # sequences_train, sequences_test
            # ),
            # feature_index_mapping_sequences, scaler,
            self.data["mimic"] = preprocessor_mimic.data_process

            scaler = preprocessor_mimic.scaler

        if "tudd" in self.data:
            preprocessor_args = {
                "data_type": data_type,
                "data": self.data["tudd"],
                "variables": self.variables,
                "parameters": self.parameters,
            }

            # include mimic scaler if it exists
            if hasattr(self, "scaler"):
                preprocessor_args["scaler"] = self.scaler

            preprocessor_tudd = Preprocessor(**preprocessor_args)

            preprocessor_tudd.process()
            self.data["tudd"] = preprocessor_tudd.data_process

    def load_mimic(self):
        print("Loading MIMIC data...")
        for variable, _ in self.variables.items():
            file_path = os.path.join(self.mimic_datapath, f"{variable}.csv")
            if os.path.exists(file_path):
                self.data["mimic"][variable] = pd.read_csv(file_path)
                if variable == "static_data":
                    no_val_keys = {"intime", "first_day_end", "stay_id"}
                    static_data_keys = [
                        f"{key}_value" if key not in no_val_keys else key
                        for key in self.variables["static_data"].keys()
                    ]
                    # make sure that only the statric data that is in the variables dict in high level_script gets loaded
                    self.data["mimic"]["static_data"] = self.data["mimic"][variable][
                        list(static_data_keys)
                    ]

            else:
                print(
                    f"Warning: {variable}.csv does not exist in {self.mimic_datapath}"
                )

    def load_tudd(self):
        file_path = os.path.join(self.tudd_datapath, "tudd_incomplete.csv")
        if os.path.exists(file_path):
            self.data["tudd"]["measurements"] = pd.read_csv(file_path, sep="|")
        else:
            raise FileNotFoundError(f"{file_path} does not exist.")

        mortality_info_path = os.path.join(self.tudd_datapath, "stays_ane_new.csv")
        if os.path.exists(mortality_info_path):
            self.data["tudd"]["mortality_info"] = pd.read_csv(
                mortality_info_path, sep="|"
            )
        else:
            raise FileNotFoundError(f"{mortality_info_path} does not exist.")

        # # mortality info
        # mortality_info_x_path = os.path.join(self.tudd_datapath, "stays_ane.csv")
        # mortality_info_y_path = os.path.join(
        #     self.tudd_datapath, "stays_others2_ane.csv"
        # )

        # mortality_info_list = []
        # for path in [mortality_info_x_path, mortality_info_y_path]:
        #     if os.path.exists(path):
        #         mortality_info_list.append(pd.read_csv(path, sep="|"))
        #     else:
        #         raise FileNotFoundError(f"{path} does not exist.")

        # self.data["tudd"]["mortality_info"] = pd.concat(
        #     mortality_info_list, ignore_index=True
        # )

    def reduce_data(self):  # TODO not implemented for tudd yet and also fot mimic_tudd
        if self.dataset_type == "mimic_mimic":
            static = self.data["mimic"]["static_data"]
            static_small = train_test_split(
                static,
                test_size=0.9,
                stratify=static[f'{self.parameters["target"]}_value'],
            )[0]
            stay_ids = static_small["stay_id"]
            for variable in self.data["mimic"].keys():
                if variable != "static_data":
                    if "stay_id" in self.data["mimic"][variable].columns:
                        self.data["mimic"][variable] = self.data["mimic"][variable][
                            self.data["mimic"][variable]["stay_id"].isin(stay_ids)
                        ]
            self.data["mimic"]["static_data"] = static_small
        elif self.dataset_type == "tudd_tudd":
            raise NotImplementedError(
                "Method for reducing TUDD data is not implemented yet."
            )
