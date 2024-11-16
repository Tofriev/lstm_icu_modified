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
    def __init__(
            self, 
            variables: list, 
            parameters = {}
            ):
        # dataset types: mimic_mimic, mimic_tudd, tudd_tudd, 
        # tudd_mimic, fractional_mimic_tudd, fractional_tudd_mimic
        self.mimic_datapath = os.path.join(project_root, 'data/raw/mimiciv/')
        self.tudd_datapath = os.path.join(project_root, 'data/raw/tudd/')
        self.variables = variables
        self.dataset_type = parameters['dataset_type']
        self.parameters = parameters
        self.data = {}

    def load_data(self):
        if 'mimic' in self.dataset_type:
            self.data['mimic'] = {}
            self.load_mimic()
            if self.parameters.get('small_data', False):
                self.reduce_data()
            self.data['sequences'], self.feature_index_mapping_mimic = self.preprocess()
        if 'tudd' in self.dataset_type:
            self.data['tudd'] = {}
            self.load_tudd()
            self.data['sequences'], self.feature_index_mapping_tudd = self.preprocess()
       
        if hasattr(self, 'feature_index_mapping_mimic') and hasattr(self, 'feature_index_mapping_tudd'):
            if self.feature_index_mapping_mimic != self.feature_index_mapping_tudd:
                print("Warning: Feature index mappings for MIMIC and TUDD datasets do not match.")
                print(self.feature_index_mapping_mimic)
                print(self.feature_index_mapping_tudd)

        

    
    def preprocess(self):
        sequences_dict = {}
        feature_index_mapping = {}
        
        if 'mimic' in self.data:
            preprocessor_mimic = Preprocessor(
                {'mimic': self.data['mimic']},
                self.variables,
                self.parameters
            )
            sequences_mimic, feature_index_mapping_mimic = preprocessor_mimic.process()
            sequences_dict.update(sequences_mimic)
            mimic_scaler = preprocessor_mimic.mimic_scaler
            ALL_FEATURES_MIMIC = preprocessor_mimic.ALL_FEATURES_MIMIC
            feature_index_mapping.update(feature_index_mapping_mimic)
        
        if 'tudd' in self.data:
            preprocessor_tudd = Preprocessor(
                {'tudd': self.data['tudd']},
                self.variables,
                self.parameters,
                mimic_scaler,
                ALL_FEATURES_MIMIC
            )
            sequences_tudd, feature_index_mapping_tudd = preprocessor_tudd.process()
            sequences_dict.update(sequences_tudd)
            feature_index_mapping.update(feature_index_mapping_tudd)

        return sequences_dict, feature_index_mapping

    def load_mimic(self):
        for variable, _ in self.variables.items():
            file_path = os.path.join(self.mimic_datapath, f"{variable}.csv")
            if os.path.exists(file_path):
                self.data['mimic'][variable] = pd.read_csv(file_path)
                if variable == 'static_data':
                    static_data_keys = self.variables['static_data'].keys()
                    self.data['mimic'][variable] = self.data['mimic'][variable][list(static_data_keys)]
                if self.data['mimic'][variable].empty:
                    print(f"Warning: {variable}.csv was loaded but contains no data.")
            else:
                print(f"Warning: {variable}.csv does not exist in {self.mimic_datapath}")
        print(self.data['mimic']['static_data'].keys())

    def load_tudd(self):
        file_path = os.path.join(self.tudd_datapath, 'tudd_complete.csv')
        if os.path.exists(file_path):
            self.data['tudd']['measurements'] = pd.read_csv(file_path, sep='|')
        else:
            raise FileNotFoundError(f"{file_path} does not exist.")

        # mortality info
        mortality_info_x_path = os.path.join(self.tudd_datapath, 'stays_ane.csv')
        mortality_info_y_path = os.path.join(self.tudd_datapath, 'stays_others2_ane.csv')

        mortality_info_list = []
        for path in [mortality_info_x_path, mortality_info_y_path]:
            if os.path.exists(path):
                mortality_info_list.append(pd.read_csv(path, sep='|'))
            else:
                raise FileNotFoundError(f"{path} does not exist.")

        self.data['tudd']['mortality_info'] = pd.concat(mortality_info_list, ignore_index=True)

    def reduce_data(self): # TODO not implemented for tudd yet and also fot mimic_tudd
        if self.dataset_type == 'mimic_mimic':
            static = self.data['mimic']['static_data']
            static_small = train_test_split(static, test_size=0.9, stratify=static[self.parameters['target']])[0]
            stay_ids = static_small['stay_id']
            for variable, attrs in self.variables.items():
                if 'stay_id' in self.data['mimic'][variable].columns:
                    self.data['mimic'][variable] = self.data['mimic'][variable][self.data['mimic'][variable]['stay_id'].isin(stay_ids)]
            self.data['mimic']['static_data'] = static_small
        elif self.dataset_type == 'tudd_tudd':
            raise NotImplementedError("Method for reducing TUDD data is not implemented yet.")

   
        

