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
        if self.dataset_type == 'mimic_mimic':
            self.data['mimic'] = {}
            self.load_mimic()
        elif self.dataset_type == 'tudd_tudd':
            self.data['tudd'] = {}
            self.load_tudd()
        elif self.dataset_type == 'mimic_tudd' or self.dataset_type == 'tudd_mimic':
            self.data['tudd'] = {}
            self.data['mimic'] = {}
            self.load_mimic()
            self.load_tudd()
        if self.parameters.get('small_data', False):
            self.reduce_data()
        sequences = self.preprocess()
        self.data['sequences'] = sequences
       

    def load_mimic(self):
        for variable, _ in self.variables.items():
            file_path = os.path.join(self.mimic_datapath, f"{variable}.csv")
            if os.path.exists(file_path):
                self.data['mimic'][variable] = pd.read_csv(file_path)
                if self.data['mimic'][variable].empty:
                    print(f"Warning: {variable}.csv was loaded but contains no data.")
            else:
                print(f"Warning: {variable}.csv does not exist in {self.mimic_datapath}")

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

    def reduce_data(self):
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

    def preprocess(self):
        preprocessor = Preprocessor(self.data, self.variables, self.parameters)
        sequences = preprocessor.process()
        return sequences
       
        

