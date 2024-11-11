from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
import hashlib
import os 
import csv

class Pipeline (object):
    def __init__(self, variables, parameters, show=False):
        self.variables = variables
        self.parameters = parameters
        self.show = show
        

    def prepare_data(self):
        DataManager = DatasetManager(variables=self.variables, parameters=self.parameters)
        DataManager.load_data()
        self.sequences = DataManager.data['sequences']

        # if self.show:
        #     seq, label = self.sequences['train'][0]
        #     print(seq.shape)
        #     print(label.shape)
        #     print(dict(
        #         sequence=torch.tensor(seq, dtype=torch.float32),
        #         label=torch.tensor(label).long(),
        #         ))
            
    def train(self):
        trainer = Trainer(self.parameters)
        if self.parameters['dataset_type'] == 'mimic_tudd_fract' and self.parameters.get('fractional_steps'):
            print("Training fractional")
            self.result_dict = trainer.train_fractional(self.sequences)
        elif self.parameters['dataset_type'] == 'mimic_mimic':
            self.result_dict = trainer.train(self.sequences['mimic']['train'], self.sequences['mimic']['test'])
        elif self.parameters['dataset_type'] == 'tudd_tudd':
            self.result_dict = trainer.train(self.sequences['tudd']['train'], self.sequences['tudd']['test'])
        elif self.parameters['dataset_type'] == 'mimic_tudd':
            self.result_dict = trainer.train(self.sequences['mimic']['train'], self.sequences['tudd']['test'])
        elif self.parameters['dataset_type'] == 'tudd_mimic':
            self.result_dict = trainer.train(self.sequences['tudd']['train'], self.sequences['mimic']['test'])

    def memorize(self, file_path='parameters_results.csv'):
        if self.parameters['fractional_steps']:
            entry = {**self.parameters, **self.result_dict}
        else:
            entry = {**self.parameters, **self.result_dict[0]}
        
        params_hash = hashlib.md5(str(sorted(entry.items())).encode()).hexdigest()
        entry['parameters_hash'] = params_hash

        entry_exists = False
        if os.path.exists(file_path):
            with open(file_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('parameters_hash') == params_hash:
                        entry_exists = True
                        break

        if not entry_exists:
            fieldnames = list(entry.keys())
            with open(file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:  
                    writer.writeheader()
                writer.writerow(entry)
        







    