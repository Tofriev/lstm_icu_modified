from classes.dataset_manager import DatasetManager
from classes.trainer import Trainer
import torch

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
        if self.parameters['dataset_type'] == 'mimic_mimic':
            trainer.train(self.sequences['mimic']['train'], self.sequences['mimic']['test'])
        if self.parameters['dataset_type'] == 'tudd_tudd':
            trainer.train(self.sequences['tudd']['train'], self.sequences['tudd']['test'])
        elif self.parameters['dataset_type'] == 'mimic_tudd':
            trainer.train(self.sequences['mimic']['train'], self.sequences['tudd']['test'])
        elif self.parameters['dataset_type'] == 'tudd_mimic':
            trainer.train(self.sequences['tudd']['train'], self.sequences['mimic']['test'])
        