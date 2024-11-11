import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from classes.data_module import IcuDataModule
from models import get_model_class  
from utils import set_seed

set_seed(42)

class Trainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.models = parameters['models']
        self.model_parameters = parameters['model_parameters']

    def train(self, train_sequences, test_sequences, return_result = False):
        for model_name in self.models:
            print(f"Training model: {model_name}")
            model_config = self.model_parameters[model_name]
            batch_size = model_config['batch_size']
            n_epochs = model_config['n_epochs']

            data_module = IcuDataModule(
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                batch_size=batch_size
            )

            ModelClass = get_model_class(model_name)
            model = ModelClass(
                n_features=self.parameters['n_features'],
                **model_config
            )

            # checkpointing, look at that later 
            # checkpoint_callback = ModelCheckpoint(
            #     dirpath=f'checkpoints/{model_name}',
            #     filename='best-checkpoint',
            #     save_top_k=1,
            #     verbose=True,
            #     monitor='val_loss',
            #     mode='min',
            # )
            logger = TensorBoardLogger("lightning_logs", name=model_name)

            data_module = IcuDataModule(train_sequences, test_sequences, batch_size=model_config['batch_size'])

            pl_trainer = pl.Trainer(
                logger=logger,
                enable_checkpointing=False,
                #callbacks=[checkpoint_callback],
                max_epochs=n_epochs,
                accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
                devices=1,
                gradient_clip_val=model_config['gradient_clip_val']
            )
            print("Using device:", next(model.parameters()).device)

            print("Training sequences sample:")
            print("Train Sequence Shape:", train_sequences[0][0].shape)
            print("Test Sequence Shape:", test_sequences[0][0].shape)

            pl_trainer.fit(model, data_module)
            result = pl_trainer.test(model, data_module)
            print(result)


            return result

    def train_fractional(self, sequences):
        fractional_data = sequences['fractional_mimic_tudd']
        fractional_results = {}
        for fraction, sequence in fractional_data.items():
            print(f"Training fraction {fraction}")
            result = self.train(sequence, sequences['tudd']['test'], return_result=True)
            fractional_results[fraction] = result[0]
        return fractional_results
        


        