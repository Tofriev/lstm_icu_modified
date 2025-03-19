import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from classes.data_module import IcuDataModule
from models import get_model_class
from utils import set_seed
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

set_seed(42)


class Trainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.models = parameters["models"]
        self.model_parameters = parameters["model_parameters"]
        self.results = {}
        self.trained_models = {}

    def train(self, train_sequences, test_sequences, return_result=False):
        for model_name in self.models:
            print(f"Training model: {model_name}")
            model_config = self.model_parameters[model_name]
            batch_size = model_config["batch_size"]
            n_epochs = model_config["n_epochs"]

            data_module = IcuDataModule(
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                batch_size=batch_size,
                model_name = self.parameters['models'] 
            )

            ModelClass = get_model_class(model_name)
            if 'lstm_static'in self.parameters['models']:
                model = ModelClass(n_seq_features=self.parameters["n_seq_features"],
                                   n_static_features=self.parameters["n_static_features"],
                                     **model_config)
            else:
                model = ModelClass(n_features=self.parameters["n_features"], **model_config)

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

            if torch.backends.mps.is_available():
                accelerator = "mps"
            elif torch.cuda.is_available():
                accelerator = "cuda"
            else:
                accelerator = "cpu"

            pl_trainer = pl.Trainer(
                logger=logger,
                enable_checkpointing=False,
                # callbacks=[checkpoint_callback],
                max_epochs=n_epochs,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=model_config["gradient_clip_val"],
            )
            print("Using device:", next(model.parameters()).device)

            print("Training sequences sample:")
            print("Train Sequence Shape:", train_sequences[0][0].shape)
            print("Test Sequence Shape:", test_sequences[0][0].shape)

            pl_trainer.fit(model, data_module)
            result = pl_trainer.test(model, data_module)

            self.trained_models[model_name] = model
            self.results[model_name] = result
            # # confusion matrix
            # true_labels = []
            # predictions = []
            # print("starting confusion matrix")
            # for batch in data_module.test_dataloader():
            #     x = batch["sequence"]
            #     y = batch["label"]
            #     print("y_shape:", y.shape)
            #     y_pred = model(x).argmax(dim=1).cpu().numpy()
            #     true_labels.extend(y.cpu().numpy())
            #     predictions.extend(y_pred)
            # print("finished confusion matrix")
            # cm = confusion_matrix(true_labels, predictions)
            # print("Confusion Matrix:")
            # print(cm)
            # plt.figure(figsize=(8, 6))
            # sns.heatmap(
            #     cm,
            #     annot=True,
            #     fmt="d",
            #     cmap="Blues",
            #     xticklabels=["Class 0", "Class 1"],
            #     yticklabels=["Class 0", "Class 1"],
            # )
            # plt.title("Confusion Matrix")
            # plt.xlabel("Predicted")
            # plt.ylabel("Actual")
            # plt.show()

            # print("Classification Report:")
            # print(classification_report(true_labels, predictions))
        print(self.results)
        return self.results, self.trained_models

    def train_fractional(self, sequences):
        fractional_data = sequences["fractional_mimic_tudd"]
        fractional_results = {}
        for fraction, sequence in fractional_data.items():
            print(f"Training fraction {fraction}")
            result = self.train(sequence, sequences["tudd"]["test"], return_result=True)
            fractional_results[fraction] = result[0]
        return fractional_results
