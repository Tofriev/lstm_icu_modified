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
        self.cm = parameters["confusion_matrix"]
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
            )

            ModelClass = get_model_class(model_name)
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

            accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using accelerator: {accelerator}")

            pl_trainer = pl.Trainer(
                logger=logger,
                enable_checkpointing=False,
                # callbacks=[checkpoint_callback],
                max_epochs=n_epochs,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=model_config["gradient_clip_val"],
            )

            print("Training sequences sample:")
            print("Train Sequence Shape:", train_sequences[0][0].shape)
            print("Test Sequence Shape:", test_sequences[0][0].shape)

            pl_trainer.fit(model, data_module)
            result = pl_trainer.test(model, data_module)

            self.trained_models[model_name] = model
            self.results[model_name] = result
            # confusion matrix
            if self.cm:
                true_labels = []
                predictions = []
                print("starting confusion matrix")
                for batch in data_module.test_dataloader():
                    x = batch["sequence"]
                    y = batch["label"]
                    print("y_shape:", y.shape)
                    y_pred = model(x).argmax(dim=1).cpu().numpy()
                    true_labels.extend(y.cpu().numpy())
                    predictions.extend(y_pred)
                print("finished confusion matrix")
                cm = confusion_matrix(true_labels, predictions)
                print("Confusion Matrix:")
                print(cm)
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Class 0", "Class 1"],
                    yticklabels=["Class 0", "Class 1"],
                )
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.show()

                print("Classification Report:")
                print(classification_report(true_labels, predictions))
        print(self.results)
        return self.results, self.trained_models

    def train_with_datamodule(self, data_module):
        """
        Exactly like train(), but uses data_module for fit/test.
        """
        logger = TensorBoardLogger("lightning_logs", name=self.parameters["models"][0])
        accelerator = "mps" if torch.backends.mps.is_available() else "cpu"

        pl_trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=False,
            max_epochs=self.parameters["model_parameters"][
                self.parameters["models"][0]
            ]["n_epochs"],
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=self.parameters["model_parameters"][
                self.parameters["models"][0]
            ]["gradient_clip_val"],
        )

        ModelClass = get_model_class(self.parameters["models"][0])
        model = ModelClass(
            n_features=self.parameters["n_features"],
            **self.parameters["model_parameters"][self.parameters["models"][0]],
        )

        pl_trainer.fit(model, data_module)
        result = pl_trainer.test(model, data_module)
        self.trained_models[self.parameters["models"][0]] = model
        self.results[self.parameters["models"][0]] = result
        return self.results, model
