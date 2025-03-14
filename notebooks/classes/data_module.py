from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import torch

###########################################
## Data module for LSTM with (optional) static features
###########################################

class IcuDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, n_static_features=None):
        """
        sequences: list of tuples either in the format:
            (sequence, static, label)  OR  (sequence, label)
        n_static_features: if sequences are only 2-tuples, a default static vector of zeros
            with this many features will be created.
        """
        self.sequences = sequences
        self.n_static_features = n_static_features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        if len(sample) == 3:
            sequence, static, label = sample
        elif len(sample) == 2:
            sequence, label = sample
            # Create a default static vector of zeros if number provided
            if self.n_static_features is not None:
                static = np.zeros(self.n_static_features, dtype=np.float32)
            else:
                static = np.array([], dtype=np.float32)
        else:
            raise ValueError("Expected sample to have 2 or 3 elements.")
        assert isinstance(sequence, np.ndarray), "Sequence must be a NumPy array."
        assert sequence.ndim == 2, f"Expected 2D sequence, got {sequence.ndim}D."
        return (sequence, static, label)

class IcuDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size, n_static_features=None):
        """
        train_sequences and test_sequences: list of tuples (sequence, static, label) OR (sequence, label)
        n_static_features: required if your sequences are only (sequence, label); used to create default static vectors.
        """
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.n_static_features = n_static_features

    def setup(self, stage=None):
        self.train_dataset = IcuDataset(self.train_sequences, n_static_features=self.n_static_features)
        self.test_dataset = IcuDataset(self.test_sequences, n_static_features=self.n_static_features)
        print("Number of training sequences:", len(self.train_dataset))
        print("Number of test sequences:", len(self.test_dataset))
        # Print out first few samples for debugging
        for i in range(3):
            sample = self.train_dataset[i]
            seq, static, label = sample
            print(f"Train sample {i}: Sequence shape={seq.shape}, Static shape={np.array(static).shape}, Label={label}")

    @staticmethod
    def collate_fn_static(batch):
        """
        Expects each sample as a tuple (sequence, static, label) and returns a dictionary with:
            "sequence": Tensor of shape (batch, time_steps, n_seq_features)
            "static": Tensor of shape (batch, n_static_features)
            "label": Tensor of shape (batch,)
        """
        sequences, statics, labels = zip(*batch)
        sequences = torch.tensor(np.array(sequences), dtype=torch.float)
        statics = torch.tensor(np.array(statics), dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        return {"sequence": sequences, "static": statics, "label": labels}

    def train_dataloader(self):
        # Extract labels from index 2 of each sample
        labels = [sample[2] for sample in self.train_sequences]
        sample_count = np.array(
            [len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)]
        )
        weight = 1.0 / sample_count
        samples_weight = np.array([weight[int(t)] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=IcuDataModule.collate_fn_static,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=IcuDataModule.collate_fn_static,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=IcuDataModule.collate_fn_static,
            num_workers=0,
        )
