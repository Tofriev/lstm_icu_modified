from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
import numpy as np
import torch


###########################################
## Data modulle for LSTM
###########################################

class IcuDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.tensor(sequence, dtype=torch.float32),
            label=torch.tensor(label).long(),
        )

class IcuDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = IcuDataset(self.train_sequences)
        self.test_dataset = IcuDataset(self.test_sequences)
        print("Number of training sequences:", len(self.train_dataset))
        print("Number of test sequences:", len(self.test_dataset))
        for i in range(3):  # Check first 3 training sequences
            sample = self.train_dataset[i]
            print(f"Train sample {i}: Sequence shape={sample['sequence'].shape}, Label={sample['label']}")      


    def train_dataloader(self):
        labels = [seq[1] for seq in self.train_sequences]
        sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / sample_count
        samples_weight = np.array([weight[int(t)] for t in labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          sampler=sampler,
                          num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0) 
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=0)
