# configs.py
from dataclasses import dataclass

@dataclass
class LSTMModelConfig:
    n_features: int
    n_classes: int
    n_hidden: int = 100
    n_layers: int = 2
    dropout: float = 0.75
    bidirectional: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    class_weights: list = None  # e.g., [1.0, 3.0] for imbalanced classes
    batch_size: int = 32
    n_epochs: int = 3
