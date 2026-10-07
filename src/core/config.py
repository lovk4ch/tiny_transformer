from dataclasses import dataclass

from core.types.sampling import SamplingMethod


@dataclass
class ModelConfig:
    max_tokens: int = 64
    ff_dim_size: int = 32
    embedding_size: int = 32

@dataclass
class TrainConfig:
    train_percent: float = 90.0
    epochs: int = 30
    learning_rate: float = 5e-3
    dataset: str = "data/texts.txt"

@dataclass
class CheckpointConfig:
    path: str = "models/tiny_transformer.pt"

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 6
    top_p: float = 0.9
    sampling: SamplingMethod = SamplingMethod.TOP_K
