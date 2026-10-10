from dataclasses import dataclass
from pathlib import Path

from tiny_transformer.core.types.sampling import SamplingMethod


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 6
    top_p: float = 0.9
    sampling: SamplingMethod = SamplingMethod.TOP_K

@dataclass
class ModelConfig:
    max_tokens: int = 128
    ff_dim_size: int = 128
    embedding_size: int = 64

@dataclass
class CheckpointConfig:
    path: Path = MODELS_DIR / "tiny_transformer.pt"

@dataclass
class TrainConfig:
    train_percent: float = 90
    epochs: int = 100
    learning_rate: float = 5e-3
    dataset_path: Path = DATA_DIR / "tinypairs-10k.json"

