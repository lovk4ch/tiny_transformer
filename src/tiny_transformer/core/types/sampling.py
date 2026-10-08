from enum import Enum, auto


class SamplingMethod(Enum):
    GREEDY = auto()
    TOP_K = auto()
    TOP_P = auto()