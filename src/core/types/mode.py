from enum import Enum, auto


class Mode(Enum):
    TRAIN = auto()
    EVALUATE = auto()
    GENERATE = auto()