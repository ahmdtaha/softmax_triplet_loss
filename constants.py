



nominal_width = 400
nominal_height = 400

max_frame_size = 256

frame_height = 224
frame_width = 224

frame_channels = 3
context_channels = 5


from enum import Enum

class Train_Mode(Enum):
    VANILLA = 0
    SEMI_HARD = 1
    HARD = 2
    AWTL_HARD = 3
    CNTR = 4

class Subset(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Temporal_Direction(Enum):
    BEFORE = 0
    AFTER = 1







#baseline lr 0.1 B 64


