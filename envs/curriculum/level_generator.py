from enum import StrEnum, auto


class LevelGenerator():
    """generates levels for agent training. Levels created as height
    map for the mujoco environment. Options for types of levels are:
    - step
    - stump
    - gap
    - stairs
    - gradient surface
    maybe addable:
    - slippery surface
    """
    
    def __init__(self) -> None:
        pass
    


class LevelType(StrEnum):
    PLANE = auto()
    SLAB = auto()
    STAIRS = auto()
    LOG = auto()
    STUMP = auto()
    RAMP = auto()