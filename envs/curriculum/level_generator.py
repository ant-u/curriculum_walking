from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import List, Tuple
import numpy as np
import random


class LevelType(StrEnum):
    SLAB = auto()
    STAIRS = auto()
    STUMP = auto()
    GAP = auto()


@dataclass
class Element():
    type: LevelType
    pos: int  # number of geom (0..49) at which element is positioned
    height: float  # for slab, stump and gap height difference to surrounding, for stairs step height
    n: int  # number of geoms involved in structure, slab stump and gap 2, stairs n_steps + 1 (margin)

@dataclass
class LevelDescription():
    n_elements: int
    n_stumps: int
    elements: List[float] = field(init=False)
    stumps: List[float | None] = field(init=False)

    def __post_init__(self):
        self.elements = np.zeros(self.n_elements)
        self.stumps = [None] * self.n_stumps


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
    
    def __init__(self, level_size: Tuple[int], level_begin: Tuple[int], size_per_obst: float=1, margin_per_obst: float=1) -> None:
        """level_size is [x,y] of space used for level design
        level_begin is [x,y] of beginning of level space. Note that y has to be middle, but x first value of space
        size_per_obs is space allocated for each single obstacle. Not nescesarrily takes up all of it.
        margin_per_obs is space that comes after obstacle, adds up with whats not used of size_per_obs."""
        self.level_size= level_size
        self.level_begin= level_begin
        self.size_per_obst = size_per_obst
        self.margin_per_obst = margin_per_obst
        self.rng = np.random.default_rng()

        space_per_obst = self.size_per_obst + self.margin_per_obst
        self.number_of_elements = int(self.level_size[0] / self.size_per_obst)
        self.max_number_of_obstacles = int(self.level_size[0] / space_per_obst)  # NOTE: 10 for 5+5 plattform at start
        self.max_number_of_stumps = 10

    def create_level(self, obstacles: float, diff_per_obs: float):
        """"""
        n_obst = int(np.ceil(obstacles*self.max_number_of_obstacles))  # 0.0 -> 0, 1 -> max
        elements: List[Element] = []
        available_positions = list(range(0,50,1))
        for _ in range(0, n_obst):
            element = self.pick_random_element()
            height, number = self.get_element_params(element)
            if element == LevelType.STUMP:
                pos, available_positions = self.pick_random_location_stump(available_positions)
            else:
                pos, available_positions = self.pick_random_location(available_positions, number)
            elements.append(Element(element, pos[0], height, number))
        elements.sort(key=lambda x: x.pos)
        result = self.calculate_element_coords(elements)
        return result

    def pick_random_element(self):
        elements = list(LevelType)
        return LevelType(self.rng.choice(elements))
    
    def get_element_params(self, element):
        match element:
            # height, number
            case LevelType.SLAB:  # backwards
                params = 0.1, 2
            case LevelType.STAIRS:  # backwards
                params = 0.1, 3
            case LevelType.STUMP:
                params = np.float64(0.3), 2
            case LevelType.GAP:
                params = 1, 2
        return params
    
    def pick_random_location(self, available, n):
        """available is array of elements that are available. 
        n_of_elements is number of elements in obstacle that should fit. In search it is increased by 1 for margin."""
        splits = np.where(np.diff(available) != 1)[0] + 1
        runs = np.split(available, splits)
        valid_runs = [r for r in runs if len(r) >= n]
        if not valid_runs:
            return None, available
        run = self.rng.choice(np.array(valid_runs, dtype=object))
        start_id = self.rng.integers(0, len(run) - n + 1)
        # start_id = random.randint(0, len(run) - n)
        selection = run[start_id : start_id + n]
        remaining = np.setdiff1d(available, selection)
        return selection, remaining
    
    def pick_random_location_stump(self, available):
        candidats = np.asarray([x for x in available if x % 5 == 0])
        candidats = candidats[np.isin(candidats, available)]
        valid = candidats[np.isin(candidats+1, available)]

        if len(valid) == 0:
            return None, available
        choice = self.rng.choice(valid)
        choice = [choice, choice + 1]
        remaining = np.setdiff1d(available, choice)
        return choice, remaining
    
    def calculate_element_coords(self, elements: List[Element]):
        res: LevelDescription = LevelDescription(self.number_of_elements, self.max_number_of_stumps)
        last_height = 0
        if elements[0].pos > 0:
            res.elements[0:elements[0].pos] = 0
        for i, e in enumerate(elements):
            match e.type:
                case LevelType.SLAB:
                    last_height = self._add_slab_position(e, last_height, res)
                case LevelType.STAIRS:
                    last_height = self._add_stairs_position(e, last_height, res)
                case LevelType.STUMP:
                    last_height = self._add_stump_position(e, last_height, res)
                case LevelType.GAP:
                    last_height = self._add_gap_position(e, last_height, res)
            if i+1 < len(elements):
                next_elem_pos = elements[i+1].pos
            else:
                next_elem_pos = self.level_size[0]
            res.elements[e.pos + e.n : next_elem_pos] = last_height
        return res

    def _add_slab_position(self, element: Element, last_height: float, res: LevelDescription):
        slab_height = last_height + element.height
        res.elements[element.pos] = slab_height
        res.elements[element.pos + 1] = slab_height
        return slab_height
    
    def _add_stairs_position(self, element: Element, last_height: float, res: LevelDescription):
        last_element = element.pos + element.n - 1
        for i, pos in enumerate(range(element.pos, last_element)):
            step_height = (i + 1) * element.height + last_height
            res.elements[pos] = step_height
        res.elements[last_element] = step_height  # margin at end of stairs
        return step_height
    
    def _add_stump_position(self, element: Element, last_height: float, res: LevelDescription):
        res.stumps[element.pos // 5] = last_height + element.height
        res.elements[element.pos] = last_height
        res.elements[element.pos + 1] = last_height
        return last_height
    
    def _add_gap_position(self, element: Element, last_height: float, res: LevelDescription):
        gap_height = last_height - element.height
        res.elements[element.pos] = gap_height
        res.elements[element.pos + 1] = last_height
        return last_height
        