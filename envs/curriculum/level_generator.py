from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional
import numpy as np
import random


class ElementType(Enum):
    SLAB = [0.5, 1, 0.5]  # green
    STAIRS = [1, 0.5, 0.5]  # red
    STUMP = [0.5, 0.5, 1]  # blue
    GAP = [0.5, 1, 1]  # turquoise


@dataclass
class Element():
    type: ElementType
    pos: int  # number of geom (0..n_geoms) at which element is positioned
    height: float  # for slab, stump and gap height difference to surrounding, for stairs step height
    n: int  # number of elements involved in structure (without margin), slab stump and gap 1, stairs n_steps
    gap_width: Optional[int] = None  # number of geoms used for gap width


@dataclass
class Level():
    elements: List[Element] = field(init=False)

    def __post_init__(self):
        self.elements = []


@dataclass
class LevelDescription():
    n_elements: int
    elements: List[float] = field(init=False)
    types: List[Optional[ElementType]] = field(init=False)

    def __post_init__(self):
        self.elements = np.zeros(self.n_elements)
        self.types = np.array([None] * self.n_elements)


class LevelGenerator:
    """generates levels for agent training. Levels created as height
    map for the mujoco environment. Options for types of levels are:
    - step
    - stump
    - gap
    - stairs
    maybe addable:
    - slippery surface
    """
    
    def __init__(self, level_length_in_m: int=50, level_range_z: Tuple[int] = [-10,10], 
                 n_geoms: int=150, element_size: int = 3, n_margin_elems: int=1) -> None:
        """Name-clarification: a single geom is a geom. when geoms are grouped together and moved as one, they become an element.
        Several elements can form an obstacle, e.g. for a slab: one lower element and another raised element become together an obstacle."""
        self.level_length_in_m = level_length_in_m
        self.level_range_in_z = level_range_z
        self.n_geoms = n_geoms
        self.geom_length = self.level_length_in_m / self.n_geoms

        self.element_size = element_size  # how many geoms are grouped together
        self.n_margin_elems = n_margin_elems
        self.margin_size = self.element_size * self.n_margin_elems
        self.rng = np.random.default_rng()  # use with seed

        space_per_element = self.geom_length * self.element_size
        space_per_obst = space_per_element + self.n_margin_elems * space_per_element
        self.max_number_of_obstacles = int(self.level_length_in_m / space_per_obst)

        # params for level difficulty
        self.max_slab_height = 1.0
        self.max_step_height = 1.0
        self.max_step_n = 5
        self.max_stump_height = 1.0
        self.max_gap_depth = 1.0
        self.max_gap_size = 1.0

    def flip(self, elem_total_height: float, last_abs_height: float) -> int:
        """Flips element z_direction by 50/50. Checks if z_range of level allows it and flips again if not."""
        choice = self.rng.choice([-1,1])
        new_height = elem_total_height * choice + last_abs_height
        if self.level_range_in_z[0] <= new_height <= self.level_range_in_z[1]:
            return choice
        return choice * -1  # NOTE: Assertion here is that individual element cannot be higher than half z_range
    
    def create_level_elements(self, obstacles: float, diff_slab: float, diff_stairs: float, diff_stump: float, diff_gap: float, seed=None) -> Level:
        """Create level with percentage of obstacles and percentual difficulty per obstacle."""
        self.rng = np.random.default_rng(seed)
        n_obst = int(np.ceil(obstacles*self.max_number_of_obstacles))  # 0.0 -> 0, 1 -> max
        level: Level = Level()
        available_positions = list(range(0,self.n_geoms))
        last_absolute_height = 0
        for _ in range(0, n_obst):
            element_t = self.pick_random_element()
            elem = self.get_element_params(element_t, diff_slab, diff_stairs, diff_stump, diff_gap, last_absolute_height)
            elem.pos, available_positions = self.pick_random_location(available_positions, elem.n)
            if elem.pos is not None:
                level.elements.append(elem)
                last_absolute_height += elem.height * elem.n
        level.elements.sort(key=lambda x: x.pos[0])
        return level

    def pick_random_element(self) -> ElementType:
        elements = list(ElementType)
        return ElementType(self.rng.choice(elements))
    
    def get_element_params(self, element_t: ElementType, diff_slab: float, diff_stairs: float, 
                           diff_stump: float, diff_gap: float, last_abs_height: float) -> Element:
        """Returns the height of element with number of needed elements."""
        upper_borders = np.array([diff_slab, diff_stairs, diff_stump, diff_gap])
        lower_borders = upper_borders # / 2
        match element_t:
            case ElementType.SLAB:
                n = 1
                height = self.rng.uniform(lower_borders[0], upper_borders[0])
                height = height * self.flip(height*n, last_abs_height)
            case ElementType.STAIRS:
                up_n = max(np.ceil(upper_borders[1] * self.max_step_n), 1)
                low_n = max((up_n // 2), 1)
                # n_stairs = self.rng.integers(low_n, up_n + 1)  # to include upper border
                n_stairs = np.ceil(upper_borders[1] * up_n)
                height = self.rng.uniform(lower_borders[1], upper_borders[1])
                height = height * self.flip(height*n_stairs, last_abs_height)
                n = n_stairs
            case ElementType.STUMP:
                height = self.rng.uniform(lower_borders[2], upper_borders[2]) 
                n = 1
            case ElementType.GAP:
                height = self.rng.uniform(lower_borders[3], upper_borders[3])
                up_n = min(np.floor(upper_borders[3] * self.element_size + 1), 3)
                low_n = np.ceil(upper_borders[3] * (self.element_size - 1))
                # gap_width = self.rng.integers(low_n, up_n + 1)  # upper boarder not included
                gap_width = np.ceil(upper_borders[3] * up_n)
                n = 1
                return Element(element_t, None, height, n, gap_width)
        return Element(element_t, None, height, n)
    
    def pick_random_location(self, available, n_elements):
        """available is array of elements that are available. 
        n_of_elements is number of elements in obstacle that should fit. In search it is increased by 1 for margin."""
        n = n_elements * self.element_size + self.margin_size  # from single number of elements to number of geoms with margin
        splits = np.where(np.diff(available) != 1)[0] + 1
        runs = np.split(available, splits)
        valid_runs = [r for r in runs if len(r) >= n]
        if not valid_runs:
            return None, available
        run = self.rng.choice(np.array(valid_runs, dtype=object))
        start_id = self.rng.integers(0, len(run) - n + 1)
        start_id -= start_id % (self.element_size + self.margin_size)  # cutting of start to start of element
        # start_id = random.randint(0, len(run) - n)
        selection = np.array(run[start_id : start_id + n], dtype=int)
        remaining = np.setdiff1d(available, selection)
        return selection, remaining
    
    def calculate_element_coords(self, level: Level) -> LevelDescription:
        """Converts level from type Level (list of elements) to actual positions of each geom.
        Used in env to set geom z_coords accordingly."""
        res: LevelDescription = LevelDescription(self.n_geoms)
        if len(level.elements) == 0:
            return res  # for no elements, return default level description
        last_height = 0
        if level.elements[0].pos[0] > 0:
            res.elements[0:level.elements[0].pos[0]] = 0
        for i, e in enumerate(level.elements):
            match e.type:
                case ElementType.SLAB:
                    last_height = self._add_slab_position(e, last_height, res)
                case ElementType.STAIRS:
                    last_height = self._add_stairs_position(e, last_height, res)
                case ElementType.STUMP:
                    last_height = self._add_stump_position(e, last_height, res)
                case ElementType.GAP:
                    last_height = self._add_gap_position(e, last_height, res)
            if i+1 < len(level.elements):
                next_elem_pos = level.elements[i+1].pos[0]
            else:
                next_elem_pos = self.n_geoms
            res.elements[e.pos[-1] + 1 : next_elem_pos] = last_height
            res.types[e.pos] = e.type
        return res

    def _add_slab_position(self, element: Element, last_height: float, res: LevelDescription):
        slab_height = last_height + element.height
        res.elements[element.pos] = slab_height
        return slab_height
    
    def _add_stairs_position(self, element: Element, last_height: float, res: LevelDescription):
        step_height = 0
        for i in range(element.n):
            step_height = (i + 1) * element.height + last_height
            pos = element.pos[i * self.element_size:(i+1) * self.element_size]
            res.elements[pos] = step_height
        res.elements[element.pos[-self.element_size:]] = step_height  # margin at end of stairs is last three positions of obstacle
        return step_height
    
    def _add_stump_position(self, element: Element, last_height: float, res: LevelDescription):
        res.elements[element.pos[0]] = element.height + last_height
        res.elements[element.pos[1:]] = last_height
        return last_height
    
    def _add_gap_position(self, element: Element, last_height: float, res: LevelDescription):
        gap_height = last_height - element.height
        res.elements[element.pos[:element.gap_width]] = gap_height
        res.elements[element.pos[element.gap_width:]] = last_height
        return last_height
        