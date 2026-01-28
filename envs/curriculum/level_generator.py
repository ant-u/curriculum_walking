from dataclasses import dataclass
from enum import StrEnum, auto
from typing import List, Tuple
import numpy as np
import random


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

        space_per_obst = self.size_per_obst + self.margin_per_obst
        self.max_number_of_obstacles = self.level_size[0] / space_per_obst  # NOTE: 10 for 5+5 plattform at start

    def create_level(self, obstacles: float, diff_per_obs: float):
        """"""
        n_obst = int(np.ceil(obstacles*self.max_number_of_obstacles))  # 0.0 -> 0, 1 -> max
        elements = []
        available_positions = list(range(0,50,1))
        for _ in range(0, n_obst):
            element = self.pick_random_element()
            height, depth, number = self.get_element_params(element)
            pos, available_positions = self.pick_random_location(available_positions, number)
            elements.append({'element': element, "pos": pos[0], "height": height, "depth": depth, "n": number})
        elements.sort(key=lambda x: x["pos"])
        result = self.calculate_element_coords(elements)
        return result

    def pick_random_element(self):
        elements = list(LevelType)
        return LevelType(np.random.choice(elements))
    
    def pick_random_location(self, available, n):
        """available is array of elements that are available. 
        n_of_elements is number of elements in obstacle that should fit. In search it is increased by 1 for margin."""
        splits = np.where(np.diff(available) != 1)[0] + 1
        runs = np.split(available, splits)
        valid_runs = [r for r in runs if len(r) >= n]
        if not valid_runs:
            return None, available
        run = random.choice(valid_runs)
        start_id = random.randint(0, len(run) - n)
        selection = run[start_id : start_id + n]
        remaining = np.setdiff1d(available, selection)
        return selection, remaining

    def get_element_params(self, element):
        match element:
            # height, depth, number
            case LevelType.SLAB:  # backwards
                params = 0.1, 2, 2
            # case LevelType.STAIRS:  # backwards
            #     params = 0.1, 1, 3
            case LevelType.STUMP:
                params = np.float64(0.1), np.float64(0.2), 2
            # case LevelType.RAMP:
            #     pass
            case LevelType.GAP:
                params = 1, 0.1, 2
        return params
    
    def calculate_element_coords(self, elements):
        res: List[Element] = []
        last_height = 0
        if elements[0]["pos"] > 0:
            x_size = elements[0]["pos"] / 2
            res.append(Element([x_size, 0, -2.5], [x_size, self.level_size[1]/2, 2.5]))
        for i, e in enumerate(elements):
            if i+1 < len(elements):
                next_elem_pos = elements[i+1]["pos"]
            else:
                next_elem_pos = self.level_size[0]
            match e["element"]:
                case LevelType.SLAB:
                    elem, last_height = self._get_slab_position(e, last_height, next_elem_pos)
                    res.append(elem)
                # case LevelType.STAIRS:
                #     pass
                case LevelType.STUMP:
                    elems, last_height = self._get_stump_position(e, last_height, next_elem_pos)
                    res.extend(elems)
                case LevelType.GAP:
                    pass
        return res

    def _get_slab_position(self, element, last_height, end):
        x_size = (end - element["pos"]) / 2
        z_size = (0 - self.level_begin[2] + last_height + element["height"]) / 2
        x_pos = element["pos"] + x_size
        z_pos = self.level_begin[2] + z_size
        return Element([x_pos, 0, z_pos], [x_size, self.level_size[1]/2, z_size]), z_pos + z_size
    
    def _get_stump_position(self, element, last_height, end):
        x_size = element["depth"] / 2
        z_size = (0 - self.level_begin[2] + last_height + element["height"]) / 2
        x_pos = element["pos"] + x_size
        z_pos = self.level_begin[2] + z_size
        stump = Element([x_pos, 0, z_pos], [x_size, self.level_size[1]/2, z_size])
        m_x_size = (end - x_pos - x_size) / 2
        m_z_size = (0 - self.level_begin[2] + last_height) / 2
        m_x_pos = m_x_size + x_pos + x_size 
        m_z_pos = self.level_begin[2] + m_z_size
        margin_elem = Element([m_x_pos, 0, m_z_pos], [m_x_size, self.level_size[1]/2, m_z_size])
        return [stump, margin_elem], last_height
        

class LevelType(StrEnum):
    SLAB = auto()
    # STAIRS = auto()
    # STUMP = auto()
    # RAMP = auto()
    # GAP = auto()


@dataclass
class Element():
    pos: List[float]
    size: List[float]