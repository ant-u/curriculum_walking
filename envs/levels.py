from typing import Tuple
import numpy as np

def get_step_level(model, height: np.float32,
        x_ratio: np.float32 = 0, y_ratio: np.float32 = 0) -> np.ndarray:
    """Get hfield of a level with one big step in the map.
    
    Args:
        x_ratio: ratio of x-axis which is raised, beginning at x = 0
        y_ratio: ratio of y-axis which is raised, beginning at y = 0

    Every point which is not included in ratio is set to height 0.
    A negative ratio means the raised part beginns at x = max or y = max. 
    Ratio of 0 is a flat surface, height = 0
    Ratio of 1 is a flat surface, height = 1
    """
    hfield, nrow, ncol = get_h_field(model)
    
    x_border = np.int32(np.round(x_ratio * ncol))  # abs border for x value
    x_start = min(0, x_border)
    x_end = x_border if x_start == 0 else ncol
    hfield[:, x_start:x_end] = height

    y_border = np.int32(np.round(y_ratio * nrow))  # abs border for y value
    y_start = min(0, y_border)
    y_end = y_border if y_start == 0 else nrow
    hfield[y_start:y_end, :] = height

    return hfield.ravel()
    


def get_h_field(model) -> Tuple[np.ndarray, np.int32, np.int32]:
    nrow = model.hfield_nrow[0]
    ncol = model.hfield_ncol[0]
    hfield_data = np.zeros(nrow * ncol, dtype=np.float32)

    hfield = hfield_data.reshape(nrow, ncol)
    return hfield, nrow, ncol



