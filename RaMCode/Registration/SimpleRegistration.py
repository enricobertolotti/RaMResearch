import numpy as np
from scipy.ndimage import interpolation


def shift_array(array, direction):
    pass
    # return np.array(shift(array, direction, order=1)).astype(np.uint8)


def simple_subtraction(array_w_ring, array_no_ring):
    array_w_ring = 255 - array_w_ring.astype(np.int16)
    array_no_ring = array_no_ring.astype(np.int16)
    return np.clip(np.subtract(array_w_ring, array_no_ring), 0, 255).astype(np.uint8)
