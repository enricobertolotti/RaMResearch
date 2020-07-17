import numpy as np
import cv2
import copy
############################ Static parameters
conversion_factor = 4.6  # in pixels per mm


#################################################### Color Conversion Functions

# Returns a rgb version of a 3D BW Array
def BWtoRGB(array, pureRed: bool = False):
    # Check if the array is already a RGB array
    if len(array.shape) >= 4:
        return array
    # Otherwise convert it to RGB
    else:
        if pureRed:
            blackarray = np.zeros(array.shape)
            return np.stack((blackarray, blackarray, array), axis=len(array.shape))
        else:
            return np.stack((array, array, array), axis=len(array.shape))


#################################################### Number Functions
def num_clip(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x

#################################################### Length Conversion Functions

def pix_to_mm(pixel_val, return_int=False):
    val = pixel_val / conversion_factor
    return int(val) if return_int else val


def mm_to_pix(length_val, return_int=False):
    val = length_val * conversion_factor
    return int(val) if return_int else val


#################################################### Simple Array Functions

# Inverts an array with a given bit_depth
def invert_array(array, int_bitdepth=8):
    max_val = (2**int_bitdepth)-1
    return np.subtract(max_val, array).astype(np.uint8)


# Clips an array at 0 and scales it to the max value given
def clip_scale(array, max_value=255):
    r_array = np.clip(array.copy(), a_min=0, a_max=None)
    return (np.divide(r_array, np.max(r_array)) * max_value).astype(np.uint8)


# Returns a thresholded image of the type specified
def threshhold(array, threshold, astype=np.int):
    if astype == np.bool_:
        return array > threshold


#################################################### Complex Array Functions

# Returns an array with each dimension padded to at least the specified amount
# Works on n-dimensional arrays
def pad_to_minimum(array, min_value, pad_value: int = 0):
    if not isinstance(min_value, (tuple, np.ndarray)):
        min_val = min_value
        min_value = np.empty(len(array.shape)).astype(np.uint16)
        min_value.fill(min_val)
    array_dim = array.shape
    for i in range(len(min_value)):
        if array_dim[i] < min_value[i]:
            pad_amount = min_value[i] - array_dim[i]
            padding = np.zeros((len(min_value), 2)).astype(np.int)
            padding[i] = [int(pad_amount / 2), pad_amount-int(pad_amount / 2)]
            array = np.pad(array, padding, mode="constant", constant_values=pad_value)
    return array


# Returns an array with the center cropped out to the length given
# Note: Only works on 3-Dimensionsal arrays
def crop_all_axis_to_length(array, length: int):
    crop_amount = np.subtract(array.shape, length).clip(min=0)
    crop_amount = [[int(c/2), c - int(c/2)] for c in crop_amount]
    d = np.empty((len(array.shape), 2)).astype(np.int)
    for i in range(len(array.shape)):
        d[i] = [crop_amount[i][0], array.shape[i] - crop_amount[i][1]]
    return array[d[0][0]:d[0][1], d[1][0]:d[1][1], d[2][0]:d[2][1]]


#################################################### Debugging Functions
def draw_point(array, position, size=10, color=0):

    array_copy = copy.deepcopy(array)
    pos = tuple(position[1:])
    size = int(size / 2)

    # Rgb settings
    if isinstance(color, (tuple, np.ndarray)):
        array_copy = BWtoRGB(array_copy) if len(array_copy.shape) < 4 else array_copy
        for i in range(len(color)):
            if color[i] > 0:
                array_copy[position[0], pos[0] - size:pos[0] + size, pos[1] - size:pos[1] + size, i] = color[i]
    else:
        array_copy[position[0], pos[0] - size:pos[0] + size, pos[1] - size:pos[1] + size] = color

    return array_copy


#################################################### Debugging Functions
def print_min_max(array, name=""):
    if name == "":
        print("Max:\t" + str(np.max(array)) + "\n")
        print("Min:\t" + str(np.min(array)) + "\n")
    else:
        print(name + " Max:\t" + str(np.max(array)))
        print(name + " Min:\t" + str(np.min(array)))


#################################################### Terminal Functions
def print_divider(text, spacers=1):
    divider = "================================================================"
    print("\n")
    for i in range(spacers):
        print(divider)
    print(text)
    for i in range(spacers):
        print(divider)
