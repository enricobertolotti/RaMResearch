from RaMResearch.Filters.Filtersv2 import shiftarray
import numpy as np
from scipy.signal import correlate
import RaMResearch.Utils.General as general


def crs_convolve(array1, array2, fullRange=False, shift=False, clip=False):

    def corr_mag(array_1, array_2):
        center_1 = [int(x / 2) for x in array_1.shape]
        # center_2 = [int(x / 2) for x in array_2.shape]
        mag1 = correlate(array_1, array_1, mode='full')[center_1[0], center_1[1], center_1[2]]
        # mag2 = correlate(array_2, array_2, mode='full')[center_2[0], center_2[1], center_2[2]]

        print("Min: " + str(np.min(mag1)) + "\t Max: " + str(np.max(mag1)))

        return mag1

    # Shift the array if required
    if shift:
        shift = int(np.max(array1) - np.min(array1)/2)
        array1, array2 = shiftarray(array1, shift), shiftarray(array2, shift)

    # Set the correlation mode
    mode = "full" if fullRange else "valid"

    # mag = corr_mag(array1, array2)

    convolved_array_uint8 = correlate(array1.astype(np.int8), array2.astype(np.int8), mode=mode)

    test = convolved_array_uint8
    print("Sum before: ", np.sum(test))
    test = np.absolute(test)
    print("Sum after: ", np.sum(test))
    print("Min: ", np.min(test), ",  Max: ", np.max(test))
    general.print_min_max(convolved_array_uint8, name="Uint8 Array")

    # Clip and Return array
    return convolved_array_uint8 if not clip else np.clip(convolved_array_uint8, 0, 255).astype(np.uint8)
