# from RaMCode.Analysis.Rotation import RotationAnalysis as ra
from RaMCode.Data.BasicDataStructs import Coordinate as coord
from RaMCode.Data.RingV2 import RingImage, import_ring_image
from RaMCode.Utils import General as general

from PIL import Image
from pathlib import Path as path
import numpy as np

import time


default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/"


def ml_export_image(image, image_id, num_slices, ring_pos):
    # Folder & Path preparations:
    subfolder = "ML_Training_Data/" + str(image_id) + "/"
    full_path = default_folder + subfolder
    path_obj = path(full_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Get the export-image

    # Create a list of depth indexes for the slices to be exported
    z_mid = int(ring_pos[0] / 2)
    slices = list(range(z_mid, z_mid + num_slices) if num_slices > 0 else list(range(image.shape[0])))

    for slice_depth in slices:
        # Define image name & path
        img_name = str(image_id) + "_clean_ring_slice_ml_training_" + str(slice_depth)
        file_path = full_path + img_name + ".png"

        # Create and save image
        test_slice = image[slice_depth, :]
        img = Image.fromarray(test_slice)
        img.save(file_path)

        print("\rSlice Exported:\t" + str(slice_depth), end="")

    print("\n")


def get_ring_mask_image(dim: tuple, position, rotation, r_small, r_large):

    # Create new header for terminal output
    general.print_divider("Slice Export for Machine Learning")

    # Get dimensions and positions
    ring_pos = position.get_coordinates_int() if isinstance(position, coord) else position
    max_crop = general.get_max_crop(dim, ring_pos)

    # Get the background image & the ring image with the right angle
    base_image = np.zeros(dim)
    ring_image_obj = import_ring_image(r_small=r_small, r_large=r_large, ring_angle=rotation, filled=True)
    ring_image = ring_image_obj.get_cropped_image(crop_dim=max_crop)

    # Define Starting Corner
    ri_s = ring_image.shape
    s_c = np.subtract(ring_pos, np.divide(max_crop, 2).astype(np.int))
    base_image[s_c[0]:s_c[0]+ri_s[0], s_c[1]:s_c[1]+ri_s[1], s_c[2]:s_c[2]+ri_s[2]] += ring_image
    base_image = base_image.astype(np.uint8)

    # Print to Terminal
    print("Image for export created...")

    return base_image
