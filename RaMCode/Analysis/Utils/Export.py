from RaMCode.Data.BasicDataStructs import Coordinate as coord
from RaMCode.Data.RingV2 import import_ring_image
from RaMCode.Utils import General as general

from PIL import Image
from pathlib import Path as path
import numpy as np
from RaMCode.Utils.Interfaces import imageview3d
default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/"




def ml_export_image(image, image_id, num_slices, ring_pos, image_type="mask", debug=False):
    # Folder & Path preparations:
    subfolder = "ML_Training_Data/Masks/" if "mask" in image_type else "ML_Training_Data/Images/"
    full_path = default_folder + subfolder
    path_obj = path(full_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Get the export-image

    # Create a list of depth indexes for the slices to be exported
    z_mid = int(ring_pos[0] / 2)
    slices = list(range(z_mid, z_mid + num_slices) if num_slices > 0 else list(range(image.shape[0])))

    for slice_depth in slices:
        # Define image name & path
        prefix = "_clean_ring_slice_ml_training_" if "mask" in image_type else "_base_ring_slice_ml_training_"
        img_name = str(image_id) + prefix + str(slice_depth)
        file_path = full_path + img_name + ".png"

        # Create and save image
        test_slice = image[slice_depth, :]
        img = Image.fromarray(test_slice)
        img.save(file_path)

        print("\rSlice Exported:\t" + str(slice_depth), end="")

    print("\n")


def get_ring_mask_image(dim, position, rotation, r_small, r_large, debug=False):

    # Create new header for terminal output
    general.print_divider("Slice Export for Machine Learning")

    # Get dimensions and positions
    ring_pos = position.get_coordinates_int() if isinstance(position, coord) else position
    max_crop = general.get_crop_dim(dim, ring_pos)

    # Get the background image & the ring image with the right angle
    base_image = np.zeros(dim)
    ring_image_obj = import_ring_image(r_small=r_small, r_large=r_large, ring_angle=rotation, filled=True)
    ring_image = ring_image_obj.get_cropped_image(crop_dim=max_crop)

    if debug:
        print("Target output image dimensions:\t" + str(dim))
        print("Actual output image dimensions:\t" + str(ring_image.shape))
        imageview3d(ring_image, windowName="Ring Image Test")

    # if not isinstance(dim[0], tuple):
    #     # Define Starting Corner
    #     ri_s = ring_image.shape
    #     s_c = np.subtract(ring_pos, np.divide(max_crop, 2).astype(np.int))
    #     base_image[s_c[0]:s_c[0]+ri_s[0], s_c[1]:s_c[1]+ri_s[1], s_c[2]:s_c[2]+ri_s[2]] += ring_image
    #     base_image = base_image.astype(np.uint8)
    #
    # else:
    base_image += ring_image

    # Print to Terminal
    print("Image for export created...")

    return base_image.astype(np.uint8)
