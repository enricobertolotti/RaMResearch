from RaMResearch.Analysis.Rotation import RotationAnalysis as ra
from PIL import Image
from pathlib import Path as path
import numpy as np

default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/"


# Take a rotation object and export the masks for machine learning
def export_ring_slices(rot_anal_obj: ra.RotationAnalysis, image_id, num_slices=100, debug=False):

    # Folder & Path preparations:
    subfolder = "ML_Training_Data/" + str(image_id) + "/"
    full_path = default_folder + subfolder
    path_obj = path(full_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Get the export-image
    export_image = rot_anal_obj.get_export_image(debug=debug)

    # Create a list of depth indexes for the slices to be exported
    z_mid = int(export_image.shape[0] / 2)
    slices = list(range(z_mid, z_mid + num_slices) if num_slices > 0 else list(range(export_image.shape[0])))

    for slice_depth in slices:
        # Define image name & path
        img_name = str(image_id) + "_clean_ring_slice_ml_training_" + str(slice_depth)
        file_path = full_path + img_name + ".png"

        # Create and save image
        test_slice = export_image[slice_depth, :]
        img = Image.fromarray(test_slice)
        img.save(file_path)
