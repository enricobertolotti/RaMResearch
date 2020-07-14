from RaMResearch.Analysis.Rotation import RotationAnalysis as ra
from pathlib import Path as path
import numpy as np

default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/"


# Take a rotation object and export the masks for machine learning
def export_ring_slices(rot_anal_obj: ra.RotationAnalysis, image_id: str):

    # Folder & Path preparations:
    subfolder = "ML_Training_Data/" + image_id + "/"
    full_path = default_folder + subfolder
    path_obj = path(full_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    # Create the base-image
    image_size = rot_anal_obj.get_analysis_image().shape
    export_image = np.zeros(image_size)

    # Overlay the ring

    # Save the individual slices
