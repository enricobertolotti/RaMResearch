import os
import pydicom as pd
import tifffile as tf
from RaMResearch.Data.DataStructs import DicomObject
import numpy as np
from pathlib import Path

## Find Dicoms In Folder
# Go through folder and return dcm and tiff filename pairs as a tuple (*.dcm, *.tif)

root_path = "/RaMData/"


def get_dicom_filepairs(foldername):
    dcm_infolder = [os.path.splitext(filename)[0] for filename in os.listdir(foldername) if filename.endswith(".dcm")]
    full_path = map(lambda filename: foldername + "/" + filename, dcm_infolder)
    return list(map(lambda filename: (filename + ".dcm", filename + ".tif"), full_path))


def import_normal_DICOM(DICOM_folder, DICOM_filename="", DICOM_image_ID=""):

    # Remove .dcm ending if necessary
    DICOM_filename = DICOM_filename.split(".dcm")[0] if ".dcm" in DICOM_filename else DICOM_filename

    # Remove path if necessary
    DICOM_filename = DICOM_filename.split("/")[-1] if "/" in DICOM_filename else DICOM_filename

    # Find the dicom file in the folder if required
    if len(DICOM_image_ID) > 0:
        dcm_infolder = [os.path.splitext(filename)[0] for filename in os.listdir(DICOM_folder) if
                        filename.endswith(".dcm") and DICOM_image_ID in filename]
        if len(dcm_infolder) > 0:
            DICOM_filename = dcm_infolder[0]

    # If the filename was left empty
    if not DICOM_filename:
        dcm_infolder = [os.path.splitext(filename)[0] for filename in os.listdir(DICOM_folder) if
                        filename.endswith(".dcm")]
        DICOM_filename = dcm_infolder[0]

    dicomobject = DicomObject(DICOM_filename)

    # Import and set the array
    dicom_file_path = DICOM_folder + '/' + DICOM_filename

    dicomobject.set_metadata(pd.dcmread(dicom_file_path + ".dcm"), ring_present=True)
    dicomobject.set_image(tf.imread(dicom_file_path + ".tif"), ring_present=True)

    dicomobject.transpose_images()
    return dicomobject


def import_experimental_dicom(foldername, dicomfilename):

    dicomobject = DicomObject(dicomfilename)

    # Import and set the array
    dicomheadername = foldername + '/' + dicomfilename + '.dcm'
    dicomfile = pd.dcmread(dicomheadername)
    dicomobject.set_image(np.rot90(dicomfile.pixel_array, k=1, axes=(1, 2)), ring_present=True)

    # Set the dicomfile as metadata file
    dicomobject.set_metadata(dicomfile, ring_present=True)

    # Try to set the slice thickness if it exists
    try:
        dicomobject.slice_thickness = dicomfile.SliceThickness
    except:
        dicomobject.slice_thickness = 0.345

    # Set the experimentally recorded rotation and depth in the object
    str_rot = list(map(lambda x: float(x), dicomfile.PatientName.family_name.split(",")))
    dicomobject.rotation = (str_rot[0], str_rot[1])

    depth = dicomfile.PatientName.given_name.replace(",", ".")

    if len(depth) > 0:
        dicomobject.depth = float(depth)
    else:
        dicomobject.depth = 10.1

    # Return the newly created object
    return dicomobject


def import_numpy_mask(dimensions, folder=""):

    # Folder + File Preparations
    folder = "Numpy_Dicom_Mask" if len(folder) == 0 else folder
    folder_path = root_path + folder + '/'
    mask_array_dim_str = [str(x) for x in dimensions]
    separator = "_"
    save_name = separator.join(mask_array_dim_str) + "_numpy_mask.npy"

    # Create Path Object
    full_path = folder_path + save_name
    file_path = Path(full_path)

    if file_path.exists():
        return np.load(full_path)
    else:
        return None
