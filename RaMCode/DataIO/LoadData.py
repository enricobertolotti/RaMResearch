import os, fnmatch
import pydicom as pd
import tifffile as tf
from RaMCode.Data.DataStructs import DicomObject
import numpy as np
from pathlib import Path

## Find Dicoms In Folder
# Go through folder and return dcm and tiff filename pairs as a tuple (*.dcm, *.tif)

root_path = "/RaMData/"


def get_dicom_filepairs(foldername):
    dcm_infolder = [os.path.splitext(filename)[0] for filename in os.listdir(foldername) if filename.endswith(".dcm")]
    full_path = map(lambda filename: foldername + "/" + filename, dcm_infolder)
    return list(map(lambda filename: (filename + ".dcm", filename + ".tif"), full_path))


def import_normal_DICOM(DICOM_filepath: str = None):

    # Get dicom ID from path
    dicomID = DICOM_filepath.split("/")[-1].split("_")[0]

    # Create DICOM Object and load data
    dicomobject = DicomObject(dicomID)
    dicomobject.set_metadata(pd.dcmread(DICOM_filepath.replace(".dcm", "") + ".dcm"), ring_present=True)

    # TODO clean this up
    # excelfile_properties = ld.load_excel_properties(dicom_id=dicomID)
    #
    # dicomobject.set_

    dicomobject.set_image(tf.imread(DICOM_filepath.replace(".dcm", "") + ".tif"), ring_present=True)
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


def get_dicom_filepaths(data_folder="", dicom_id=None):

    # Returns (folder, dicom_filename)
    def search_dicom():
        result = []
        # loop through folders and check if there is a file that matches the pattern
        for root, dirs, files in os.walk(data_folder):
            for name in files:
                match_name = "*.dcm"
                if fnmatch.fnmatch(name, match_name):
                    result.append(os.path.join(root, name))
        return result

    complete_dicom_list = search_dicom()

    # Search for the dicom file in the folders if a dicom_id is given
    if dicom_id:
        return [[file_path for file_path in complete_dicom_list if str(dicom_id) in file_path][0]]

    # Otherwise return complete list of dicom files
    return complete_dicom_list
