import os
import pydicom as pd
import tifffile as tf
from RaMResearch.Data import DataStructs as ds


## Find Dicoms In Folder
# Go through folder and return dcm and tiff filename pairs as a tuple (*.dcm, *.tif)

def get_dicom_filepairs(foldername):
    dcm_infolder = [os.path.splitext(filename)[0] for filename in os.listdir(foldername) if filename.endswith(".dcm")]
    full_path = map(lambda filename: foldername + filename, dcm_infolder)   
    return list(map(lambda filename: (filename + ".dcm", filename + ".tif"), full_path))


## Import Files To A Dicomobject
# Takes a tuple of a _*.dcm_ file and a _*.tif_ file and returns a Dicomobject as defined in Datastructs.py

def create_dicom_object(filename_tuples):
    dicomobject = ds.DicomObject(filename_tuples[0].split(".")[0])

    for (dicomname, tifname) in filename_tuples:
        dicomobject.set_metadata(pd.dcmread(dicomname), ring_present="(5)" in dicomname)
        dicomobject.set_image(tf.imread(tifname), ring_present="(5)" in dicomname)

    return dicomobject
