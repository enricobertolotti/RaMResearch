import time
import numpy as np
from scipy import ndimage
import cv2
from RaMCode.Utils import Interfaces as intrfc

from RaMCode.Data import DataStructs as ds


# In place image filter
def gauslogfilter(imageobject, ring_present=True, verticalsigma=7, logsigma=4, gaus1D=True,
                  morphological=True, morphkernelsize=3, debug=False):

    # Get the image depending on the object type
    if isinstance(imageobject, ds.DicomObject):
        array = imageobject.get_image(ring_present=ring_present).get_image(filtered=True)
    elif isinstance(imageobject, ds.Image):
        array = imageobject.get_image(filtered=True)
    else:
        array = imageobject

    # Store current time
    starttime = time.time()

    # Loop through the layers of the dicom file
    for layer in range(array.shape[0]):

        # Apply vertical filtering
        if gaus1D:
            array[layer, :] = ndimage.gaussian_filter1d(array[layer, :], verticalsigma, axis=0, order=0)

        # Apply LoG and morphological filters
        array[layer, :] = ndimage.gaussian_laplace(array[layer, :], logsigma)
        if morphological:
            morphkernel = np.ones((morphkernelsize, morphkernelsize), np.uint8)
            array[layer, :] = cv2.morphologyEx(array[layer, :], cv2.MORPH_OPEN, kernel=morphkernel)
    endtime = time.time()

    # Store filtering information
    if isinstance(imageobject, ds.DicomObject):
        imageobject.get_image(ring_present=ring_present).filter_info.append(
            ("Gauslogfilter", (verticalsigma, logsigma)))
    elif isinstance(imageobject, ds.Image):
        imageobject.filter_info.append(("Gauslogfilter", (verticalsigma, logsigma)))

    print("Gauslogfilter: " + str(endtime - starttime) + " seconds")

