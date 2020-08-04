import numpy as np
from scipy.ndimage import gaussian_laplace as gl, gaussian_filter as g
from scipy.ndimage import filters
import time
import copy

from RaMCode.Data import DataStructs as ds
from RaMCode.DataIO import StoreData as sd, LoadData as ld


def crop_array(array, size, midpoint):
    c = midpoint - [int(x / 2) for x in size]
    return array[c[0]:c[0]+size[0], c[1]:c[1]+size[1], c[2]:c[2]+size[2]]


def shiftarray(array, amount):
    return (array-amount).astype(np.int16)


def normalize(array):
    array_min, array_max = np.min(array), np.max(array)
    factor = (array_max - array_min) * 255
    return np.multiply(np.subtract(array, array_min), factor).astype(np.int8)


def multidim_gauss_grad(array, sigma=1):
    starttime = time.time()
    arrayofimages = [array]
    for i in range(3):
        arrayofimages.append(filters.gaussian_gradient_magnitude(copy.deepcopy(array), sigma=i))

    arrayofimages = list(map(normalize, arrayofimages))
    print("Multidimensional Gaussian Gradient Execution Time:\t" + str(time.time() - starttime))
    return arrayofimages


def multidim_gauss_blur(array, sigma=1):
    starttime = time.time()
    arrayofimages = []
    for i in [0, 0.3, 0.5, 0.7, 1]:
        arrayofimages.append(g(copy.deepcopy(array.astype(np.int8)), sigma=i*2))
    print("Multidimensional Gaussian Blur Execution Time:\t" + str(time.time() - starttime))
    return arrayofimages


def multidim_gauss_laplace(array, sigma=1):
    starttime = time.time()
    arrayofimages = []
    for i in range(5):
        arrayofimages.append(gl(copy.deepcopy(array).astype(np.int16), sigma=i), )
    print("Multidimensional Gaussian Laplace Execution Time:\t" + str(time.time() - starttime))
    return arrayofimages


def multidim_filter_comparison(h_func, v_func, array):
    if v_func is None:
        return [image for image in h_func(array)]
    else:
        return [np.concatenate(v_func(image), axis=1) for image in h_func(array)]


def insert_white_around_edges(dicom_object: ds.DicomObject):

    angle = (90 - dicom_object.angular_scope) * 3.14 / 180.0
    dim = dicom_object.image_with_ring.shape

    # Check if mask exists
    existing_mask = ld.import_numpy_mask(dim)

    if existing_mask is not None:
        mask_array = existing_mask
    else:
        # Create mask
        mask_array = np.empty(dim)
        xmax = int(dim[1]*4/9)
        zmax = int(dim[0]*3/7)

        value = 255
        offset = 80

        for x in range(xmax):
            maxy = np.ceil(np.tan(angle) * (xmax-x)).astype(np.int16)
            mask_array[:, x, 0:maxy] = value
            mask_array[:, dim[1]-x-1, 0:maxy] = value

        for z in range(zmax):
            maxy = np.ceil(np.tan(angle) * (zmax-z)*2).astype(np.int16)
            mask_array[z, :, 0:maxy] = value
            mask_array[dim[0]-z-1, :, 0:maxy] = value

        mask_array[:, :, 0:offset] = value

        z_mid = int(np.ceil(dim[0]/2))
        x_mid = int(np.ceil(dim[1]/2))

        for z_off in range(z_mid + 1):
            for x_off in range(x_mid + 1):
                y_start = np.clip(2.3 * ((x_mid - x_off) / 30) ** 2, 0, 100).astype(np.uint8)
                extra_offset = np.clip(2.3 * ((z_mid - z_off) / 20) ** 2, 0, 100).astype(np.uint8)

                mask_array[z_off, x_off, dim[2] - y_start - extra_offset:dim[2]] = value
                mask_array[dim[0] - z_off-1, dim[1] - x_off - 1, dim[2] - y_start - extra_offset:dim[2]] = value

                mask_array[dim[0] - z_off-1, x_off, dim[2] - y_start - extra_offset:dim[2]] = value
                mask_array[z_off, dim[1] - x_off - 1, dim[2] - y_start - extra_offset:dim[2]] = value

        # Store the mask in the filesystem
        sd.store_mask(mask_array.astype(np.uint8))

    dicom_object.image_with_ring = np.clip(dicom_object.image_with_ring.astype(np.uint16) + mask_array, 0, 255).astype(np.uint8)