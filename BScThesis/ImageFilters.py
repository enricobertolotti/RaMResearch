import numpy as np
import cv2
from scipy import ndimage
from BScThesis import DicomScripts as ds
import time
from skimage.filters import threshold_local, scharr, prewitt, roberts


def thresh(array, threshhold):
    size = array.shape
    if len(size) == 3:
        for i in range(size[0]):
            array[i] = cv2.threshold(array[i, :], threshhold, 0, cv2.THRESH_TOZERO)[1]
        return array
    return cv2.threshold(array, threshhold, 0, cv2.THRESH_TOZERO)[1]


def threshholdBinary(array, threshhold):
    size = array.shape
    if len(size) == 3:
        for i in range(size[0]):
            array[i] = cv2.threshold(array[i, :], threshhold, 0, cv2.THRESH_TOZERO)[1]
        return array
    return cv2.threshold(array, threshhold, 0, cv2.THRESH_TOZERO)[1]


def adaptiveThreshold(array, block_size):
    offset = 0
    size = array.shape

    if len(size) == 3:
        returnarray = np.zeros(array.shape)
        for i in range(size[0]):
            returnarray[i] = cv2.adaptiveThreshold(array[i], maxValue=255,
                                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   thresholdType=cv2.THRESH_BINARY, blockSize=block_size, C=-30)
        return returnarray.astype(np.uint8)
    return array > threshold_local(array, block_size=block_size, offset=offset)


def sobel_dicom(array, ksize):

    def applysobel(array2D, ksize):
        grad_x = cv2.Sobel(array2D, ddepth=-1, dx=1, dy=0, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(array2D, ddepth=-1, dx=0, dy=1, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    if len(array.shape) == 3:
        returnarray = np.zeros(array.shape)
        for i in range(array.shape[0]):
            returnarray[i] = applysobel(array[i], ksize=ksize)
            print(np.max(returnarray[i]), "   ", np.min(returnarray[i]))
        return returnarray.astype(np.uint8)

    return applysobel(array, ksize=ksize).astype(np.uint8)


def prewitt_dicom(array):

    def ra_range(ra_array):
        return np.min(ra_array), np.max(ra_array)

    if len(array.shape) == 3:
        returnarray = np.zeros(array.shape)
        for i in range(array.shape[0]):
            returnarray[i] = prewitt(array[i])
        returnarray = np.interp(returnarray, ra_range(returnarray), (0, 255))
        return np.interp(returnarray, ra_range(returnarray), (0, 255)).astype(np.uint8)

    filteredim = prewitt(array)
    return np.interp(filteredim, ra_range(filteredim), (0, 255)).astype(np.uint8)


def scharr_dicom(array):

    def ra_range(ra_array):
        return np.min(ra_array), np.max(ra_array)

    if len(array.shape) == 3:
        returnarray = np.zeros(array.shape)
        for i in range(array.shape[0]):
            returnarray[i] = scharr(array[i])
        returnarray = np.interp(returnarray, ra_range(returnarray), (0, 255))
        return returnarray.astype(np.uint8)

    filteredim = scharr(array)
    return np.interp(filteredim, ra_range(filteredim), (0, 255)).astype(np.uint8)


def roberts_dicom(array):

    def ra_range(ra_array):
        return np.min(ra_array), np.max(ra_array)

    if len(array.shape) == 3:
        returnarray = np.zeros(array.shape)
        for i in range(array.shape[0]):
            returnarray[i] = roberts(array[i])
        returnarray = np.interp(returnarray, ra_range(returnarray), (0, 255))
        return returnarray.astype(np.uint8)

    filteredim = roberts(array)
    return np.interp(filteredim, ra_range(filteredim), (0, 255)).astype(np.uint8)


def cannyedgevol(array, threshlow, threshhigh):
    returnarray = np.zeros(array.shape)
    for i in range(array.shape[0]):
        returnarray[i] = cv2.Canny(array[i, :], threshlow, threshhigh)
    returnarray = np.multiply(returnarray, 255)
    return returnarray.astype(np.uint8)


def gausdist2d(array, mu, sigma):

    # Create Gaussian Distribution
    def gaussian(x, muvar, sig):
        return np.exp(-np.power(x - muvar, 2.) / (2 * np.power(sig, 2.)))

    # Create 3D Gaussian value matrix
    def gaussianTable1D(pixelrange, axis=0):

        if axis == 0:
            xs = np.linspace(-2, 2, num=pixelrange[1])
        else:
            xs = np.linspace(-2, 2, num=pixelrange[2])

        y = []
        for x in xs:
            y.append(gaussian(x, mu, sigma))
        normy = y / (np.max(y))

        if axis == 0:
            gausdistarray = np.repeat(normy[:, np.newaxis], pixelrange[1], axis=1)
        else:
            gausdistarray = np.repeat(normy[np.newaxis, :], pixelrange[0], axis=0)

        return gausdistarray

    gaussianarray = gaussianTable1D(array.shape)
    returnarray = np.multiply(array, gaussianarray)

    return returnarray.astype(np.uint8)


def gausdist3d(array, mu, sigma):

    # Create Gaussian Distribution
    def gaussian(x, muvar, sig):
        return np.exp(-np.power(x - muvar, 2.) / (2 * np.power(sig, 2.)))

    # Create 3D Gaussian value matrix
    def gaussianTable1D(pixelrange, axis=0):

        if axis == 0:
            xs = np.linspace(-2, 2, num=pixelrange[1])
        else:
            xs = np.linspace(-2, 2, num=pixelrange[2])

        y = []
        for x in xs:
            y.append(gaussian(x, mu, sigma))
        normy = y / (np.max(y))

        if axis == 0:
            returnarray = np.repeat(normy[:, np.newaxis], pixelrange[2], axis=1)
        else:
            returnarray = np.repeat(normy[np.newaxis, :], pixelrange[1], axis=0)

        returnarray = np.repeat(returnarray[np.newaxis, :], pixelrange[0], axis=0)
        return returnarray

    gaussianarray = gaussianTable1D(array.shape)
    returnarray = np.multiply(array, gaussianarray)

    return returnarray.astype(np.uint8)


def weightdepth(dicomfile, sigma):

    def gaussian(x, muvar, sig):
        return np.exp(-np.power(x - muvar, 2.) / (2 * np.power(sig, 2.)))

    # Create a copy of the imput array
    array = np.copy(dicomfile)

    # If the input array is 2D:
    if array.shape == 2:
        return array

    # Iterate and weight each layer:
    else:
        xs = np.linspace(-2, 2, num=array.shape[0])

        for i in range(len(xs)):
            xs[i] = gaussian(xs[i], 0, sigma)

        for layer in range(array.shape[0]):
            array[layer, :] = np.multiply(array[layer, :], xs[layer])
        return array


def morphological_filter(array):

    image = array
    filtered_image = ndimage.gaussian_filter(image, [2, 2, 2], 0)
    morphkernel = np.ones((4, 4), np.uint8)

    for layer in range(filtered_image.shape[0]):
        filtered_image[layer, :] = cv2.threshold(filtered_image[layer, :], 160, 255, cv2.THRESH_BINARY)[1]
        filtered_image[layer] = cv2.morphologyEx(filtered_image[layer], cv2.MORPH_OPEN, morphkernel)

    return filtered_image.astype(np.uint8)


def morph_dicom(array, kernelsize):

    kernel = np.ones((kernelsize, kernelsize), np.uint8)
    returnarray = np.zeros(array.shape)

    for layer in range(array.shape[0]):
        returnarray[layer] = cv2.morphologyEx(array[layer, :], cv2.MORPH_OPEN, kernel=kernel)

    debug = np.min(returnarray), np.max(returnarray)
    return returnarray.astype(np.uint8)


def gauslogfilter(array3D, verticalsigma, logsigma, gaus1D=True, morphological=True, morphkernelsize=4):

    starttime = time.time()
    # Create a copy of the imput array
    array = np.copy(array3D)

    # Use the probability to help find the ring
    # array = weightdepth(array, 0.5)

    # kernel for morphological operations

    # if the input array is 2D:
    if array.shape == 2:
        dicomfile = ndimage.gaussian_laplace(ndimage.gaussian_filter1d(array3D, verticalsigma, 0), logsigma)

    # Otherwise the input array is a 3D file:
    else:
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
        print("gauslogfilter: " + str(endtime - starttime) + " seconds")
        return array


# sigmaColor
# # Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel
# # neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.

# sigmaSpace
# Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel
# neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.


# Global variables
sigma_Color_global = 0
sigma_Space_global = 0


def dicom_bilateral_filter(image, sigma_Color, sigma_Space, view=False):

    filtered_im = cv2.bilateralFilter(image, d=-1, sigmaColor=41, sigmaSpace=7)

    if view:

        global sigma_Color_global
        sigma_Color_global = sigma_Color

        global sigma_Space_global
        sigma_Space_global = sigma_Space

        def sigma_ColorHandler(value):
            global sigma_Color_global
            print("Sigma_Color = ", value, "Sigma_Space = ", sigma_Space_global)
            sigma_Color_global = value
            filtered_image = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_Color_global, sigmaSpace=sigma_Space_global)
            cv2.imshow(windowname, filtered_image)

        def sigma_SpaceHandler(value):
            global sigma_Space_global
            print("Sigma_Color = ", sigma_Color_global, "Sigma_Space = ", sigma_Space_global)
            sigma_Space_global = value
            filtered_image = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_Color_global, sigmaSpace=sigma_Space_global)
            cv2.imshow(windowname, filtered_image)

        windowname = "Dicom Bilateral Filter"
        cv2.namedWindow(windowname)

        cv2.createTrackbar('Sigma_Color', windowname, 0, 100, sigma_ColorHandler)
        cv2.createTrackbar('Sigma_Space', windowname, 0, 100, sigma_SpaceHandler)

        cv2.imshow(windowname, filtered_im)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

    return filtered_im


def dicom_bilateral_filter_3D(dicomfile, sigma_Color, sigma_Space, image_type="filtered"):

    if "filter" in image_type:
        array_3d = dicomfile.get_image("filter").get_array("base").copy()
    else:
        array_3d = dicomfile.get_image("normal").get_array("base").copy()

    starttime = time.time()

    # Create return array
    returnarray = np.zeros(array_3d.shape)

    for layer in range(array_3d.shape[0]):
        returnarray[layer] = dicom_bilateral_filter(array_3d[layer], sigma_Color=sigma_Color, sigma_Space = sigma_Space)

    dicomfile.set_image(returnarray.astype(np.uint8), image_type="filtered")

    endtime = time.time()
    print("dicom_bilateral_filter_3D: " + str(endtime - starttime) + " seconds")


def quauntize_3D(dicomfile, image_type="normal", array_type="base", n=8, o=0.5):

    def dicom_quantize(image, num_levels, offset):
        step = int(256/num_levels)
        image = (image / step).astype(np.uint8) * step
        return (image + step*offset - 1).astype(np.uint8)

    array_3D = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)

    return np.array([dicom_quantize(array_3D[depth], n, o) for depth in range(array_3D.shape[0])])


# ClipLimit:
# Tile Grid Size: The local area where adaptive equalisation is performed

def adaptive_histogram_equalisation(dicomfile, image_type="filter", cliplimit=2, tilegridsize=(12, 12), view=False):
    starttime = time.time()
    img = dicomfile.get_image(image_type=image_type).get_array("base").copy()
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    return_image = np.array([clahe.apply(img[layer].copy()) for layer in range(img.shape[0])]).astype(np.uint8)
    dicomfile.get_image(image_type=image_type).set_array(return_image, array_type="base")
    endtime = time.time()
    print("adaptive_histogram_equalisation: " + str(endtime-starttime) + " seconds")

    if view:
        ds.viewer3d(dicomfile.get_image(image_type=image_type).get_array("base"), "Adaptive Histogram Equalisation")


def overlayimage(image1, image2):

    def pad_image_to_size(image, endsize):
        pd_left = int(0.5*endsize[0] - 0.5*image.shape[0])
        pd_right = endsize[0] - image.shape[0] - pd_left
        pd_top = int(0.5 * endsize[0] - 0.5 * image.shape[0])
        pd_bottom = endsize[1] - image.shape[1] - pd_top
        padding = ((pd_left, pd_right), (pd_top, pd_bottom))
        return np.pad(image, padding)

    im1_padded = pad_image_to_size(image1, image2.shape)

    # Create rgb
    returnimage = np.zeros((image2.shape[0], image2.shape[1], 3))
    returnimage[:, :, 0] = image2.astype(np.uint8)
    for i in range(3):
        returnimage[:, :, i] = np.multiply(image2, im1_padded/255).astype(np.uint8)

    cv2.namedWindow("test")
    cv2.imshow("test", returnimage)
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    return returnimage

