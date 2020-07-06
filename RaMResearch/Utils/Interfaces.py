import cv2
import numpy as np
from RaMResearch.Utils.General import BWtoRGB

import matplotlib.pyplot as plt

##### Global Parameters
depth = 0


######## Helper Functions

def castarray(imagearray):
    return imagearray.astype(np.uint8)


######## Array Display Functions

def imageview2D(array, windowName="TestWindow"):
    if isinstance(array, list):
        showarray = array[0]
        for i in range(1, len(array)):
            showarray = np.append(showarray, array[i], axis=1)
        imageview2D(showarray)
    else:
        cv2.imshow(windowName, castarray(array))
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
        cv2.destroyAllWindows()


# Viewer for arrays
def imageview3d(arrays, windowName="TestWindow"):

    def viewer3d(array, localwindowName):
        
        def trackbarHandler(value):
            global depth
            print("Depth = ", value)
            depth = value
            cv2.imshow(localwindowName, castarray(array[depth, :]))

        array = array.astype(np.uint8)
        cv2.namedWindow(localwindowName)
        cv2.createTrackbar('Depth', localwindowName, 0, array.shape[0] - 1, trackbarHandler)
        cv2.imshow(localwindowName, castarray(array[0, :]))
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

    if isinstance(arrays, list):
        arrays = [BWtoRGB(array) for array in arrays]
        if len(arrays) == 1:
            viewer3d(arrays[0], windowName)
        else:
            showarray = arrays[0]
            for i in range(1, len(arrays)):
                showarray = np.append(showarray, arrays[i], axis=2)
            viewer3d(showarray, windowName)
    else:
        arrays = BWtoRGB(arrays)
        viewer3d(arrays, windowName)


# Overlay image in red
def overelay_view4D(background_array, overlayarray, windowName="TestWindow"):

    def overlayview4D(bg_array, frontarray, localWindowname):

        def get_overlayimage(array1, array2):
            global depth
            bg_slice = array1[depth, ...].astype(np.int16)
            overlay = array2[depth, ...].astype(np.int16)
            return cv2.add(bg_slice, overlay)

        def showimage():
            cv2.imshow(localWindowname, castarray(get_overlayimage(bg_array, frontarray)))

        def trackbarHandler(value):
            global depth
            print("Depth = ", value)
            depth = value
            showimage()

        cv2.namedWindow(localWindowname)
        cv2.createTrackbar('Depth', localWindowname, 0, background_array.shape[0] - 1, trackbarHandler)

        showimage()

        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break

        cv2.destroyAllWindows()

    overlayview4D(BWtoRGB(background_array), BWtoRGB(overlayarray, True), localWindowname=windowName)


######## Plot Functions

def plot1D(yvals, xvals=[], plot_title="Default_Plot"):

    # If no x-values were passed to the function take a default range
    if not xvals:
        xvals = list(range(len(yvals)))

    plt.plot(xvals, yvals)
    plt.title(plot_title)
    plt.show()
