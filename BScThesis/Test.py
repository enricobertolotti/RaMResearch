import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy
import scipy.misc
from pydicom.data import get_testdata_files
import time

print("starting timer")
start = time.time()

time.sleep(3)

endtime = time.time()

print("Elapsed time: " + str(endtime - start))


def nothing(x):
    pass


filename = get_testdata_files("emri_small.dcm")[0]
ds = pydicom.dcmread(filename)

# plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
# plt.show()

# Normalize the image
npdicom = (ds.pixel_array - ds.pixel_array.min()) / ds.pixel_array.max()
npdicom8 = (npdicom*255).astype(numpy.uint8)

# Create Threshold variables
(ThreshLow, ThreshHigh) = (100, 200)

# Window Name
windowname = 'Test Window'

# Create Window
cv2.namedWindow(windowname)
# Create Trackbars

# Find boundaries
edges = cv2.Canny(npdicom8, ThreshLow, ThreshHigh)


cv2.namedWindow('OriginalImage')
im_color = cv2.applyColorMap(npdicom8, cv2.COLORMAP_PINK)
cv2.imshow('OriginalImage', im_color)

ksize = 99

# Gaussian Blur
cv2.namedWindow('Gaussian Adaptive Thresholding')
cv2.createTrackbar('KSize', 'Gaussian Adaptive Thresholding', ksize, 10, nothing)

cv2.namedWindow('Combined')
cv2.createTrackbar('ThreshLow', 'Combined', ThreshLow, 255, nothing)
cv2.createTrackbar('ThreshHigh', 'Combined', ThreshHigh, 255, nothing)


while(1):




    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of trackbars
    ThreshLow = cv2.getTrackbarPos('ThreshLow', 'Combined')
    ThreshHigh = cv2.getTrackbarPos('ThreshHigh', 'Combined')
    edges = cv2.Canny(npdicom8, ThreshLow, ThreshHigh)
    cv2.imshow(windowname, edges)

    ksizenew = cv2.getTrackbarPos('KSize', 'Gaussian Adaptive Thresholding')

    if not ksizenew % 2:
        ksizenew += 1


    ksize = ksizenew
    img = cv2.GaussianBlur(npdicom8, (ksize, ksize), 1, 0)
    img = cv2.adaptiveThreshold(cv2.GaussianBlur(npdicom8, (ksize, ksize), 1, 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 7, 2)
    cv2.imshow('Gaussian Adaptive Thresholding', img)

    im_h = cv2.hconcat([img, edges])
    cv2.imshow('Combined', im_h)

cv2.destroyAllWindows()