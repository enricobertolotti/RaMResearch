import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy
from pydicom.data import get_testdata_files
from scipy import ndimage



def nothing(x):
    pass


filename = get_testdata_files("emri_small.dcm")[0]
ds = pydicom.dcmread(filename)


# Normalize the image
npdicom = (ds.pixel_array - ds.pixel_array.min()) / ds.pixel_array.max()
npdicom8 = (npdicom*255).astype(numpy.uint8)

windowname = '3D View'

cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
cv2.resizeWindow(windowname, 500, 500)
zPos = 0
(zMax, _, _) = npdicom8.shape

cv2.createTrackbar('z-Depth', windowname, zPos, zMax-1, nothing)
scale = 3
width = npdicom8.shape[2] * scale
height = npdicom8.shape[1] * scale
dim = (width, height)

img = numpy.ndarray((zMax, width, height))

for i in range(zMax):
    img[i] = cv2.resize(npdicom8[i], dim, interpolation=cv2.INTER_CUBIC)


while(1):
    k = cv2.waitKey(1)
    if k == 27:
        break

    # Depth = cv2.getTrackbarPos('z-Depth', windowname)
    Depth = 1
    Threshhold = cv2.getTrackbarPos('z-Depth', windowname)
    (_, img) = cv2. threshold(img[0], Threshhold, 0, cv2.THRESH_BINARY_INV)
    # cv2.imshow(windowname, cv2.applyColorMap(img[Depth, ...], cv2.COLORMAP_PINK))
    cv2.imshow(windowname, img)
cv2.destroyAllWindows()
