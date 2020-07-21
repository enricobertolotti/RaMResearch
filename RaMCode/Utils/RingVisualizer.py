import numpy as np
import cv2


def getPointsOnCircle(diameters, slice_dim):
    x_mid, y_mid = np.divide(slice_dim, 2).astype(np.uint8)
    inner_radius, outer_radius = np.divide(diameters, 2).astype(np.uint8)
    image = np.zeros(slice_dim)
    cv2.circle(image, (x_mid, y_mid), outer_radius, (255, 255, 255), thickness=1)
    cv2.circle(image, (x_mid, y_mid), inner_radius, (255, 255, 255), thickness=1)
    cv2.imshow("testwindow", image)
    cv2.waitKey(0)


def getRingPoints(main_diameter, ring_thickness, array_dim):

    return_diams = []

    def getDiameters(diameter, thickness, height):
        offset = np.sqrt(thickness ** 2 - height ** 2)
        return diameter - offset, diameter + offset

    for i in range(0, ring_thickness+1):
        return_diams.append(getDiameters(main_diameter, ring_thickness, i))

    return return_diams


print(getRingPoints(90, 10, 10))
# getPointsOnCircle((20, 5), (60, 60))
