import numpy as np
import cv2
import pandas as pd
from skimage.transform import resize
from enum import Enum
from colour import Color
import datetime as dt
from BScThesis import DataAnalysisAndFitting as df, ExportScripts as es, DicomScripts as ds
import re

# Countours
import imutils

# Blob detector
import matplotlib.pyplot as plt


# Stores pessary size in mm
class PessarySize(Enum):
    US1  = 44
    US2  = 51
    US3  = 57
    US4  = 64
    US5  = 70
    US6  = 76
    US7  = 83
    US8  = 89
    US9  = 95
    US10 = 102
    US11 = 108
    US12 = 114
    US13 = 121


class PessaryType(Enum):
    PT1 = "Base_Incl_TypePessaryUS#Ring_without_support"
    PT2 = "Base_Incl_TypePessaryUS#Ring_with_support"
    PT3 = "Base_Incl_TypePessaryUS#Gellhorn"
    PT4 = "Base_Incl_TypePessaryUS#Shaatz"
    PT5 = "Base_Incl_TypePessaryUS#Donut"
    PT6 = "Base_Incl_TypePessaryUS#Cube"
    PT7 = "Base_Incl_TypePessaryUS#Gehrung"


class Patient:
    id = None
    pessary = False
    pessarytype = None
    pessarysize = None

    def __init__(self, patientid):
        self.id = patientid

    def sethaspessary(self, haspessary):
        self.pessary = haspessary

    def setpessarytype(self, pessarytype):
        self.pessarytype = pessarytype

    def setpessarysize(self, pessarysize):
        self.pessarysize = pessarysize

    def getpatientid(self):
        return self.id

    def gethaspessary(self):
        return self.pessary

    def getpessarytype(self):
        return self.pessarytype

    def getpessarysize(self):
        return self.pessarysize


# Returns a scaled 3D axis plot
def getReferenceAxis(filepath, xaxis, yaxis, zaxis):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    for i in range(3):
        for j in range(3):
            image[i, j] = 255
    res_image = resize(image, (yaxis, zaxis))
    return np.repeat(res_image[np.newaxis, :, :], xaxis, axis=0)


def normalize(image):
    imagemax = image.max()
    if imagemax > 0:
        return (image - image.min()) / (imagemax - image.min())
    else:
        return (image - image.min()) * imagemax


# noinspection PyTypeChecker
# Go through excel file and return a patient object
def makePatientData(dicomfilename, excelfilepath):

    # Get Patient ID from filename
    filename = dicomfilename.split('/')
    patientidnum = [int(s) for s in filename[-1].split('_') if s.isdigit()]

    # Create New Patient Object
    patient = Patient(patientidnum[0])

    # Read excel file
    df_excel = pd.read_excel(excelfilepath, sheet_name="Study results")

    # Has Pessary:
    pessaryindex = 25  # Z = Index 25

    # Pessary Type:
    typeindex = range(42, 49)  # AQ-AW = Index 42 - 48

    # Pessary Size:
    sizeindex = range(50, 63)  # AY-BK = Index 50 - 62

    for rowindex in range(len(df_excel.iloc[:, 0])):
        if df_excel.iloc[rowindex, 0] == str(patientidnum[0]):
            print(patientidnum[0], "found at", rowindex)

            patient.sethaspessary(df_excel.iloc[rowindex, pessaryindex] == 1)

            # If the patient has a pessary
            if patient.gethaspessary():

                # Find and set patient ring size
                ringsizeindex = list(df_excel.iloc[rowindex, sizeindex].values).index(1)
                patient.setpessarysize(list(PessarySize)[ringsizeindex].value)

                # Find and set patient ring type
                ringtypeindex = list(df_excel.iloc[rowindex, typeindex].values).index(1)
                patient.setpessarytype(list(PessaryType)[ringtypeindex].value)
            break

    return patient


def printWhiteSpots(array):
    arraymax = array.max()

    if len(array.shape) == 2:
        for (x, y), val in np.ndenumerate(array):
            if val > 0.95 * arraymax:
                print(x, y, val, sep="\t")
    else:
        for (x, y, z), val in np.ndenumerate(array):
            if val > 0.95 * arraymax:
                print(x, y, z, val, sep="\t")


def cvconvoluteimage(image, pattern):

    convimage = cv2.filter2D(image, cv2.CV_16U, pattern / pattern.size)
    convinvimage = cv2.filter2D(np.subtract(255, image), cv2.CV_16U, (255 - pattern) / pattern.size)
    print(convimage.max(), convimage.min(), convinvimage.max(), convinvimage.min())
    returnimage = np.multiply(convimage,  convinvimage)
    return returnimage


def imageView2D(image, windowname):
    cv2.namedWindow(windowname)

    cv2.imshow(windowname, image)

    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

#
# def findPatternInImage(image, pattern):
#     volumedim = image.shape
#
#     if len(volumedim) == 2:
#         return cvconvoluteimage(image, pattern)
#     else:
#         returnarray = np.zeros(volumedim)
#         for i in range(volumedim[0]):
#             returnarray[i] = cvconvoluteimage(image[i, :], pattern)
#             print(i)
#         return np.divide(returnarray, returnarray.max())


def findPatternInImageTemplate(image, pattern):

    ra_shape = (image.shape[0], image.shape[1] - pattern.shape[0] + 1, image.shape[2] - pattern.shape[1] + 1)
    returnarray = np.zeros(ra_shape)

    for i in range(image.shape[0]):
        returnarray[i] = cv2.matchTemplate(image[i, :], pattern, cv2.TM_CCOEFF)
        print(i)

    return np.clip(returnarray, 0.6*np.max(returnarray), np.max(returnarray))


def getminmaxloc(array):
    maxvalue = 0
    # (value, (depth, y, z))
    maxelement = [0, (0, 0, 0)]
    for i in range(array.shape[0]):
        localmax = cv2.minMaxLoc(array[i, :])
        if localmax[1] > maxvalue:
            maxelement[0] = localmax[1]
            maxelement[1] = (i, localmax[3][0], localmax[3][1])
            maxvalue = localmax[1]
    return maxelement


def drawcrosshairs(dicomfile, xyzpos, linethickness, offset=(0, 0)):
    array = np.zeros(dicomfile.shape)
    depth = xyzpos[0]
    yz = (xyzpos[1] + offset[0], xyzpos[2] + offset[1])
    arrayyzdim = array[depth, :].shape
    cv2.line(array[depth, :], (yz[0], 0), (yz[0], arrayyzdim[0]), 255, linethickness)
    cv2.line(array[depth, :], (0, yz[1]), (arrayyzdim[1], yz[1]), 255, linethickness)
    return np.clip(array + dicomfile, 0, 255).astype(np.uint8)


def drawcontourconnections(dicomfile, linethickness, depth):

    def getcolor(color_array, connection_weight):
        color_index = int((1 - connection_weight) * (len(color_array) - 1))
        color = color_array[color_index]
        rgbcolor = color.get_rgb()
        return np.asarray(rgbcolor) * 255

    green = Color("green")
    colors = list(green.range_to(Color("red"), 90))

    colorimage = gray2rgb(dicomfile.get_image("filtered").get_array()[depth].copy())

    for (pos, _, weight) in dicomfile.getcontourconnections(slice_depth=depth):
        if weight >= 0:
            line_color = getcolor(colors, weight)[..., ::-1]
            cv2.line(colorimage, pos[0], pos[1], color=line_color, thickness=linethickness, )

    dicomfile.contour_images.append((colorimage, depth))
    print("Drew Contours: " + str(depth))
    # cv2.imshow("Contour Connections", colorimage)


def getAbsPos(originalimage, convoledimage, convolvedpos):
    xoffset = int((originalimage.shape[1] - convoledimage.shape[1]) / 2)
    yoffset = int((originalimage.shape[2] - convoledimage.shape[2]) / 2)
    return [convolvedpos[0], convolvedpos[1]+xoffset, convolvedpos[2]+yoffset]


def cvcontour(array, depth):

    # Threshold incoming array
    _, image = cv2.threshold(np.copy(array), 128, 255, cv2.THRESH_BINARY)

    # Create and get contours
    cnts  = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Turn image into colored image
    # col_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Create return array
    contourarray_w_center = []

    # Loop over the contours
    for c in cnts:
        # Compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cZ = depth
        loc = (cX, cY, cZ)

        contourarray_w_center.append((loc, c))

        # draw the contour and center of the shape on the image
        # cv2.drawContours(col_image, [c], -1, (0, 255, 0), 3)
        # cv2.circle(blank_image, (cX, cY), 7, 255, -1)
        # cv2.putText(blank_image, "center", (cX - 20, cY - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    return contourarray_w_center

    # contours_poly = [None] * len(cnts)
    # boundRect = [None] * len(cnts)
    #
    # color = (0, 0, 255)
    #
    # # Create bounding boxes
    # for i, c in enumerate(cnts):
    #     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    #     boundRect[i] = cv2.boundingRect(contours_poly[i])
    #     print(boundRect[i])
    #
    # # Draw bounding boxes:
    # for i in range(len(cnts)):
    #     if boundRect[i][2] * boundRect[i][3] > 200:
    #         cv2.rectangle(col_image, (int(boundRect[i][0]),
    #         int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] +
    #         boundRect[i][3])), color, 2)
    #
    # return rgb2gray(col_image)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def gray2rgb(grayscale_image):
    return np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=2)


def cvcontour3D(array):
    arraydim = array.shape

    returnarray = np.zeros(arraydim)

    for depth in range(array.shape[0]):
        returnarray[depth] = cvcontour(array[depth])

    return returnarray


def getdatetime_str():
    datetimeobject = dt.datetime.today()
    date = datetimeobject.date()
    time = datetimeobject.time()
    return date.isoformat() + "_" + str(time.hour) + "-" + str(time.minute)


def visualize_rotation_plots(dicomfilelist):

    visualize_rotation_plot(dicomfilelist, image_type="normal")
    # visualize_rotation_plot(dicomfilelist, image_type="filter")

    es.save_ring_pos(dicomfilelist, which_pos="normal_fit", folder="rotation_analysis/slice_comparison/",
                     filename="plot_params")
    # es.save_ring_pos(dicomfilelist, which_pos="filter_fit", folder="rotation_analysis/slice_comparison/",
    #                  filename="plot_params")


def visualize_rotation_plot(dicomfilelist: [ds.Dicomimage], image_type="normal"):

    def get_dicom_number(dicom):
        if dicom.is_experimental_dicom:
            return dicom.getname()
        else:
            name = dicom.getname()
            pattern = re.compile(r'(\d\d\d\d\d\d)')
            name_number = pattern.search(name)
            return name_number[0]

    if isinstance(dicomfilelist, list):
        dicomfile = dicomfilelist[0]
    else:
        dicomfile = dicomfilelist

    folderglobal = "rotation_analysis/slice_comparison/" + image_type
    filename = dicomfile.getname() + "_" + image_type + "_plot_fit.png"
    filenumber = get_dicom_number(dicomfile)
    fullpath = es.file_preparations(dicomfile, folderglobal)

    plt.clf()

    def normalize_coefflist(coeff_list):
        return (coeff_list - coeff_list.min()) / (coeff_list.max() - coeff_list.min())

    for dicomfile in [dicomfile]:
        color = pyplot_get_rand_color()
        # Get Rotation Coefficients
        coeff_array = normalize_coefflist(dicomfile.getsimilarity_coeff_list(image_type=image_type))
        params = df.sin_fitting(coeff_array)     # Fit sin wave through data
        if "normal" in image_type:
            dicomfile.similarty_func_coeff_normal = params
        else:
            dicomfile.similarty_func_coeff_filtered = params
        x_vals = range(len(coeff_array))
        y_vals = [df.sin_func(x, params[0], params[1], params[2], params[3]) for x in x_vals]
        if dicomfile.is_experimental_dicom:
            label = str(dicomfile.experimental_rot)
        else:
            label = filenumber
        plt.scatter(x_vals, coeff_array, label=label+" - raw data", s=10, marker=".", linestyle="None", color=color)
        plt.plot(x_vals, y_vals, label=label+" - fit", color=color)

    plt.axis([0, 360, 0, 1])
    plt.title("Rotation Analysis Visualization [" + filenumber + "]")
    plt.legend()
    plt.grid(True)
    plt.xlabel('Slice Rotation [Deg]')
    plt.ylabel('Relative Slice Overlap / Intensity')
    plt.grid(True)

    plt.savefig(fullpath + filename)
    debug = 1

    # Turn this on to get the plot
    # plt.show()


def pyplot_get_rand_color():
    possiblecolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    return possiblecolors[np.random.randint(len(possiblecolors))]