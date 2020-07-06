import tifffile
import pydicom as pd
import cv2
import numpy as np
import os
from scipy import ndimage, spatial
from BScThesis import PatternMaker as pm, HelperFunctions as hf, DicomImage as di, ImageFilters as imfil
from itertools import combinations


# Dicom Class to store all the data together in one object
class Dicomimage:

    name = ""                 # No file ending
    patient = None
    metadata = None

    # Arrays
    image: di.StandardImage = None                # Original Array
    filtered_image: di.FilteredImage = None        # Filtered Arrays

    # Coordinates
    ringcoordinates = None      # In Pixels
    ringdiameter = None         # In mm
    closestedge_point = None

    # Dicom Data
    slice_thickness = 0.345

    # Correlation Variables
    pattern = None
    correlated_array = None

    # For Experimental Images:
    is_experimental_dicom = False
    experimental_rot = None
    experimental_depth = None

    # Countours
    contours = []
    contour_pairs = []
    contour_images = []
    contour_images_normal = []
    contour_ring_loc = None
    contour_thresh = 0

    # Contour based rotation array
    contour_ring_top_view_filtered = None
    contour_ring_top_view_normal = None
    rot_ellipse = None
    rotation_export = None
    determined_angle = 0
    correlation_angle_array = None

    # Rotation analysis
    similarty_func_coeff_normal = None
    similarity_coeff_array_normal = []
    similarty_func_coeff_filtered = None
    similarity_coeff_array_filtered = []

    # Slices
    slice_rot_list_export_normal = []
    slice_rot_list_export_filter = []

    def __init__(self, name):
        self.reset_parameters()
        self.setname(name)

    def reset_parameters(self):

        self.patient = None
        self.metadata = None

        self.image: di.StandardImage = None  # Original Array
        self.filtered_image: di.FilteredImage = None
        self.ringcoordinates = None  # In Pixels
        self.ringdiameter = None  # In mm
        self.closestedge_point = None
        self.slice_thickness = 0.345
        self.pattern = None
        self.correlated_array = None

        # For Experimental Images:
        self.is_experimental_dicom = False
        self.experimental_rot = None
        self.experimental_depth = None

        # Countours
        self.contours = []
        self.contour_pairs = []
        self.contour_images = []
        self.contour_images_normal = []
        self.contour_ring_loc = None
        self.contour_thresh = 0

        # Contour based rotation array
        self.contour_ring_top_view_filtered = None
        self.contour_ring_top_view_normal = None
        self.rot_ellipse = None
        self.rotation_export = None
        self.determined_angle = 0
        self.correlation_angle_array = None

        # Rotation analysis
        self.similarty_func_coeff_normal = None
        self.similarty_func_coeff_filtered = None
        self.similarity_coeff_array_normal = []
        self.similarity_coeff_array_filtered = []

        # Slices
        self.slice_rot_list_export_normal = []
        self.slice_rot_list_export_filter = []

    def set_image(self, image_array, image_type="normal"):

        if "normal" in image_type.lower():
            if self.image is None:
                self.image = di.StandardImage(image_array)
            else:
                self.image.set_array(image_array)
        else:
            if self.filtered_image is None:
                self.filtered_image = di.FilteredImage(image_array)
            else:
                self.filtered_image.set_array(image_array)

    def get_image(self, image_type="normal"):
        if "normal" in image_type.lower() or "base" in image_type.lower():
            return self.image
        else:
            return self.filtered_image

    def get_all_arrays(self):
        # Loop through self and return the arrays that are dicom images and not none
        return [getattr(self, a).get_array() for a in dir(self) if isinstance(getattr(self, a), di.DicomImage)]

    def setmetadata(self, metadata):
        self.metadata = metadata

    def setname(self, name):
        splitname = name.split('/')
        self.name = splitname[-1]

    def setringcoordinates(self, coordinates):
        self.ringcoordinates = coordinates

    def setringdiameter(self, diameter):
        self.ringdiameter = diameter

    def setpatient(self, patient):
        self.patient = patient

    def createsearchpattern(self):
        self.pattern = pm.circlePatternImage(12, self.slice_thickness, 6)

    def getname(self):
        return self.name

    def getmetadata(self):
        return self.metadata

    def getringcoordinates(self):
        return self.ringcoordinates

    def getringdiameter(self):
        return self.ringdiameter

    def getpatient(self):
        return self.patient

    def getcontourconnections(self, slice_depth):
        return [contour for contour in self.contour_pairs if contour[1] == slice_depth]

    def getclosestedgepoint(self):
        return self.closestedge_point

    def getsimilarity_coeff_list(self, image_type="normal"):
        if "normal" in image_type.lower():
            return self.similarity_coeff_array_normal
        else:
            return self.similarity_coeff_array_filtered

    def create_ringfocusarray(self, size=50, image_type="normal"):

        # Get desired array
        image = self.get_image(image_type)
        array = image.get_array()

        # Define padding tuple and create padded array
        totpad = ((size, size), (0, 0), (size, size))
        pdarray = np.pad(array, totpad, mode='constant')

        # Calculate new centerpoint if needed
        if self.closestedge_point is None:
            self.setclosestringpoint()

        # Offset position to account for padding
        rp = self.closestedge_point
        cp = (rp[0] + size, rp[2], rp[1] + size)

        # Return section of original array
        returnarray = pdarray[cp[0] - size:cp[0] + size, cp[1] - size:cp[1] + size,
                      cp[2]:cp[2] + int(size / 2)]

        image.set_array(returnarray.transpose(2, 1, 0), array_type="ring-focus")
        image.focus_array_dim = returnarray.shape
        image.focus_array_pos = self.closestedge_point

    def createrotatedarrays(self, rot_values=(0, 360), image_type="normal"):

        image = self.get_image(image_type=image_type)
        array = image.get_array(array_type="focus")
        dim = array.shape

        # Create Returnarray
        returnarray = np.zeros([rot_values[1], dim[0], dim[1], dim[2]])

        # Loop through angles using nearest neighbor interpolation
        for i in range(rot_values[0], rot_values[1]):
            angle = int(i - rot_values[0])
            for layer in range(dim[0]):
                returnarray[i, layer, :] = ndimage.interpolation.rotate(array[layer], angle, (1, 0),
                                                                        reshape=False, order=0)

        image.set_array(returnarray.astype(np.uint8), array_type="focus_rotated_array")
        image.set_rotation_params(rot_value=rot_values)

    def setclosestringpoint(self):
        rc = self.ringcoordinates
        searchlayer = self.get_image("filtered").get_array()[rc[0], rc[2], 0:rc[1]]
        xmax = np.max(searchlayer)
        sl_length = len(searchlayer)

        # Get better Z Value by checking ther sum of the layers above and under

        layers_to_be_checked = 10
        array_sum = np.zeros(layers_to_be_checked * 2)

        size = 80
        # Section / slice around the point to be calculated
        ss = [rc[2] - size, rc[2] + size, rc[1] - size, rc[1] + size]

        for zdepth in range(-layers_to_be_checked, layers_to_be_checked):
            debug = 1
            array_sum[zdepth + 10] = np.sum(
                self.filtered_image.get_array()[rc[0] + zdepth, ss[0]:ss[1], ss[2]:ss[3]])

        max_pos = np.argmax(array_sum)
        print("Whole array: ", array_sum, "\t Max element position: ", max_pos)

        # Update ring position
        urp = np.add(rc, [max_pos - 10, 0, 0])
        self.ringcoordinates = urp

        # Get X Value
        prev_val = 0

        for ix in range(1, sl_length):
            cur_val = searchlayer[sl_length-ix]
            if prev_val > xmax/2 > cur_val:
                self.closestedge_point = [urp[0], sl_length-ix, urp[2]]
                break
            else:
                prev_val = cur_val

        # TODO create check
        # if self.closestedge_point is None:

    def makecountours(self):
        pass

    def drawcountours(self):
        pass

    def calcsimilaritycoefficient(self, image_type="normal"):

        image = self.get_image(image_type=image_type)
        rotarray = image.get_array(array_type="focus_rotated_array")
        coefficient_matrix = np.zeros(rotarray.shape[0])
        threshvalue = 125

        for rotvalue in range(rotarray.shape[0]):
            crosscuts = image.getcrosscut(array_type="focus_rotated_array", rot_array_angle=rotvalue)
            # Get arrays
            multiplied_array_1 = crosscuts[0].astype(np.uint16)
            multiplied_array_2 = crosscuts[1].astype(np.uint16)

            # Threshold
            if "normal" in image_type.lower():
                multiplied_array_1[multiplied_array_1 > np.max(multiplied_array_1)/2] = 0
                multiplied_array_2[multiplied_array_2 > np.max(multiplied_array_2)/2] = 0

            # Multiply
            multiplied_array = np.divide(np.sum(multiplied_array_1), np.sum(multiplied_array_2))
            coefficient_matrix[rotvalue] = np.sum(multiplied_array)

        if "normal" in image_type.lower():
            self.similarity_coeff_array_normal = coefficient_matrix
        else:
            self.similarity_coeff_array_filtered = coefficient_matrix

    def create_contours(self):
        fil_array = self.get_image("filtered").get_array()

        for slice_pos in range(fil_array.shape[0]):
            slice_contours = hf.cvcontour(fil_array[slice_pos], slice_pos)
            for contour in slice_contours:
                self.contours.append(contour)

    def create_contour_pairs(self, min_thresh: float = 0):
        dicomsize = self.get_image().get_array().shape
        for slice_depth in range(self.get_image("filtered").get_array("base").shape[0]):
            centerpoints = [(x[0][0], x[0][1]) for x in self.contours if x[0][2] == slice_depth]
            if len(centerpoints) > 1:
                contourpairs = combinations(centerpoints, 2)
                connections = [(x, slice_depth, get_point_distances(x, slice_depth)) for x in contourpairs]
                # list of (pair, depth, [x_dist, y_dist, spatial_distance, angle, depth])
                for (pair, slice_dep, properties) in connections:
                    if get_connection_weight(properties, dicomsize) >= min_thresh:
                        self.contour_pairs.append((pair, slice_dep, get_connection_weight(properties, dicomsize)))

    def get_max_contour(self):
        max_contourpair = self.contour_pairs[0]
        for (contour_pair, cnt_depth, weight) in self.contour_pairs:
            if weight > max_contourpair[2]:
                max_contourpair = (contour_pair, cnt_depth, weight)
        self.contour_ring_loc = max_contourpair
        return max_contourpair

    def color_max_contour(self):
        (cnt_pair, cnt_depth, weight) = self.get_max_contour()
        for cnt in cnt_pair:
            for contour in self.contours:
                if contour[0] == (cnt[0], cnt[1], cnt_depth):
                    image = [cnt_image[0] for cnt_image in self.contour_images if cnt_image[1] == cnt_depth][0]

                    if len(self.contour_images_normal) == 0:
                        normalimage_copy = self.get_image("normal").get_array("base")[cnt_depth].copy()
                        normalimage_copy = cv2.cvtColor(normalimage_copy, cv2.COLOR_GRAY2BGR)
                    else:
                        normalimage_copy = self.contour_images_normal[0][0]

                    cv2.drawContours(image, contour[1], -1, (0, 255, 0), thickness=4)
                    cv2.drawContours(normalimage_copy, contour[1], -1, (0, 255, 0), thickness=4)

                    if len(self.contour_images_normal) == 0:
                        self.contour_images_normal.append((normalimage_copy, cnt_depth))

    def set_contour_ringcoordinates(self):
        if self.contour_ring_loc is not None:
            (((x1, y1), (x2, y2)), z, _) = self.contour_ring_loc
            self.ringcoordinates = (z, int(np.mean([x1, x2])), int(np.mean([y1, y2])))

    def contour_based_closest_ring(self):

        def getarray_around_point(point, hsize, vsize):
            (x_loc, y_loc, z_loc) = point
            (dH, dV) = (int(hsize/2), int(vsize/2))
            return self.get_image("normal").get_array("base")[z_loc, x_loc-dH:x_loc+dH, y_loc-dV:y_loc+dV]

        (((x1, y1), (x2, y2)), z, _) = self.contour_ring_loc

        x, y = (x1-3, y1) if x1 < x2 else (x2-3, y2)
        zr, xr, yr = self.ringcoordinates

        brightness = []

        offset_range = (-12, 12)

        for offset in range(offset_range[0], offset_range[1]):
            cutout = getarray_around_point((x, y, z+offset), hsize=3, vsize=6)
            ringcenter = getarray_around_point((xr, yr, zr + offset), hsize=3, vsize=6)
            brightness.append(np.max(cutout-ringcenter))

        # TODO Make the x-offset dependent on the contour size
        cp = (z + (brightness.index(max(brightness)) + offset_range[0]), x-3, y)
        self.closestedge_point = cp

    def contour_based_closest_ring_v2(self):

        def get_close_contour(xpos, ypos, zpos, contourlist):
            distances = []
            for i in range(len(contourlist)):
                (xc, yc, zc), cnt = contourlist[i]
                if zc == zpos:
                    distances.append((i, np.linalg.norm((xc-xpos, yc - ypos))))
            mindist = (1, 1000)
            for (index, dist) in distances:
                mindist = (index, dist) if dist < mindist[1] else mindist
            if mindist[1] > 40:
                return False, None
            else:
                return True, contourlist[mindist[0]][1]

        def get_contour_area(cnt):
            return cv2.contourArea(cnt)

        (((x1t, y1t), (x2t, y2t)), z, _) = self.contour_ring_loc

        x1, y1 = (x1t, y1t) if x1t < x2t else (x2t, y2t)

        cnt_list = self.contours
        size_array = []

        # Check layers above and below
        offset_range = (-14, 14)
        for offset in range(offset_range[0], offset_range[1]):
            z_off = z + offset
            found1, contour1 = get_close_contour(x1t, y1t, z_off, cnt_list)
            found2, contour2 = get_close_contour(x2t, y2t, z_off, cnt_list)
            if found1 and found2:
                size_array.append(get_contour_area(contour1) * get_contour_area(contour2))
            else:
                size_array.append(0)

        cp = (z + (size_array.index(max(size_array)) + offset_range[0]), x1 - 3, y1)
        self.closestedge_point = cp

    def filter_ringfocusarray(self, image_type):
        array = self.get_image(image_type=image_type).get_array(array_type="ring_focus")
        self.contour_ring_top_view_normal = array.copy()
        self.contour_ring_top_view_filtered = imfil.gauslogfilter(array, 7, 5, gaus1D=False, morphological=False)

    def rotation_by_convolution(self):

        image = self.contour_ring_top_view_filtered
        ellipse_max_size = int(np.max([image.shape[1], image.shape[2]])/1.5)
        ellipse_size = (int(ellipse_max_size / 4), ellipse_max_size)

        angles = (-90, 90)
        ellipse_rot = pm.create_rotating_ellipse(ellipse_size[0], ellipse_size[1], start_angle=angles[0],
                                                 stop_angle=angles[1], outline=False, rgb=False)
        self.rot_ellipse = ellipse_rot
        convolution_results = []

        for rotation in range(len(ellipse_rot)):
            convolutionresult = hf.findPatternInImageTemplate(image=image, pattern=ellipse_rot[rotation])
            # max_val = np.max(convolutionresult)
            convolution_results.append(hf.getminmaxloc(convolutionresult))

        # z_depth = image.shape[0]
        # for i in range(len(convolution_results)):
        #     value, (z, x, y) = convolution_results[i]
        #     value = int(value * (1 - ((7 - z)/7)))
        #     convolution_results[i] = value, (z, x, y)

        # Store all of the angle information
        self.correlation_angle_array = (angles, [conv_res[0] for conv_res in convolution_results])

        # maxvalue = 0
        # maxpos = 0
        # for i in range(len(convolution_results)):
        #     if convolution_results[i][0] >= maxvalue:
        #         maxvalue = convolution_results[i][0]
        #         maxpos = i

        maxvalue = 0
        maxpos = 0
        for i in range(len(convolution_results)):
            if convolution_results[i][0] >= maxvalue:
                maxvalue = convolution_results[i][0]
                maxpos = i

        self.determined_angle = (maxpos + angles[0], convolution_results[maxpos])

        self.rotation_export = (self.contour_ring_top_view_filtered, self.contour_ring_top_view_normal)

    def create_slices_for_export(self, image_type="normal"):
        image = self.get_image(image_type)
        for i in range(0, 91):
            (crscts_h, crscts_v)  = image.getcrosscut(array_type="focus_rot", rot_array_angle=i)
            if "normal" in image_type.lower():
                self.slice_rot_list_export_normal.append((i, crscts_h.copy(), crscts_v.copy()))
            else:
                self.slice_rot_list_export_filter.append((i, crscts_h.copy(), crscts_v.copy()))
        ringpos = self.ringcoordinates
        ringcrscut = image.get_array()[ringpos[0], ringpos[2], :]
        image.set_array(ringcrscut, "ring_crosscut")


def getDicomFiles(foldername):
    returnlist = []
    for file in os.listdir(foldername):
        if file.endswith(".dcm"):
            returnlist.append(os.path.splitext(file)[0])
    return returnlist


def importdicomfile(dicomfilename, experiment=False):

    if experiment:
        dicomfile = import_experimental_dicom(dicomfilename)
    else:
        dicomfile = import_standard_dicom(dicomfilename)

    print("Analizing: " + dicomfile.getname())
    return dicomfile


def import_standard_dicom(dicomfilename):

    # Initialize Dicom Object
    dicomobject = Dicomimage(dicomfilename)

    # Import and add image data
    imagename = dicomfilename + ".tif"
    imagearray = tifffile.imread(imagename)
    dicomobject.set_image(imagearray, image_type="normal")

    # Import and add dicom header
    dicomheadername = dicomfilename + '.dcm'
    dicommetadata = pd.dcmread(dicomheadername)
    dicomobject.setmetadata(dicommetadata)

    # Return the dicom object
    return dicomobject


def import_experimental_dicom(dicomfilename):

    # Create dicomfile object
    dicomobject = Dicomimage(dicomfilename)

    # Import and set the array
    dicomheadername = dicomfilename + '.dcm'
    dicomfile = pd.dcmread(dicomheadername)
    dicomobject.set_image(dicomfile.pixel_array)

    # Set the dicomfile as metadata file
    dicomobject.setmetadata(dicomfile)

    # Try to set the slice thickness if it exists
    try:
        slicethickness = dicomfile.SliceThickness
        dicomobject.slice_thickness = slicethickness
    except:
        dicomobject.slice_thickness = 0.345

    # Set the experimentally recorded rotation and depth in the object
    rotation = tuple(dicomfile.PatientName.family_name.split(","))

    # TODO implement if depth is recorded
    depth = dicomfile.PatientName.given_name.replace(",", ".")

    if len(depth) > 0:
        ring_depth = float(depth)
    else:
        ring_depth = 10.1

    dicomobject.is_experimental_dicom = True
    dicomobject.experimental_rot = rotation
    dicomobject.experimental_depth = ring_depth

    # Return the newly created object
    return dicomobject


def importdicomfiles(dicomnamelist):

    # Initialize Arrays
    dicomlist = []

    # Iterate over given files
    for i in range(len(dicomnamelist)):
        # Image name
        filename = dicomnamelist[i]

        # Initialise dicom object
        dicomobject = Dicomimage(filename)

        # Import and add image data
        imagename = filename + ".tif"
        imagearray = tifffile.imread(imagename)
        dicomobject.setarray(imagearray)

        # Import and add dicom header
        dicomheadername = filename + '.dcm'
        dicommetadata = pd.dcmread(dicomheadername)
        dicomobject.setmetadata(dicommetadata)

        # Append dicom object to return array
        dicomlist.append(dicomobject)

    # Return dicomimage object
    return dicomlist


# Global variables for viewers
rot = 0
depth = 0
rot_offset = (0, 0)


def viewer3d(array, windowName):

    def trackbarHandler(value):
        global depth
        print("Depth = ", value)
        depth = value
        cv2.imshow(windowName, array[depth, :])

    cv2.namedWindow(windowName)
    cv2.createTrackbar('Depth', windowName, 0, array.shape[0] - 1, trackbarHandler)
    cv2.imshow(windowName, array[0, :])
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()


def viewer3d_wRot(dicomfile, image_type="normal", windowName="3D Rotational Viewer"):

    image = dicomfile.get_image(image_type=image_type)
    array4D = image.get_array(array_type="focus_rotated_array")

    global rot_offset
    rot_offset = image.get_rotation_params()

    global rot
    rot = rot_offset[0]

    global depth
    depth = 0

    # Callback function for depth trackbar change
    def depthtrackbarHandler(value):
        global depth
        depth = value
        print("Depth = ", value, "\t Rotation = ", rot - rot_offset[0])
        cv2.imshow(windowName, array4D[rot, depth, :])

    # Callback function for rotation trackbar change
    def rottrackbarHandler(value):
        global rot
        rot = value
        print("Depth = ", depth, "\t Rotation = ", value - rot_offset[0])
        cv2.imshow(windowName, array4D[rot, depth, :])

    # Create Windows and Trackbars
    cv2.namedWindow(windowName)
    cv2.createTrackbar('Depth', windowName, 0, array4D.shape[1] - 1, depthtrackbarHandler)
    cv2.createTrackbar('Rotation', windowName, 0, array4D.shape[0] - 1, rottrackbarHandler)
    cv2.imshow(windowName, array4D[rot, depth, :])

    # Wait for user to exit program
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    # Close the windows that were opened
    cv2.destroyAllWindows()


def sliceVisualizer(dicomfile, image_type="normal", windowName="Slice Visualizer"):

    image = dicomfile.get_image(image_type=image_type)
    array4D = image.get_array(array_type="focus_rotated_array")

    global rot_offset
    rot_offset = image.get_rotation_params()

    global rot
    rot = rot_offset[0]

    array4D_dim = array4D[0].shape
    hor_height = int(array4D_dim[2] / 2)
    ver_height = int(array4D_dim[1] / 2)

    def rottrackbarHandler(value):
        global rot
        rot = value
        print("Rotation = ", value - rot_offset[0])
        sliceHorizontal = np.transpose(array4D[rot, :, :, hor_height], axes=(1, 0))
        sliceVertical = np.transpose(array4D[rot, :, ver_height, :], axes=(1, 0))
        showArray = np.append(sliceVertical, sliceHorizontal, axis=1)
        cv2.imshow(windowName, showArray)

    cv2.namedWindow(windowName)
    cv2.createTrackbar('Rotation', windowName, 0, array4D.shape[0] - 1, rottrackbarHandler)
    firstsliceHorizontal = np.rot90(array4D[rot, :, :, hor_height])
    firstsliceVertical = np.rot90(array4D[rot, :, ver_height, :])
    firstshowArray = np.append(firstsliceVertical, firstsliceHorizontal, axis=1)
    cv2.imshow(windowName, firstshowArray)

    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()


def imageview3d(arrays, windowName="TestWindow"):
    if isinstance(arrays, list):
        if len(arrays) == 1:
            viewer3d(arrays[0], windowName)
        else:
            showarray = arrays[0]
            for i in range(1, len(arrays)):
                showarray = np.append(showarray, arrays[1], axis=2)
            viewer3d(showarray, windowName)
    else:
        viewer3d(arrays, windowName)


def get_point_distances(pos, z_depth):
    x_dist = np.abs(pos[1][0] - pos[0][0]) + 0.00000001  # To avoid dividing by 0
    y_dist = np.abs(pos[1][1] - pos[0][1])
    spatial_distance = spatial.distance.euclidean(pos[1], pos[0])
    angle = np.arctan(y_dist / x_dist) * 180/np.pi
    return [x_dist, y_dist, spatial_distance, angle, z_depth]


# Returns a number between 0 and 1, with 0 being no chance of ring crossection, and 1 being very sure
def get_connection_weight(connection, dicom_dim, target_angle=0, target_length=55):
    [_, _, length, angle, z_depth] = connection
    length_dif = np.abs(length - target_length) / 60
    angle_dif = np.abs(angle - target_angle) / 20
    depth_dif = np.abs(z_depth - dicom_dim[0]/2) / 70
    return 1 - np.clip(angle_dif + length_dif + depth_dif, 0, 1)
