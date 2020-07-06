from copy import deepcopy as deepcopy
import imutils
import cv2
import numpy as np
from RaMResearch.Data import BasicDataStructs as bds
from RaMResearch.Data import DataStructs as ds
from itertools import combinations
from scipy.stats import norm
import matplotlib.pyplot as plt
from RaMResearch.Utils.General import BWtoRGB


# Store a single contour with a center point
class Contour:

    contour = None

    # Store the center of the contour as coordinate with (z, x, y)
    center: bds.Coordinate = None

    # Store the area of the contour
    area: float = None

    # Stored as (width, height)
    shape: (float, float) = None   # Store the angle of the contour
    tilt: float = None

    is_ring_contour: bool = False

    def __init__(self, contour, depth):
        self.clear_variables()
        self.contour = contour
        self.depth = depth
        self.calc_center()
        self.calc_area()
        self.calc_shape()
        self.calc_tilt()

    def clear_variables(self):
        self.contour = None
        self.center = None
        self.area = 0
        self.tilt = 0
        self.is_ring_contour = False

    def set_contour(self, contour):
        self.contour = contour

    def set_center(self, center: bds.Coordinate):
        self.center = center

    def get_center(self):
        return self.center

    def get_contour(self):
        return self.contour

    def calc_center(self):
        # Compute the center of the contour
        M = cv2.moments(self.contour)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])
        cZ = self.depth

        # Save the center of the contour
        self.center = bds.Coordinate(zyx_coord=[cZ, cX, cY])

    def calc_area(self):
        self.area = cv2.contourArea(self.contour)

    def calc_shape(self):
        _, _, w, h = cv2.boundingRect(self.contour)
        self.shape = w, h

    def get_area(self):
        return self.area

    def get_height(self):
        return self.shape[1]

    def get_width(self):
        return self.shape[0]

    def calc_tilt(self):
        [vx, vy, _, _] = cv2.fitLine(self.contour, cv2.DIST_L2, 0, 0.01, 0.01)
        self.tilt = -np.arctan(vy / vx) * 180/np.pi

    def set_ring_contour(self, is_ring_contour: bool):
        self.is_ring_contour = is_ring_contour

    def get_ring_contour(self):
        return self.is_ring_contour


class ContourConnection:

    # Store individual contours
    contour_1: Contour = None
    contour_2: Contour = None

    # Connection properties
    distance: float = 0
    angle: float = 0

    # Weights
    distance_weight: float = None   # Distance weight
    angle_weight: float = None      # Angle weight
    size_weight: float = None       # Size weight
    global_weight: float = None     # Overall weight

    # Image Dimensions
    image_size: (int, int) = None

    is_ring_cntr: bool = False

    def __init__(self, contour_1: Contour, contour_2: Contour, image_size):
        self.clear_variables()
        self.contour_1 = contour_1
        self.contour_2 = contour_2
        self.image_size = image_size
        self.calc_global_weight()

    def clear_variables(self):
        self.contour_1, self.contour_2 = None, None
        self.distance, self.angle = 0, 0
        self.distance_weight, self.angle_weight, self.size_weight = 0, 0, 0
        self.global_weight = 0
        self.image_size = (0, 0)
        self.is_ring_cntr = False

    def get_distance(self):
        return self.distance

    def get_angle(self):
        return self.angle

    def get_weight(self):
        return self.global_weight

    def calc_distance(self):
        x_dist = np.abs(self.contour_1.get_center().get_xpos() - self.contour_2.get_center().get_xpos())
        y_dist = np.abs(self.contour_1.get_center().get_ypos() - self.contour_2.get_center().get_ypos())
        self.distance = np.sqrt(x_dist ** 2 + y_dist ** 2)

    def calc_angle(self):
        x_dist = self.contour_1.get_center().get_xpos() - self.contour_2.get_center().get_xpos()
        y_dist = self.contour_1.get_center().get_ypos() - self.contour_2.get_center().get_ypos()
        x_dist = x_dist if x_dist != 0 else 0.00000001    # To avoid dividing by 0
        angle = np.arctan(y_dist / x_dist) * 180/np.pi
        return angle

    def calc_weights(self):

        # Define helper functions
        def diff(size): return np.clip(abs((size - 40) * 0.01), 0, 1)

        # Calc connection properties
        self.calc_distance()
        self.calc_angle()

        # Define weight variables
        angle_weight = 15  # The amount of degrees that the angle is allowed to differ
        distance_weight = 10  # The amount of pixels that the distance is allowed to diifer
        target_angle = self.get_target_angle()

        # Calc and store parameters
        self.distance_weight = np.clip(np.abs(55-self.distance)/distance_weight, 0, 1)
        self.angle_weight = np.clip(np.abs(target_angle-self.angle)/angle_weight, 0, 1)
        self.size_weight = np.clip(diff(self.contour_1.get_height())+diff(self.contour_2.get_height()), 0, 1)

        debug = 1

    def get_target_angle(self):
        x_dist = self.contour_1.get_center().get_xpos() if self.contour_1.get_center().get_xpos() > 0 else 0.00000001
        y_dist = int(self.image_size[1] / 2) - self.contour_1.get_center().get_ypos()
        return -np.arctan(y_dist / x_dist) * 180 / np.pi

    def calc_global_weight(self):
        self.calc_weights()
        self.global_weight = np.clip(1 - self.distance_weight - self.angle_weight - self.size_weight, 0, 1)

    def get_all_contours(self):
        return [self.contour_1, self.contour_2]

    def is_ring_contour(self):
        return self.is_ring_cntr

    def set_is_ring_contour(self, is_ring_contour):
        self.is_ring_cntr = is_ring_contour
        self.contour_1.is_ring_contour = is_ring_contour
        self.contour_2.is_ring_contour = is_ring_contour

    def get_midpoint(self):
        pos1 = self.contour_1.get_center().get_coordinates_int()
        pos2 = self.contour_2.get_center().get_coordinates_int()
        avg = np.add(pos1, pos2) / 2
        return [int(var) for var in avg]


# Store all slice contours of the dicom object
class SliceContourAnalysis:

    # Store slice information
    slice_depth: int = None
    slice_image: np.ndarray = None

    # Stores all contours in a list, sorted by x-Value
    all_contours: [Contour] = None

    # Stores all contour combinations and their likelyhood
    all_contour_combinations: [ContourConnection] = []

    ring_contour: ContourConnection = None
    is_main_ring_slc: bool = False

    # Initializes a new contour analysis
    def __init__(self, image: np.ndarray, slice_depth: int):
        self.clear_variables()
        self.slice_depth = slice_depth  # Set slice depth
        self.slice_image = image        # Set image for reference
        self.create_contours(image)     # Create contours on image
        self.calc_ring_contour()        # Calculate ring contour
        self.sort_contours()
        debug = 1

    def clear_variables(self):
        self.slice_depth = -1
        self.slice_image = np.empty((0, 0))
        self.all_contours = []
        self.all_contour_combinations = []
        self.ring_contour = None
        self.is_main_ring_slc = False

    # Method to create contours from image
    def create_contours(self, image):

        array = deepcopy(image)     # Create a copy of the image
        _, array = cv2.threshold(array, 128, 255, cv2.THRESH_BINARY)    # Threshold incoming array

        # Create and get contours
        cnts = cv2.findContours(array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        self.all_contours = [Contour(c, self.slice_depth) for c in cnts]

    def get_ring_contour(self):
        return self.ring_contour

    def calc_ring_contour(self):
        self.create_all_combinations()
        self.set_ring_contour()

    def create_all_combinations(self):
        self.all_contour_combinations = [ContourConnection(cntr_pair[0], cntr_pair[1], self.slice_image.shape)
                                         for cntr_pair in combinations(self.all_contours, 2)]

    def set_ring_contour(self):
        max_prob = 0    # Initialize to 0
        max_pos = 0     # Initializeinitialize to 0

        # Get highest weight
        for i in range(len(self.all_contour_combinations)):
            c_weight = self.all_contour_combinations[i].get_weight()
            if c_weight > max_prob:
                max_prob = c_weight
                max_pos = i

        if len(self.all_contour_combinations) > 0:
            self.ring_contour = self.all_contour_combinations[max_pos]
            self.ring_contour.set_is_ring_contour(True)

    def get_all_contours(self):
        return self.all_contour_combinations

    # Return image with all slices
    def get_image(self, all_contours=False):
        base_image = BWtoRGB(deepcopy(self.slice_image)).astype(np.uint8)
        contour_image = BWtoRGB(np.zeros(self.slice_image.shape)).astype(np.uint8)
        if all_contours:
            for contour in self.all_contours:
                cv2.drawContours(contour_image, contour.contour, -1, 255, 2, cv2.LINE_AA)
        else:
            contours = np.array([cntr.contour for cntr in [self.ring_contour[0], self.ring_contour[1]]])
            cv2.drawContours(contour_image, contours, -1, 2, cv2.LINE_AA)
        return cv2.add(base_image, contour_image)

    # Sorts the contours first by x-Value and then by y-Value
    def sort_contours(self):
        # Sort X
        self.all_contours.sort(key=lambda x: x.get_center().get_xpos(), reverse=False)

    def is_main_ring_contour(self):
        return self.is_main_ring_slc

    def set_is_main_ring_contour(self, is_main_ring_slc):
        self.is_main_ring_slc = is_main_ring_slc


# Store complete analysis in contour analysis object
class ContourAnalysis:

    image_3d: np.ndarray = None
    contour_results: [SliceContourAnalysis] = []

    # Most likely ring contour sotred as front contour, rear contour
    ring_contour: ContourConnection = None
    ring_contour_certainty: float = 0

    def __init__(self, image, debug=False):

        # Load in image
        if isinstance(image, ds.Image):
            self.image_3d = deepcopy(image.get_image(filtered=True))
        else:
            self.image_3d = deepcopy(image)

        # Run analysis
        self.calc_contour(debug=debug)         # Get all contours
        self.calc_ring_contour(debug=debug)    # Determine ring contour

    def get_contour_results(self, depth=None):

        # If the contours havent been computed yet
        if self.contour_results is None:
            self.calc_contour()

        # Return all contour results if depth is none, otherwise return the corresponding depth
        if depth is None:
            return self.contour_results
        else:
            return self.contour_results[depth]

    def calc_contour(self, debug=False):
        self.contour_results = [SliceContourAnalysis(self.image_3d[i, :], i) for i in range(self.image_3d.shape[0])]

    # Return ring_contours object if available, else calculate it first
    def get_ring_contour(self):
        if self.ring_contour is None:
            self.calc_ring_contour()
        return self.ring_contour

    def calc_ring_contour(self, debug=False):

        # Store all contours from all layers
        evaluation_array = [(s_cntr.get_ring_contour(), s_cntr.slice_depth) for s_cntr in self.contour_results]

        # Weight Depth
        gauss_dist = get_gauss_dist_norm(len(evaluation_array), plot=debug)
        weighted_depth_array = [c_conn.get_weight() * gauss_dist[depth] for c_conn, depth in evaluation_array
                                if c_conn is not None]

        max_pos = int(np.argmax(weighted_depth_array))

        # Store most likely ring contour and its certainty
        self.ring_contour = evaluation_array[max_pos][0]
        self.contour_results[max_pos].set_is_main_ring_contour(True)

    def calc_connected_ring_contours(self):
        pass

    # Returns an image with the contour connections
    # If the with_contours flag is set, the contours will be drawn as well
    def get_image(self, with_connections=False, with_contours=False, with_angle=False,
                  with_area=False, with_color=False, with_height=False, with_midpoint=False, threshold=0):

        def draw_contour_connection(conn, image, is_main_slice=False):
            contour_general_col, contour_ring_col = (0, 0, 255), (0, 255, 0)
            # Switch color based on if a contour is a ring connection
            if with_color:
                color = contour_ring_col if conn.is_ring_contour() and is_main_slice else contour_general_col
            else:
                wght = int(conn.get_weight() * 255)
                color = (wght, wght, wght)

            cntr1, cntr2 = conn.get_all_contours()
            depth = cntr1.get_center().get_zpos()

            if with_connections:
                # Draw Connections
                pt1 = cntr1.get_center().get_xpos(), cntr1.get_center().get_ypos()
                pt2 = cntr2.get_center().get_xpos(), cntr2.get_center().get_ypos()
                cv2.line(image[depth, :], pt1, pt2, color=color, thickness=1)

            if with_contours:
                cv2.drawContours(image[depth, :], [cntr1.contour], -1, color, 2, cv2.LINE_AA)
                cv2.drawContours(image[depth, :], [cntr2.contour], -1, color, 2, cv2.LINE_AA)

            if with_angle:
                # Add angle description
                angle = str(c_conn.get_angle())
                c1_coords = cntr1.center.get_coordinates_int()
                c2_coords = cntr2.center.get_coordinates_int()
                mid_point = (int((c1_coords[1] + c2_coords[1]) / 2), int((c1_coords[2] + c2_coords[2]) / 2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image[depth, :], angle, mid_point, font, 0.4, color=color)

            if with_area or with_height:
                for contour in [cntr1, cntr2]:
                    text = str(contour.get_area()) if with_area else str(contour.get_height())
                    center_point = (contour.get_center().get_xpos() + 10, contour.get_center().get_ypos())
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image[depth, :], text, center_point, font, 0.4, color=color)

        image_size = self.image_3d.shape

        # If the contour option is set create an RGB image, otherwise a bw image
        return_image = np.zeros(image_size)
        
        if with_color:
            return_image = BWtoRGB(return_image)

        # Draw regular contours
        all_c = []
        for slc in self.contour_results:
            all_c += slc.get_all_contours()
        for c_conn in all_c:
            if c_conn.get_weight() >= threshold:
                draw_contour_connection(c_conn, return_image)

        # Draw ring contour
        draw_contour_connection(self.ring_contour, return_image, is_main_slice=True)

        # Draw Midpoint
        if with_midpoint:
            mid = self.ring_contour.get_midpoint()
            cv2.circle(return_image[mid[0], :], tuple(mid[1:]), 5, 255, thickness=-1)

        # Return final image
        return return_image


def get_gauss_dist_norm(layers, plot=False):

    mu = int(layers / 2)        # Centerpoint of the curve
    variance = int(layers/10)  # Original Value layers / 3
    sigma = np.sqrt(variance)
    x_vals = np.arange(0, layers, 1)
    y_vals = norm.pdf(x_vals, mu, sigma)
    y_vals *= 1 / np.max(y_vals)

    # Test plot
    if plot:
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)

        ax.set(xlabel='layernum', ylabel='Weight',
               title='Z-Depth Weight Plot')
        ax.grid()
        plt.show()

    return y_vals





# HELPER FUNCTIONS =================================

# def get_contour_angle(contour_1: Contour, contour_2: Contour):
#     x_dist = contour_1.center.xpos - contour_2.center.xpos
#     y_dist = contour_1.center.ypos - contour_2.center.ypos
#     x_dist = x_dist if x_dist != 0 else 0.00000001    # To avoid dividing by 0
#     angle = np.arctan(y_dist / x_dist) * 180/np.pi
#     return angle


# def get_contour_height_diff(contour_1: Contour, contour_2: Contour):
#     def diff(size):
#         return np.clip(abs((size-40)*0.01), 0, 1)
#     height1, height2 = contour_1.get_height(), contour_2.get_height()
#     weight =
#     return weight


# def get_target_angle(contour, slice_size_y):
#     x_dist = contour.center.xpos if contour.center.xpos > 0 else 0.00000001
#     y_dist = int(slice_size_y/2) - contour.center.ypos
#     return -np.arctan(y_dist / x_dist) * 180/np.pi


# def get_connection_weight_slice(distance_diff, angle_dif, size_diff):
#     angle_weight = 15  # The amount of degrees that the angle is allowed to differ
#     distance_weight = 10  # The amount of pixels that the distance is allowed to diifer
#     weight = np.clip(1 - np.abs(distance_diff/distance_weight) - np.abs(angle_dif/angle_weight) - size_diff, 0, 1)
#     return weight