from copy import deepcopy as deepcopy
import imutils
import cv2
import numpy as np
from RaMResearch.Data import BasicDataStructs as bds
from RaMResearch.Data import DataStructs as ds
from itertools import combinations
from scipy.stats import norm
import matplotlib.pyplot as plt
from RaMResearch.Utils.General import BWtoRGB, num_clip, print_min_max, print_divider


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

    def dist_to_point(self, point):
        # Calc distance on 2d slice
        if len(point) == 2:
            _, x1, y1 = self.center.get_coordinates()
            x2, y2 = point
            return np.sqrt((x1 - x2)**2), np.sqrt((y1 - y2)**2)
        # Otherwise calculate 3d distance
        elif len(point) == 3:
            z1, x1, y1 = self.center.get_coordinates()
            z2, x2, y2 = point
            return np.sqrt((z1 - z2) ** 2), np.sqrt((x1 - x2) ** 2), np.sqrt((y1 - y2) ** 2)
        else:
            raise ValueError("Point has to have 2 or 3 dimensions")


class ContourConnection:

    contours: [Contour] = []

    # Connection properties
    distance: float = 0
    target_distance: float = 0
    angle: float = 0
    target_angle: float = 0

    # Weights
    distance_weight: float = None   # Distance weight
    angle_weight: float = None      # Angle weight
    size_weight: float = None       # Size weight
    global_weight: float = None     # Overall weight

    # Image Dimensions
    image_size: (int, int) = None

    is_ring_cntr: bool = False

    def __init__(self, contour_1: Contour, contour_2: Contour, image_size):
        # Clear variables
        self.clear_variables()

        # Set Values
        self.image_size = image_size
        self.target_distance = 55.0
        self.contours.append(contour_1)
        self.contours.append(contour_2)

        # Start calculations
        self.organize_contours()
        self.calc_contour_angle()
        self.calc_target_angle()
        self.calc_distance()
        self.calc_global_weight()

    def organize_contours(self):
        center_point = [0, int(self.image_size[1] / 2)]
        dist1, dist2 = self.contours[0].dist_to_point(center_point), self.contours[1].dist_to_point(center_point)
        if dist2 < dist1:
            self.contours.reverse()

    def clear_variables(self):
        self.contours = []
        self.distance, self.angle = 0, 0
        self.distance_weight, self.angle_weight, self.size_weight = 0, 0, 0
        self.global_weight = 0
        self.target_angle = 0
        self.image_size = (0, 0)
        self.is_ring_cntr = False

    def get_distance(self):
        return self.distance

    def get_target_distance(self):
        return self.target_distance

    def get_angle(self):
        return self.angle

    def get_target_angle(self):
        return self.target_angle

    def get_weight(self):
        return self.global_weight

    def calc_distance(self):
        x_dist = np.abs(self.contours[0].get_center().get_xpos() - self.contours[1].get_center().get_xpos())
        y_dist = np.abs(self.contours[0].get_center().get_ypos() - self.contours[1].get_center().get_ypos())
        self.distance = np.sqrt(x_dist ** 2 + y_dist ** 2)

    @staticmethod
    def calc_angle(point1, point2):
        diff = np.subtract(point1, point2).astype(float)
        if diff[0] == 0:
            pause = 1
        diff[0] = diff[0] if diff[0] != 0 else 0.00000001
        return np.arctan(diff[1] / diff[0]) * 180 / np.pi

    def calc_contour_angle(self):
        pos1 = self.contours[0].get_center().get_coordinates()[1:]
        pos2 = self.contours[1].get_center().get_coordinates()[1:]
        self.angle = self.calc_angle(pos1, pos2)

    def calc_target_angle(self):
        image_pt = [0, int(self.image_size[1] / 2)]
        cntr1_pt = self.contours[0].get_center().get_coordinates()[1:]
        self.target_angle = self.calc_angle(image_pt, cntr1_pt)

    def calc_weights(self):

        # Define helper functions
        def diff(size): return np.clip(abs((size - 40) * 0.01), 0, 1)

        # Define weight variables
        angle_weight: int = 15  # The amount of degrees that the angle is allowed to differ
        distance_weight = 10  # The amount of pixels that the distance is allowed to diifer

        # Calc and store parameters
        self.distance_weight = num_clip(np.abs(55-self.distance)/distance_weight, 0, 1)
        self.angle_weight = num_clip(np.divide(np.abs(self.target_angle-self.angle), angle_weight), 0, 1)
        self.size_weight = num_clip(diff(self.contours[0].get_height())+diff(self.contours[1].get_height()), 0, 1)

    def calc_global_weight(self):
        self.calc_weights()
        self.global_weight = num_clip(1 - self.distance_weight - self.angle_weight - self.size_weight, 0, 1)

    def get_all_contours(self):
        return self.contours

    def is_ring_contour(self):
        return self.is_ring_cntr

    def set_is_ring_contour(self, is_ring_contour):
        self.is_ring_cntr = is_ring_contour
        self.contours[0].is_ring_contour = is_ring_contour
        self.contours[1].is_ring_contour = is_ring_contour

    def get_midpoint(self):
        pos1, pos2 = [x.get_center().get_coordinates() for x in self.contours]
        avg = np.add(pos1, pos2) / 2
        return [int(var) for var in avg]

    def debug(self):
        print("Connection Weight:    " + "\t\t" + str(self.get_weight()))
        print("Contour Length Actual:" + "\t\t" + str(self.get_distance()))
        print("Contour Length Target:" + "\t\t" + str(self.get_target_distance()))
        print("Contour Angle Actual: " + "\t\t" + str(self.get_angle()))
        print("Contour Angle Target: " + "\t\t" + str(self.get_target_angle()))


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

    def clear_variables(self):
        self.slice_depth = -1
        self.slice_image = np.empty((0, 0))
        self.all_contours = []
        self.all_contour_combinations = []
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
            contours = np.array([cntr.contour for cntr in self.ring_contour.get_all_contours()])
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

    def debug(self):
        self.ring_contour.debug()


# Store complete analysis in contour analysis object
class ContourAnalysis:

    image_3d: np.ndarray = None
    contour_results: [SliceContourAnalysis] = []

    # Most likely ring contour sotred as front contour, rear contour
    ring_dim: tuple = ()
    ring_contour: ContourConnection = None
    ring_contour_certainty: float = 0

    # Temporary values
    slice_weight: [float] = np.empty(360)

    def __init__(self, image, ring_dim: tuple, debug=False):

        print_divider("Contour Analysis")

        self.ring_dim = ring_dim

        # Load in image
        if isinstance(image, ds.Image):
            self.image_3d = deepcopy(image.get_image(filtered=True))
        else:
            self.image_3d = deepcopy(image)

        # Run analysis
        self.calc_contour()         # Get all contours
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

    def calc_contour(self):
        self.contour_results = [SliceContourAnalysis(self.image_3d[i, :], i) for i in range(self.image_3d.shape[0])]

    # Return ring_contours object if available, else calculate it first
    def get_ring_contour(self):
        if self.ring_contour is None:
            self.calc_ring_contour()
        return self.ring_contour

    def calc_ring_contour(self, debug=False):

        # Create weights for each slice
        num_slices = len(self.contour_results)
        self.slice_weight = get_gauss_dist_norm(num_slices, plot=False, width=1)

        # Store
        evaluation_array = np.empty(num_slices)

        # Store all contours from all layers
        for i in range(num_slices):
            cntr = self.contour_results[i].get_ring_contour()
            evaluation_array[i] = cntr.get_weight() * self.slice_weight[i] if cntr is not None else 0

        if debug:
            print("Contour Analysis Slice Weights")
            max_pos = np.argmax(evaluation_array)
            for i in range(num_slices):
                if evaluation_array[i] > 0.0:
                    max_indicator = "<<<<<<<<<<<<<<< Max Slice" if i == max_pos else ""
                    print(str(i) + ":\t" + str(evaluation_array[i]) + "\t\t" + max_indicator)
        
        print_min_max(self.slice_weight, name="Weighted Array")
        
        max_pos = int(np.argmax(evaluation_array))

        # Store most likely ring contour and its certainty
        self.ring_contour = self.contour_results[max_pos].get_ring_contour()
        self.contour_results[max_pos].set_is_main_ring_contour(True)
        print("Ring Midpoint:\t" + str(self.contour_results[max_pos].get_ring_contour().get_midpoint()))

        if debug:
            for layer_depth in range(num_slices):
                ctr_conn = self.contour_results[layer_depth].get_ring_contour()
                if ctr_conn is not None:
                    print("=====================================================")
                    print("Layer depth:       " + "\t\t\t" + str(layer_depth))
                    print("Slice Depth Weight:" + "\t\t\t" + str(self.slice_weight[layer_depth]))
                    ctr_conn.debug()

    def calc_connected_ring_contours(self):
        pass

    # Returns an image with the contour connections
    # If the with_contours flag is set, the contours will be drawn as well
    def get_image(self, with_connections=False, with_contours=False, with_angle=False,
                  with_area=False, with_color=False, with_height=False, with_midpoint=False, with_length=False,
                  with_weight=False, with_contour_num=False, threshold=0, debug=False):

        def draw_contour_connection(conn, image, is_main_slice=False):

            def put_text(str_text, position):
                cv2.putText(image[depth, :], str(str_text), tuple(position), font, 0.4, color=color)

            def put_text_offset(str_text, offset_number):
                point = tuple(np.add(textpoint, np.multiply(offset_val, offset_number)))
                put_text(str_text, point)

            contour_general_col, contour_ring_col = (0, 0, 255), (0, 255, 0)
            # Switch color based on if a contour is a ring connection
            if with_color:
                color = contour_ring_col if conn.is_ring_contour() and is_main_slice else contour_general_col
            else:
                wght = int(conn.get_weight() * 255)
                color = (wght, wght, wght)

            cntr1, cntr2 = conn.get_all_contours()
            depth = cntr1.get_center().get_zpos()

            # Global Settings
            # c1_coords = cntr1.center.get_coordinates_int()
            # c2_coords = cntr2.center.get_coordinates_int()
            textpoint = [20, 20]
            # mid_point = [int((c1_coords[1] + c2_coords[1]) / 2), int((c1_coords[2] + c2_coords[2]) / 2)]
            offset_val = [0, 15]
            offset_num = 0
            font = cv2.FONT_HERSHEY_SIMPLEX

            if with_connections:
                # Draw Connections
                pt1 = cntr1.get_center().get_xpos(), cntr1.get_center().get_ypos()
                pt2 = cntr2.get_center().get_xpos(), cntr2.get_center().get_ypos()
                cv2.line(image[depth, :], pt1, pt2, color=color, thickness=1)

            if with_contours:
                cv2.drawContours(image[depth, :], [cntr1.contour], -1, color, 2, cv2.LINE_AA)
                cv2.drawContours(image[depth, :], [cntr2.contour], -1, color, 2, cv2.LINE_AA)

            if with_angle:
                # Text Content
                angle_target_txt = str(conn.get_target_angle())
                angle_actual_txt = str(conn.get_angle())
                # Write Text to screen
                put_text_offset("Angle target: " + angle_target_txt, offset_num)
                offset_num += 1
                put_text_offset("Angle actual: " + angle_actual_txt, offset_num)
                offset_num += 1

            if with_area or with_height:
                for contour in [cntr1, cntr2]:
                    text = str(contour.get_area()) if with_area else str(contour.get_height())
                    center_point = (contour.get_center().get_xpos() + 10, contour.get_center().get_ypos())
                    cv2.putText(image[depth, :], text, center_point, font, 0.4, color=color)

            if with_length:
                # Get length
                length_val = conn.get_distance()

                # Write text to screen
                put_text_offset("Length: " + str(length_val), offset_num)
                offset_num += 1

            if with_weight:
                # Get weight value
                weight_val_conn = conn.get_weight()
                weight_val_slice = self.slice_weight[conn.get_midpoint()[0]]

                # Write text to screen
                put_text_offset("C_Weight: " + str(weight_val_conn), offset_num)
                offset_num += 1
                put_text_offset("Slice_Weight: " + str(weight_val_slice), offset_num)
                offset_num += 1

            if with_contour_num:
                pt1, pt2 = [np.add(x.get_center().get_coordinates()[1:], [0, 30]) for x in [cntr1, cntr2]]
                put_text("1", pt1)
                put_text("2", pt2)

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
        r_c = self.ring_contour
        draw_contour_connection(r_c, return_image, is_main_slice=True)

        # Draw Midpoint
        if with_midpoint:
            mid = self.ring_contour.get_midpoint()
            cv2.circle(return_image[mid[0], :], tuple(mid[1:]), 5, 255, thickness=-1)

        # Print out to console if debug flag is set
        if debug:
            print("Ring Connection Distance:\t" + str(r_c.get_distance()))
            print("Ring Connection Angle:\t" + str(r_c.get_angle()))
            print("Ring Connection Weight:\t" + str(r_c.get_weight()))

        # Return final image
        return return_image


# Width determines the variance of the bell curve
def get_gauss_dist_norm(layers, plot=False, width: float = 10.0):

    mu = int(layers / 2)        # Centerpoint of the curve
    variance = int(layers / width)  # Original Value layers / 3
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
