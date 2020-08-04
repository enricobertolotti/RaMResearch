import numpy as np


class DicomImage:

    # Base array
    array = None

    # An array focused around the ring
    focus_array = None
    focus_array_dim = np.zeros(3)
    focus_array_pos = np.zeros(3)

    # A rotated array around the closest point
    focus_array_rot = None
    focus_array_rot_anglerange = np.zeros(2)

    # Countours
    contours = []
    contour_pairs = []
    contour_images = []
    contour_images_normal = []
    contour_ring_loc = None
    contour_thresh = 0

    # Ring Crosscut array
    ring_cross_array = None

    def __init__(self, array):
        self.array = array

    def set_array(self, array, array_type="base"):
        if "base" in array_type.lower():
            self.array = array
        elif "focus" in array_type.lower() and "rot" in array_type.lower():
            self.focus_array_rot = array
        elif "ring" in array_type.lower() and "cross" in array_type.lower():
            self.ring_cross_array = array
        else:
            self.focus_array = array

    def get_array(self, array_type="base"):
        if "base" in array_type.lower() or "normal" in array_type.lower():
            return self.array
        elif "focus" in array_type.lower() and "rot" in array_type.lower():
            return self.focus_array_rot
        elif "ring" in array_type.lower() and "cross" in array_type.lower():
            return self.ring_cross_array
        else:
            return self.focus_array

    def set_rotation_params(self, rot_value):
        self.focus_array_rot_anglerange = rot_value

    def get_rotation_params(self):
        return self.focus_array_rot_anglerange

    def dicom_transpose(self, axis=(2, 1, 0)):
        self.array = self.array.transpose(axis)

    # Rotate 90 degrees counter clockwise
    def rot90cc(self, number_of_rot=1):
        self.array = np.rot90(self.array, k=number_of_rot, axes=(1, 2))

    def getcrosscut(self, array_type="base", rot_array_angle=0):

        # Check which array to get the crosscut of
        if "base" in array_type.lower():
            array = self.array
        elif "focus" in array_type.lower() and "rot" in array_type.lower():
            array = self.focus_array_rot[rot_array_angle]
        else:
            array = self.focus_array

        # Get dimensions and return slices
        array_dim = array.shape
        hcrosscut = array[:, :, int(array_dim[2] / 2)]
        vcrosscut = array[:, int(array_dim[1] / 2), :]
        return [hcrosscut, vcrosscut]


class StandardImage(DicomImage):
    pass


class FilteredImage(DicomImage):

    # Laplacian of Gaussian Parameters
    gauslog_filter_param_vertsigma = 0
    gauslog_filter_param_logsigma = 0

    def set_gauslog_params(self, params):
        self.gauslog_filter_param_vertsigma = params[0]
        self.gauslog_filter_param_logsigma = params[1]

    def get_gauslog_params(self):
        return [self.gauslog_filter_param_vertsigma, self.gauslog_filter_param_logsigma]
