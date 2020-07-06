from RaMResearch.Data import BasicDataStructs as bds
import RaMResearch.Data.RingV2 as rv2
import numpy as np
import RaMResearch.Utils.General as g
import RaMResearch.Utils.Interfaces as intrfce
from RaMResearch.Analysis.Utils import ArrayOperations as arrayops
import time


# Class to store all entities that are associated with a rotation analysis
class RotationAnalysis:

    # The image to create
    ring_cloud: rv2.RingPointCloud = None

    # Image for analysis
    analysis_image: np.ndarray = None
    ring_cross_coord: bds.Coordinate = None

    # Analysis results stored as [front_rot][side_rot] = convolution result
    angle_results: [[int]] = np.empty((360, 360))

    def __init__(self, image, ring_dim, ring_coord, debug=False):
        self.reset_parameters()
        self.create_ring(ring_dim)
        self.analysis_image = image
        self.set_ring_crosscut(ring_coord)
        self.run_analysis(debug=debug)

    def reset_parameters(self):
        self.angle_results = np.empty((360, 360))

    def set_ring_crosscut(self, ring_coord):
        if isinstance(ring_coord, (tuple, np.ndarray, list)):
            self.ring_cross_coord = bds.Coordinate([ring_coord[0], ring_coord[1], ring_coord[2]])
        elif isinstance(ring_coord, bds.Coordinate):
            self.ring_cross_coord = ring_coord
        else:
            raise TypeError("ring_coord must be a tuple, ndarray or a coordinate object")

    def get_ring_crosscut(self):
        return self.ring_cross_coord

    # Create a ring with (sm_diameter, lg_diameter)
    def create_ring(self, ring_dim):
        self.ring_cloud = rv2.RingPointCloud(ring_dim[0], ring_dim[1])

    # Return rotation analysis result
    def get_result(self):
        return np.unravel_index(np.argmax(self.angle_results), self.angle_results.shape)

    def get_rotated_image(self, angle):
        r_small, r_large = self.ring_cloud.radius_small, self.ring_cloud.radius_large
        return rv2.import_ring_array(r_small, r_large, angle, filled=True)

    def run_analysis(self, debug=False):

        # Crop image to a standard 256 x 256 x 256 array and invert it
        crs_corr_image = self.analysis_image

        crs_corr_image = g.draw_point(crs_corr_image, self.ring_cross_coord.get_coordinates_int(), color=255)
        crs_corr_image = g.crop_all_axis_to_length(g.pad_to_minimum(crs_corr_image, 256), 256)
        crs_corr_image = g.invert_array(crs_corr_image.astype(np.int8), int_bitdepth=16)

        g.print_min_max(crs_corr_image, name="Cross Correlation Image")

        list_sums = []

        # Loop through every angle of image
        for i in range(0, 180):
            # Get the ring image, crop and invert
            time1 = time.time()
            ring_image_obj = self.get_rotated_image(i)

            # intrfce.imageview3d(ring_image_obj.get_image(), windowName="Test View Ring Image")

            crop_dim = (64, 64, 64)
            cropped_ring_im = ring_image_obj.get_cropped_image(crop_dim)

            g.print_min_max(cropped_ring_im, name="Cross Correlation Ring Image")
            print("Time to get image = \t" + str(time.time() - time1))
            
            # intrfce.imageview3d(cropped_ring_im, windowName="Test View Ring Image Cropped")

            # Pattern match images
            time1 = time.time()
            print("Correlating...")
            crs_corr = arrayops.multiply_w_offset(crs_corr_image, cropped_ring_im,
                                                  self.ring_cross_coord.get_coordinates_int())
            print("Finished Correlating")
            print("Time to correlate = \t" + str(time.time() - time1))
            crs_corr_sum = np.sum(crs_corr)
            list_sums.append(crs_corr_sum)
            print("Cross Correlation Sum (Angle " + str(i) + "):\t" + str(crs_corr_sum))

        if debug:
            intrfce.plot1D(list_sums)
        print("List Sums:\t" + str(list_sums))
