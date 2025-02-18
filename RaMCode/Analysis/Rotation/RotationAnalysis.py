from RaMCode.Data import BasicDataStructs as bds
import RaMCode.Data.RingV2 as rv2
import numpy as np
import RaMCode.Utils.General as g
from RaMCode.Analysis.Utils import ArrayOperations as arrayops
import matplotlib.pyplot as plt
import time

import RaMCode.Utils.Interfaces as intrfc


# Class to store all entities that are associated with a rotation analysis
class RotationAnalysis:

    # The image to create
    ring_cloud: rv2.RingPointCloud = None

    # Image for analysis
    analysis_image: np.ndarray = None
    ring_cross_coord: bds.Coordinate = None

    # Analysis results stored as [front_rot][side_rot] = convolution result
    angle_results: [[int]] = np.zeros((360, 360))

    # Export arrays
    angle_plot: plt = None

    def __init__(self, image, ring_dim, ring_coord, debug=False):
        g.print_divider("Rotation Analysis")
        self.reset_parameters()
        self.create_ring(ring_dim)
        self.analysis_image = image
        self.set_ring_crosscut(ring_coord)
        self.run_analysis(analysis_range=180, step=5, debug=debug)
        self.get_export_image()

    def reset_parameters(self):
        self.angle_results = np.zeros((360, 360))
        if self.angle_plot is not None:
            self.angle_plot.clear()

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

    # Return pyplot object of the angle analysis
    def get_plot(self):
        return self.angle_plot

    # Returns the reference analysis image
    def get_analysis_image(self):
        return self.analysis_image

    # Returns ring cloud object
    def get_ring_cloud(self):
        return self.ring_cloud

    def create_plot(self, save_path: str = None, debug=False):

        # Title Code
        round_str = save_path.split(sep="/")[-1].split(sep="_")[-1][5:]
        name = save_path.split(sep="/")[-1].split(sep="_")[0] + ": Rotation Analysis Round " + round_str

        yvals = []
        xvals = []
        for angle in range(len(self.angle_results[0])):
            if self.angle_results[0][angle] > 0.01:
                yvals.append(self.angle_results[0][angle])
                xvals.append(angle)

        plt.plot(xvals, yvals)
        plt.title(name)
        plt.axis()
        plt.ylabel("Dice Coefficient")
        plt.xlabel("Ring Angle [Degrees]")
        plt.grid()

        # Save if flag is set
        if save_path:
            plt.savefig(fname=save_path + ".jpg", format="jpg", dpi=300)

        # Display and create new figure
        if debug:
            plt.show()
        else:
            plt.clf()

    # Returns a tuple with (alpha-rot, beta-rot)
    def get_angle_result(self):
        return np.unravel_index(np.argmax(self.angle_results, axis=None), self.angle_results.shape)

    # Main analysis code
    def run_analysis(self, analysis_range=(0, 180), step=1, debug=False):

        corr_run_time = []
        image_load_time = []

        # Crop image to a standard 256 x 256 x 256 array and invert it
        crs_corr_image = self.analysis_image

        # Show the point of the ring crosscut if debug
        if debug:
            crs_corr_image = g.draw_point(crs_corr_image, self.ring_cross_coord.get_coordinates_int(), color=255)

        crs_corr_image = g.crop_all_axis_to_length(g.pad_to_minimum(crs_corr_image, 256), 256)  # Pad and Crop
        crs_corr_image = g.invert_array(crs_corr_image.astype(np.int8), int_bitdepth=16)        # Invert

        # Create a boolean image
        crs_corr_image = g.threshhold(crs_corr_image, np.max(crs_corr_image)/2, astype=np.bool_)

        g.print_min_max(crs_corr_image, name="Cross Correlation Image")

        range_start, range_stop = 0, analysis_range if isinstance(analysis_range, int) else analysis_range

        # Create range of analysis angles
        angles = []
        init_angle = range_start
        while init_angle <= range_stop:
            angles.append(init_angle)
            init_angle += step

        # Loop through every angle of image
        for i in angles:

            # Get the ring image, crop and invert
            start_time = time.time()

            # Calculate max crop dim:
            crop_max_dim = g.get_max_crop(array_dim=crs_corr_image.shape,
                                          pos=self.ring_cross_coord.get_coordinates_int())

            cropped_ring_im = self.ring_cloud.get_image(outline=False, crop_dim=crop_max_dim, angle=i)\
                .get_image().astype(np.bool_)

            g.print_min_max(cropped_ring_im, name="Cross Correlation Ring Image")

            image_load_time.append(time.time()-start_time)
            start_time = time.time()

            crs_corr = arrayops.multiply_w_offset(crs_corr_image, cropped_ring_im,
                                                  self.ring_cross_coord.get_coordinates_int())

            crs_corr_sum = np.divide(np.sum(crs_corr), crop_max_dim[0]*crop_max_dim[1]*crop_max_dim[2])
            self.angle_results[0][i] = crs_corr_sum
            print("Cross Correlation Sum (Angle " + str(i) + "):\t" + str(crs_corr_sum))

            corr_run_time.append(time.time() - start_time)

        # Print average correlation time
        print("Average time to load images:\t" + str(np.average(image_load_time)))
        print("Average time to compute dice:\t" + str(np.average(corr_run_time)))

    def get_export_image(self, debug=False):

        # Create new header for terminal output
        g.print_divider("Slice Export for Machine Learning")

        # Get dimensions and positions
        ring_pos = self.ring_cross_coord.get_coordinates_int()
        max_crop = g.get_max_crop(self.analysis_image.shape, ring_pos)

        # Get the background image & the ring image with the right angle
        base_image = np.zeros(self.analysis_image.shape)
        ring_image = self.ring_cloud.get_image(outline=False, angle=self.get_angle_result()[0], crop_dim=max_crop)\
            .get_image()

        # Define Starting Corner
        ri_s = ring_image.shape
        s_c = np.subtract(ring_pos, np.divide(max_crop, 2).astype(np.int))
        base_image[s_c[0]:s_c[0]+ri_s[0], s_c[1]:s_c[1]+ri_s[1], s_c[2]:s_c[2]+ri_s[2]] += ring_image
        base_image = base_image.astype(np.uint8)

        if debug:
            # Overlay the ring in red
            intrfc.overelay_view4D(self.analysis_image, overlayarray=base_image, windowName="Ring Overlay View")

        return base_image
