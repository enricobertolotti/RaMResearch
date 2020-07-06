import numpy as np
from copy import deepcopy as dpcopy
from RaMResearch.Analysis.Position import ContourAnalysis as ca
from RaMResearch.Analysis.Rotation import RotationAnalysis as ra


# Class to handle all image related tasks
class Image:

    # Unfiltered image
    raw_image: np.ndarray = None
    transform_info = []

    # Filtered image with filtering information
    filtered_image: np.ndarray = None
    filter_info: [tuple] = []

    def __init__(self, image):
        # Initialize both image and filtered image with image
        self.raw_image = dpcopy(image)
        self.filtered_image = dpcopy(image)

    def get_image(self, filtered=False):
        if not filtered:
            return self.raw_image
        else:
            return self.filtered_image

    def set_image(self, image, filtered=False):
        if not filtered:
            self.raw_image = image
        else:
            self.filtered_image = image

    def get_filter_info(self):
        return self.filter_info

    def transpose(self, axis=(2, 1, 0)):
        self.raw_image = self.raw_image.transpose(axis) if self.raw_image is not None else None
        self.filtered_image = self.filtered_image.transpose(axis) if self.filtered_image is not None else None
        self.transform_info.append(("Transposed", axis))

    # Returns both images for a comparison view
    def get_both_images(self):
        return np.append(self.raw_image, self.filtered_image, axis=1)


# Class to handle all image metadata
class DicomMetadata:

    slice_thickness = None

    def __init__(self, slice_thickness=None):
        self.slice_thickness = slice_thickness

    def set_slice_thickness(self, slice_thickness):
        self.slice_thickness = slice_thickness

    def get_slice_thickness(self):
        return self.slice_thickness


# Define a class to store the analysis results
class AnalysisObject:

    # Contour analysis object
    contour_analysis: ca.ContourAnalysis = None

    # Rotation analysis object
    rotation_analysis: ra.RotationAnalysis = None

    # Initializer
    def __init__(self, contour_analysis: ca.ContourAnalysis = None, rotation_analysis: ra.RotationAnalysis = None):
        self.contour_analysis = contour_analysis
        self.rotation_analysis = rotation_analysis

    def set_analysis_result(self, analysis_object):
        if isinstance(analysis_object, ra.RotationAnalysis):
            self.rotation_analysis = analysis_object
        elif isinstance(analysis_object, ca.ContourAnalysis):
            self.contour_analysis = analysis_object
        else:
            raise Exception("Analysis type doesnt exist yet")

    def get_analysis_results(self, analysis_type="contour_analysis"):
        if analysis_type == "contour_analysis":
            return self.contour_analysis
        elif analysis_type == "rotation_analysis":
            return self.rotation_analysis
        else:
            raise Exception("Analysis type doesnt exist yet: contour_analysis or rotation_analysis")


# Main class to handle dicom images
class DicomObject:

    # Basic Information
    name = ""
    patient = None
    angular_scope = 40  # In degrees

    # Metadata for both images
    metadata_no_ring = None
    metadata_with_ring = None

    # Experiment Fields
    rotation = None
    depth = None

    # Original Images
    image_no_ring: Image = None
    image_with_ring: Image = None

    # Analysis Object
    dicomanalysis: AnalysisObject = None

    def __init__(self, name):
        self.name = name
        self.dicomanalysis = AnalysisObject()

    def get_image(self, ring_present: bool = True):
        if ring_present:
            return self.image_with_ring
        else:
            return self.image_no_ring

    def get_name(self):
        return self.name

    def get_patient(self):
        return self.patient

    def get_metadata(self, ring_present: bool):
        if ring_present:
            return self.metadata_with_ring
        else:
            return self.metadata_no_ring

    def set_image(self, image, ring_present: bool):
        if ring_present:
            self.image_with_ring = Image(image.copy())
        else:
            self.image_no_ring = Image(image.copy())

    def transpose_images(self, axis=(2, 1, 0)):
        self.image_no_ring.transpose(axis) if self.image_no_ring is not None else None
        self.image_with_ring.transpose(axis) if self.image_with_ring is not None else None

    def set_name(self, name):
        self.name = name
        
    def set_metadata(self, metadata: DicomMetadata, ring_present: bool):
        if ring_present:
            self.metadata_with_ring = metadata
        else:
            self.metadata_no_ring = metadata

    def invert_image(self, ring_present: bool):
        self.set_image(255 - self.get_image(ring_present=ring_present).raw_image, ring_present)

    def run_analysis(self, analysis_type="contour_analysis", debug=False):
        if analysis_type == "contour_analysis":
            self.dicomanalysis.set_analysis_result(
                ca.ContourAnalysis(self.get_image(ring_present=True).get_image(filtered=True), debug=debug))
        elif analysis_type == "rotation_analysis":
            c_a = self.dicomanalysis.get_analysis_results(analysis_type="contour_analysis")
            if c_a is not None:
                ring_pos = c_a.get_ring_contour().get_midpoint()
                self.dicomanalysis.set_analysis_result(
                    ra.RotationAnalysis(self.get_image(ring_present=True).get_image(filtered=False),
                                        ring_dim=(27, 200), ring_coord=ring_pos, debug=debug))
            else:
                raise Exception("Position analysis doesnt exist yet")

    def get_analysis(self, analysis_type="contour_analysis"):
        return self.dicomanalysis.get_analysis_results(analysis_type)