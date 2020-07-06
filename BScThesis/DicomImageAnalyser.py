from BScThesis import HelperFunctions as hf, ExportScripts as es, DicomScripts as ds, ImageFilters as imfil
from os import path, stat
import numpy as np

############# Global Variables #############

ringthickness = 12   # mm


############# Function Definitions #########


# Filename without any file-ending, ringdiameter in mm
def analizeSingleDicom(dicomfilename, excelfile="", experiment=False):

    # Import Dicom file and metadata
    dicomfile = ds.importdicomfile(dicomfilename, experiment)

    # Rotate and transpose DICOM to standard orientation
    standardize_dicom_rotation(dicomfile, experiment=experiment)

    # Export thresholded DICOM experiment
    # es.export_filtered_dicom(dicomfile, image_type="normal", array_type="base", filter_type="LoG",
    #                          values=[(3, 3), (7, 4), (13, 5), (21, 6)])
    #
    # # Export thresholded DICOM experiment
    # es.export_filtered_dicom(dicomfile, image_type="normal", array_type="base", filter_type="LoG",
    #                          values=[(3, 4), (7, 4), (13, 4), (21, 4)])
    # es.save_dicomimage_as_layers(dicomfile)

    # Create all filtered, rotated and transposed arrays
    create_filtered_arrays(dicomfile, experiment=experiment)

    # es.export_filtered_dicom(dicomfile, image_type="filter", array_type="base", filter_type="pattern",
    #                          values=[])

    # Show both arrays next to each other
    ds.imageview3d(dicomfile.get_all_arrays(), dicomfile.getname())

    # Contour analysis
    # dicom_contour_analysis(dicomfile, min_thresh=0.4, view=True)
    dicom_contour_analysis(dicomfile, min_thresh=0, view=True)

    # Get patient for dicom file if not an experiment
    # if not experiment:
    #     dicomfile.setpatient(hf.makePatientData(dicomfilename, excelfile))

    # # Basic ring localization with correlation
    # position_determination_with_correlation(dicomfile, view=True)
    # es.export_dicom_array(dicomfile, "correlation_array")

    # # Get the point of the ring closest to the camera
    # set_closest_point(dicomfile, view=True)

    #
    #
    # # Test bilateral filter
    # array = dicomfile.get_image("normal").get_array("base")
    # depth = dicomfile.getclosestedgepoint()[0]
    # imfil.dicom_bilateral_filter(array[depth], 10, 10, view=True)

    # For the report
    # rs.plot_ring_crosscut(dicomfile, image_type="normal", array_type="base")
    # set_closest_point(dicomfile, view=False)
    #
    # # Create focus array and rotate it around its axis
    rotation_analysis(dicomfile, image_type="normal", view=False)
    #
    # es.save_dicomimage_as_layers(dicomfile, imagetype="normal", arraytype="base")

    return dicomfile


# Loop through dicom files and analize them
def analizeDicom(dicomfileorfolder, excelpath="", experiment_images=False):

    analysed_files = []

    if path.isdir(dicomfileorfolder):
        for file in ds.getDicomFiles(dicomfileorfolder):
            if experiment_images:
                filesize = stat(dicomfileorfolder + '/' + file + ".dcm").st_size
                if filesize > 100000:
                    analysed_files.append(analizeSingleDicom(dicomfileorfolder + '/' + file,
                                                             excelpath, experiment_images))
            else:
                analysed_files.append(analizeSingleDicom(dicomfileorfolder + '/' + file, excelpath, experiment_images))
            view_rotation_analysis_results(analysed_files)
    else:
        if experiment_images:
            filesize = stat(dicomfileorfolder + '/' + dicomfileorfolder + ".dcm").st_size
            if filesize > 100000:
                analysed_files.append(analizeSingleDicom(dicomfileorfolder + '/' + dicomfileorfolder, excelpath))
        else:
            analysed_files.append(analizeSingleDicom(dicomfileorfolder, excelpath, experiment_images))
        view_rotation_analysis_results(analysed_files)

    return analysed_files[0]


def standardize_dicom_rotation(dicomobject, experiment):
    if experiment:
        # Rotate Dicom file
        dicomobject.get_image().rot90cc()

    else:
        # Transpose Dicom file
        dicomobject.get_image().dicom_transpose()


def create_filtered_arrays(dicomobject, experiment):
    # imfil.adaptive_histogram_equalisation(dicomobject, image_type="normal", cliplimit=2, tilegridsize=(6, 6),
    # view=True)

    # imfil.dicom_bilateral_filter_3D(dicomobject, sigma_Color=41, sigma_Space=7, image_type="normal")

    # dicomobject.set_image(imfil.quauntize_3D(dicomobject, image_type="normal"))

    filteredim = dicomobject.get_image("normal").get_array("base")
    dicomobject.set_image(imfil.gauslogfilter(filteredim, 7, 4, gaus1D=True), image_type="filtered")
    dicomobject.get_image(image_type="filtered").set_gauslog_params([7, 4])
    # dicomobject.set_image(imfil.gauslogfilter(dicomobject.get_image().get_array(), 7, 4), image_type="filtered")
    # dicomobject.get_image(image_type="filtered").set_gauslog_params([7, 4])


def position_determination_with_correlation(dicomfile, view=False):

    # Create pattern that will be searched for
    dicomfile.createsearchpattern()

    # 3D correlation array
    convolvedvolume = hf.findPatternInImageTemplate(dicomfile.get_image("filtered").get_array(), dicomfile.pattern)

    # TODO Check the need for normalization
    # Normalize the corollated array ==> although check the
    convolvednormalized = hf.normalize(convolvedvolume)
    a = np.min(convolvednormalized), np.max(convolvednormalized)
    dicomfile.correlated_array = (convolvednormalized*255).astype(np.uint8)
    # Find the maximum of the array and
    maxpos = hf.getminmaxloc(convolvednormalized)
    print(maxpos)

    # Convert the ring position to absolute and store in the dicomfile
    crossabspos = hf.getAbsPos(dicomfile.get_image("filtered").get_array(), convolvednormalized, maxpos[1])
    dicomfile.setringcoordinates(crossabspos)

    # If the view flag has been set to true
    if view:
        norm_array_with_crosshair = hf.drawcrosshairs(dicomfile.get_image().get_array(), crossabspos, 1)
        filtered_array_with_crosshairs = hf.drawcrosshairs(dicomfile.get_image("filtered").get_array(), crossabspos, 1)

        # Save the array to the folder
        es.save_ring_pos(dicomfile, which_pos="center", folder="position_crosshair", filename="pos")
        es.save_dicomimage_as_layers(dicomfile, norm_array_with_crosshair, imagetype="normal", arraytype="base",
                                     folder="position_crosshair", name_ext="pos_crosshair")
        es.save_dicomimage_as_layers(dicomfile, filtered_array_with_crosshairs, imagetype="filtered", arraytype="base",
                                     folder="position_crosshair", name_ext="pos_crosshair")
        # ds.imageview3d([norm_array_with_crosshair, filtered_array_with_crosshairs], windowName="Correlation Crosshair")


def set_closest_point(dicomfile, view=False):
    # Find point closest to the camera
    if dicomfile.closestedge_point is None:
        dicomfile.setclosestringpoint()

    # If the view flag has been set to true
    if view:
        norm_array_with_crosshair = hf.drawcrosshairs(dicomfile.get_image().get_array(), dicomfile.closestedge_point, 1)
        filtered_array_with_crosshairs = hf.drawcrosshairs(dicomfile.get_image("filtered").get_array(), dicomfile.closestedge_point, 1)
        ds.imageview3d([norm_array_with_crosshair, filtered_array_with_crosshairs], windowName="Closest Edgepoint")


def rotation_analysis(dicomfile, image_type="normal", view=False):

    # Create the front view array arround the closest edge point of the ring
    # dicomfile.frontview = dicomfile.ringfocusarray(dicomfile.get_image().get_array())   #### depriciated
    dicomfile.create_ringfocusarray(size=40, image_type="normal")
    dicomfile.create_ringfocusarray(size=40, image_type="filter")
    # Rotate the array around the z-axis
    dicomfile.createrotatedarrays(rot_values=(0, 360), image_type="normal")
    dicomfile.createrotatedarrays(rot_values=(0, 360), image_type="filter")

    dicomfile.create_slices_for_export(image_type="normal")
    dicomfile.create_slices_for_export(image_type="filter")

    es.save_rotation_slices(dicomfile, image_type="normal")
    es.save_rotation_slices(dicomfile, image_type="filter")

    dicomfile.filter_ringfocusarray(image_type=image_type)

    dicomfile.rotation_by_convolution()

    es.export_dicom_rotation(dicomfile)

    # Calculate the similarity of the perpendicular slices
    dicomfile.calcsimilaritycoefficient(image_type="normal")
    dicomfile.calcsimilaritycoefficient(image_type="filter")

    # If the view flag has been set to true
    if view:
        if "filter" in image_type.lower():
            ds.viewer3d(dicomfile.contour_ring_top_view_filtered, "Ring Top View Filtered")
        else:
            ds.viewer3d(dicomfile.contour_ring_top_view_normal, "Ring Top View Normal")
        ds.viewer3d_wRot(dicomfile, image_type=image_type)
        ds.sliceVisualizer(dicomfile, image_type=image_type)


def dicom_contour_analysis(dicomfile, min_thresh: float = 0, view=False):

    # Calculate contours and all possible combinations of centerpoints
    dicomfile.contour_thresh = min_thresh
    dicomfile.create_contours()
    dicomfile.create_contour_pairs(min_thresh=min_thresh)

    # Get all depths where contours are present
    lay_w_con = []
    [lay_w_con.append(dep) for (_, dep, _) in dicomfile.contour_pairs if dep not in lay_w_con]

    for depth in lay_w_con:
        hf.drawcontourconnections(dicomfile, linethickness=1, depth=depth)

    dicomfile.color_max_contour()
    es.save_dicom_contour_analysis(dicomfile)

    dicomfile.set_contour_ringcoordinates()
    dicomfile.contour_based_closest_ring_v2()


def view_rotation_analysis_results(dicomfilelist, save="False"):

    if dicomfilelist is not None and not isinstance(dicomfilelist, list):
        hf.visualize_rotation_plots(dicomfilelist)
    elif dicomfilelist is not None:
        hf.visualize_rotation_plots(dicomfilelist[-1])
