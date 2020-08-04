# RaMResearch Imports
from RaMCode.Data import ManualSegmentation as manseg, DataStructs as ds
from RaMCode.DataIO import LoadData as ld, StoreData as sd
from RaMCode.Filters import LegacyFilters as leg_fil
from RaMCode.Utils import Interfaces as intrfce, Scripts as scr, General as general

# Miscellaneous Imports
from typing import List

folder_experimental = "../TestImages/CleanUSScanCopy"
folder_normal_DICOM = "../TestImages/Round 3"
folder_clean_DICOM = "../RaMData/CleanDicomFiles_Round1"


def run(interactive=True):
    def print_info():
        general.print_divider("RaM Research - Pessary Analysis", spacers=2)

    
    def position_analysis():
        general.print_divider("RaM Research - Pessary Analysis", spacers=2)


# Get dicom by angle code
def get_dicom_by_angle():
    # diameter = 55 pixels

    scr.create_CSV_file(folder_experimental)
    angles = (0, 0)
    imagename = scr.get_experimental_DICOM_byangle(folder_experimental, angles[0], angles[1])

    if imagename is not None:
        print("Image ID:\t" + imagename)
        image = ld.import_experimental_dicom(folder_experimental, dicomfilename=imagename)
        image.invert_image(ring_present=True)
        # intrfce.imageview3d(image.image_with_ring, windowName=image.get_name())
    else:
        print("No image found with the angles: \nAlpha:\t" + str(angles[0]) + "\nBeta:\t" + str(angles[1]))


# # Main analysis code
# def run_analysis(position_analysis=True, rotation_analysis=False, debug=False, startimage=1, debug_type: list = []):
#     def load_dicoms():
#         return ld.get_dicom_filepairs(folder_clean_DICOM)
#
#     def position_debug_view():
#         test_contours = dicom_array[-1].dicomanalysis.get_analysis_results("contour_analysis").get_image(
#             with_connections=True, with_contours=True, with_angle=True, with_area=False, with_color=True,
#             with_height=False, with_midpoint=True, with_length=True, with_weight=True, with_contour_num=True,
#             threshold=0.1, debug=True)
#         normal_image = dicom_array[-1].get_image(True).get_image(filtered=False)
#         intrfce.imageview3d([test_contours, normal_image], windowName="Test Ring Contour")
#
#     # File Operations
#     clean_image_filenames = load_dicoms()
#     dicom_array = []
#
#     # Execution Loop
#     for dicom_filename, _ in clean_image_filenames:
#
#         # Import DICOM
#         dicom_array.append(ld.import_normal_DICOM(DICOM_folder=folder_clean_DICOM, DICOM_filename=dicom_filename))
#
#         ring_dim = (27, 100)
#
#         if len(dicom_array) >= startimage:
#
#             general.print_divider("DICOM Analysis:\t" + dicom_array[-1].get_name(), spacers=2)
#
#             # Filter DICOM
#             # imageobject, ring_present=True, verticalsigma=7, logsigma=4, gaus1D=True,
#             #                   morphological=True, morphkernelsize=3):
#             leg_fil.gauslogfilter(dicom_array[-1], verticalsigma=7, logsigma=4, debug=debug)
#
#             # Position analysis
#             if position_analysis or rotation_analysis:
#                 p_debug = True if debug and "position" in debug_type else False
#                 dicom_array[-1].run_analysis(analysis_type="contour_analysis", ring_dim=ring_dim, debug=p_debug)
#                 if p_debug:
#                     position_debug_view()
#
#             # Rotation analysis
#             if rotation_analysis:
#                 r_debug = True if debug and "rotation" in debug_type else False
#                 dicom_array[-1].run_analysis(analysis_type="rotation_analysis", ring_dim=ring_dim,
#                                              debug=r_debug)
#                 rot_anal_obj = dicom_array[-1].get_analysis(analysis_type="rotation_analysis")
#                 rot_anal_obj.create_plot(save_path=sd.get_plot_savepath(dicom_array[-1].get_dicom_number()),
#                                          debug=r_debug)
#                 export.export_ring_slices(rot_anal_obj, dicom_array[-1].get_dicom_number(), num_slices=-1,
#                                           debug=r_debug)
#             # Otherwise get rotation from manual segmentation
#             else:
#
#                 # Get parameters for export image
#                 dicom_id = dicom_array[-1].get_dicom_number()
#                 rot = int(dicom_array[-1].get_manual_segmentation().get_rotation())
#                 dim = dicom_array[-1].get_image().get_image().shape
#                 pos = dicom_array[-1].get_analysis(analysis_type="contour_analysis").get_ring_position()
#                 r_small, r_large = ring_dim
#                 num_slices = 100
#
#                 # Get Images
#                 mask_image = export.get_ring_mask_image(dim, pos, rot, r_small, r_large)
#                 base_image = dicom_array[-1].get_image(ring_present=True).get_image(filtered=False)
#
#                 intrfce.imageview3d(mask_image, windowName="TestImage")
#
#                 # Save Image to folder
#                 export.ml_export_image(mask_image, dicom_id, num_slices, ring_pos=pos, image_type="mask")
#                 export.ml_export_image(base_image, dicom_id, num_slices, ring_pos=pos, image_type="base")
#
#         else:
#             print("Skipped Image " + str(len(dicom_array)))
#
#
# # Execute functions
# # If "rotation_analysis" is set to False, will use the manual segmentation rotation value
# run_analysis(position_analysis=True, rotation_analysis=False, debug=True, debug_type=["rotation"])


def run_obsolete(position_analysis=True, rotation_analysis=False, ml_export=True,
                 debug_type: List[str] = []):

    def run_position_analysis(dicom_image: ds.DicomObject):
        # Define helper function
        def position_debug_view():
            test_contours = dicom_array[-1].dicomanalysis.get_analysis_results("contour_analysis").get_image(
                    with_connections=True, with_contours=True, with_angle=True, with_area=False, with_color=True,
                    with_height=False, with_midpoint=True, with_length=True, with_weight=True, with_contour_num=True,
                    threshold=0.1, debug=True)
            normal_image = dicom_array[-1].get_image(True).get_image(filtered=False)
            intrfce.imageview3d([test_contours, normal_image], windowName="Test Ring Contour")
        # Set debug flag
        debug = True if "position" in debug_type else False
        # Filter dicom image:
        leg_fil.gauslogfilter(dicom_array[-1], verticalsigma=7, logsigma=4, debug=debug)
        # Run contour analysis
        dicom_image.run_analysis(analysis_type="contour_analysis", ring_dim=ring_dim, debug=debug)
        # Show debug view if required
        if debug:
            position_debug_view()

    def run_rotation_analysis(dicom_image: ds.DicomObject):
        debug = True if "rotation" in debug_type else False
        dicom_image.run_analysis(analysis_type="rotation_analysis", ring_dim=ring_dim, debug=debug)
        rot_anal_obj = dicom_image.get_analysis(analysis_type="rotation_analysis")
        rot_anal_obj.create_plot(save_path=sd.get_plot_savepath(dicom_image.get_dicom_number()), debug=debug)

    # Load Data from the CSV file
    manual_seg = manseg.ManualSegmentation()
    all_dicom_ids = manual_seg.get_dicom_list()

    # TODO get ring dim from excel file
    ring_dim = (27, 100)

    # Load dicom files
    dicom_array = []

    # Run analysis for all files
    for dicom_id in all_dicom_ids:

        # Load Dicom File
        dicom_obj = ld.import_dicom(dicom_id=dicom_id)
        dicom_array.append(dicom_obj)

        # Position Analysis Code
        if position_analysis:
            # Run position analysis
            run_position_analysis(dicom_array[-1])
        else:
            # Get position from the manual segmentation file
            # TODO implement this
            pass

        # Rotation Analysis Code
        if rotation_analysis:
            # Run rotation analysis
            run_rotation_analysis(dicom_array[-1])

        # Export Code
        if ml_export:
            # Run Export Scripts
            # TODO finalize this
            use_rotation_analysis_result = rotation_analysis
            use_position_analysis_result = position_analysis

            pass


# If file is run as main file
if __name__ == '__main__':
    # Main Function Call

    position_analysis()

    run_obsolete(position_analysis=True, rotation_analysis=False, ml_export=True,
                 debug_type=["position", "rotation"])
