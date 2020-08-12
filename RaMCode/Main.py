# RaMResearch Imports
from RaMCode.Data import ManualSegmentation as manseg, DataStructs as ds
from RaMCode.DataIO import LoadData as ld, StoreData as sd, GeneralUtils as gutils_dataio
from RaMCode.Filters import LegacyFilters as leg_fil
from RaMCode.Utils import Interfaces as intrfce, Scripts as scr, General as general
from RaMCode.Utils import Interaction as interaction

# Miscellaneous Imports
from typing import List

folder_experimental = "../TestImages/CleanUSScanCopy"
folder_normal_DICOM = "../TestImages/Round 3"
folder_clean_DICOM = "../RaMData/CleanDicomFiles_Round1"


def run(interactive=True, debug=False):

    def initialize():
        general.print_divider("RaM Research - Pessary Analysis", spacers=2)
        print("What code do you want to run?")
        print("1: Position Analysis\n2: Rotation Analysis\n3: Export Slices For Machine Learning")
        return interaction.get_list(message="Multiple options are allowed, separated by comma: ")

    def get_working_folder():
        if interactive and interaction.get_boolean("Do you want to use a custom data folder? (Yes/No): "):
            data_folder = interaction.get_directory(message="Custom Data Folder: ", print_file_list=debug)
        else:
            data_folder = gutils_dataio.get_default_file_folder(folder_type="data")
        if debug:
            print("Data Folder:\t" + data_folder)
        return data_folder

    # cwd is absolute path to the directory and should be set before calling function
    def get_dicom_file_list():
        if interactive and interaction.get_boolean("Do you want to analyse all dicoms? (Yes/No): "):
            return ld.get_dicom_filepaths(cwd)
        elif interactive:
            return interaction.get_number(message="Enter Dicom ID: (The number at the start of the dicom filename): ")

    def load_dicom(dicom_filepath):
        return ld.import_normal_DICOM(DICOM_filepath=dicom_filepath)

    def load_manual_segmentation():
        if interactive and interaction.get_boolean(message="Use Custom Manual Segmentation File Location? (Yes/No): "):
            file_path = interaction.get_directory(message="Custom Manual Segmentation File Location: ")
        else:
            file_path = gutils_dataio.get_default_file_folder(folder_type="segmentation")
        return manseg.ManualSegmentation(file_path_csv=file_path)

    # Main Position Analysis Function
    def position_analysis():

        # Helper functions:
        def position_debug_view(dicom_object: ds.DicomObject):
            debug_contours = dicom_object.dicomanalysis.get_analysis_results("contour_analysis").get_image(
                with_connections=True, with_contours=True, with_angle=True, with_area=False, with_color=True,
                with_height=False, with_midpoint=True, with_length=True, with_weight=True, with_contour_num=True,
                threshold=0.1, debug=True)
            normal_image = dicom_object.get_image(True).get_image(filtered=False)
            intrfce.imageview3d([debug_contours, normal_image], windowName="Test Ring Contour")

        general.print_divider("Position Analysis", spacers=1)
        if interactive and interaction.get_boolean("Execute Position Estimation? (Yes/No): "):
            dicom_files = get_dicom_file_list()
            for file_path in dicom_files:
                # Import dicom
                dicom_obj = load_dicom(dicom_filepath=file_path)

                # Perform analysis
                # 1. Filter dicom image:
                leg_fil.gauslogfilter(dicom_obj, verticalsigma=7, logsigma=4, debug=debug)

                # 2. Get values required for analysis
                dicom_id = dicom_obj.get_dicom_number()
                ring_dim = man_seg_obj.get_ring_dim(dicom_id=dicom_id)

                # 3. Run contour analysis
                dicom_obj.run_analysis(analysis_type="contour_analysis", ring_dim=ring_dim, debug=debug)

                # If debug show the image
                if debug:
                    position_debug_view(dicom_obj)

                # Store dicom in array for further analysis (rotation, export etc.. )
                dicom_obj_array.append(dicom_obj)

    # Main Rotation Analysis Function
    def rotation_analysis():
        general.print_divider("Rotation Analysis", spacers=1)

    def store_analysis_data():

        general.print_divider(text="Saving Data....")

        for dicom_obj in dicom_obj_array:
            dicom_id = dicom_obj.get_dicom_number()
            res_dict = dicom_obj.get_analysis_object().get_simple_result()

            # Get individual variables
            man_seg_obj.set_position(dicom_id, res_dict['pos'], man_seg=False)
            man_seg_obj.set_rotation(dicom_id, rotation=res_dict['rot'], man_seg=False)

    # Main Export Function
    def slice_export():
        pass

    options = initialize()
    general.insert_spacer(spacers=1)

    cwd = get_working_folder()

    man_seg_obj = load_manual_segmentation()

    dicom_obj_array = []

    if 1 in options:
        position_analysis()

    if 2 in options:
        rotation_analysis()

    if 1 in options or 2 in options:
        store_analysis_data()

    if 3 in options:
        slice_export()


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

    run(interactive=True, debug=False)

    # run_obsolete(position_analysis=True, rotation_analysis=False, ml_export=True,
    #              debug_type=["position", "rotation"])
