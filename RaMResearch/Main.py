from RaMResearch.Utils import Interfaces as intrfce, Scripts as scr
from RaMResearch.DataIO import LoadData as ld, StoreData as sd
from RaMResearch.Filters import LegacyFilters as leg_fil
from RaMResearch.Analysis.Utils import Export as export

# from RaMResearch import Filters as fil

folder_experimental = "../TestImages/CleanUSScanCopy"
folder_normal_DICOM = "../TestImages/Round 3"
folder_clean_DICOM = "../RaMData/CleanDicomFiles_Round1"


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


# Main analysis code
def run_analysis(position=True, rotation=False, debug=False, startimage=1, debug_type: list = []):
    def load_dicoms():
        return ld.get_dicom_filepairs(folder_clean_DICOM)

    def position_debug_view():
        test_contours = dicom_array[-1].dicomanalysis.get_analysis_results("contour_analysis").get_image(
            with_connections=True, with_contours=True, with_angle=True, with_area=False, with_color=True,
            with_height=False, with_midpoint=True, with_length=True, with_weight=True, with_contour_num=True,
            threshold=0.1, debug=True)
        normal_image = dicom_array[-1].get_image(True).get_image(filtered=False)
        intrfce.imageview3d([test_contours, normal_image], windowName="Test Ring Contour")

    # File Operations
    clean_image_filenames = load_dicoms()
    dicom_array = []

    # Execution Loop
    for dicom_filename, _ in clean_image_filenames:

        # Import DICOM
        dicom_array.append(ld.import_normal_DICOM(DICOM_folder=folder_clean_DICOM, DICOM_filename=dicom_filename))

        # Get ring size?
        ring_dim = (27, 100)

        if len(dicom_array) >= startimage:
            # Filter DICOM
            # imageobject, ring_present=True, verticalsigma=7, logsigma=4, gaus1D=True,
            #                   morphological=True, morphkernelsize=3):
            leg_fil.gauslogfilter(dicom_array[-1], verticalsigma=7, logsigma=4, debug=debug)

            # Position analysis
            if position or rotation:
                p_debug = True if debug and "position" in debug_type else False
                dicom_array[-1].run_analysis(analysis_type="contour_analysis", ring_dim=ring_dim, debug=p_debug)
                if p_debug:
                    position_debug_view()

            # Rotation analysis
            if rotation:
                r_debug = True if debug and "rotation" in debug_type else False
                dicom_array[-1].run_analysis(analysis_type="rotation_analysis", ring_dim=ring_dim,
                                             debug=r_debug)
                rot_anal_obj = dicom_array[-1].get_analysis(analysis_type="rotation_analysis")
                rot_anal_obj.create_plot(save_path=sd.get_plot_savepath(dicom_array[-1].get_dicom_number()),
                                         debug=r_debug)
                export.export_ring_slices(rot_anal_obj, dicom_array[-1].get_dicom_number(), num_slices=-1,
                                          debug=r_debug)
        else:
            print("Skipped Image " + str(len(dicom_array)))


# Execute functions
run_analysis(position=True, rotation=True, debug=True, debug_type=["rotation"])
