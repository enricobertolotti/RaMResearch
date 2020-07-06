import numpy as np
from pathlib import Path as path
import RaMResearch.Data.RingV2 as rv2
from RaMResearch.Data import DataStructs as ds

import pandas as pd


root_path = "/RaMData/"
default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/Numpy_Ring_Definitions"


def store_ring(ring_obj: rv2.RingPointCloud, folder=default_folder, angle=None):

    r_small = ring_obj.radius_small
    r_large = ring_obj.radius_large

    if angle is not None:
        save_array = [ring_obj.get_rot_image(angle, True), ring_obj.get_rot_image(angle, False)]
    else:
        save_array = [ring_obj.get_image(True), ring_obj.get_image(False)]

    ring_angle = 0 if angle is None else angle

    save_path = folder + '/' + str(r_small) + '/' + str(r_large) + '/' + str(ring_angle)
    save_name = str(r_small) + '_' + str(r_large) + '_' + str(ring_angle) + '_'

    # Create path if necessary
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save Numpy Array
    np.save(save_path + '/' + save_name + "outline.npy", save_array[0])
    np.save(save_path + '/' + save_name + "filled.npy", save_array[1])


def store_mask(mask_array, folder=""):

    # Folder Preparations
    folder = "Numpy_Dicom_Mask" if len(folder) == 0 else folder
    save_path = root_path + folder + '/'

    # Filename Preparations
    mask_array_dim = mask_array.shape
    mask_array_dim_str = [str(x) for x in mask_array_dim]
    separator = "_"
    save_name = separator.join(mask_array_dim_str) + "_numpy_mask.npy"

    # Create path if necessary
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path + '/' + save_name, mask_array)


def store_dicom_analysis_data(dicom_image: ds.DicomObject, folder="", filename=""):

    # Set defaults if nothing was given
    default_foldername = "Analysis_Results/" if folder == "" else folder
    default_filename = "Dicom_analysis.csv" if filename == "" else filename

    def file_exists(fullpath):
        test = path(fullpath).is_file()
        return test

    def create_file(file_path, fullpath):

        # Create Path and Parents
        file_path_obj = path(file_path)
        file_path_obj.mkdir(parents=True, exist_ok=True)

        # Create file
        coloumns = ["ID", "CP", "POS_EDGE_CLOSE", "POS_EDGE_FAR", "FRONT_ROT", "SIDE_ROT"]
        df = pd.DataFrame(columns=coloumns)
        df.to_csv(fullpath)

    def load_csv_into_df(path_to_file):
        return pd.read_csv(path_to_file)

    def check_if_row_exists(df: pd.DataFrame, dicom_name):
        return not df.loc[df["ID"] == dicom_name].empty

    def add_data(df: pd.DataFrame, dicomimage: ds.DicomObject):

        # Dicom Identiier Name
        dicom_id = dicom_image.name

        # If data exists, delete it
        if check_if_row_exists(df=df, dicom_name=dicom_id):
            df.set_index("ID")
            df.drop(dicom_id, axis=0)

        # Analysis Variables
        analysis_file = dicomimage.dicomanalysis

        cp = get_analysis_variable(analysis_file, var="cp")
        pos_r_close = get_analysis_variable(analysis_file, var="posringclose")
        pos_r_far = get_analysis_variable(analysis_file, var="posringfar")
        frontrot = get_analysis_variable(analysis_file, var="frontrot")
        siderot = get_analysis_variable(analysis_file, var="siderot")

        # Create array which will be a pandas row
        dicom_frame = [1, dicom_id, cp, pos_r_close, pos_r_far, frontrot, siderot]

        # Append data and sort
        return df.append(pd.DataFrame([dicom_frame], columns=df.columns), ignore_index=False).sort_values('ID').reset_index(drop=True)

    def get_analysis_variable(analysed_obj: ds.AnalysisObject, var=""):
        ca = analysed_obj.get_analysis_results("contour_analysis")
        ra = analysed_obj.get_analysis_results("rotation_analysis")
        if var == "cp":
            return ca.get_ring_contour().get_midpoint()
        elif var == "posringclose":
            return ca.get_ring_contour().get_all_contours()[0].get_center()
        elif var == "posringfar":
            return ca.get_ring_contour().get_all_contours()[1].get_center()
        elif var == "rotation1":
            return ra.get_result()[0]
        elif var == "rotation1":
            return ra.get_result()[1]
        else:
            return None

    # Full path to the file
    full_path = root_path + default_foldername + default_filename

    # If file doesnt exist create it
    if not file_exists(full_path):
        create_file(root_path + default_folder, full_path)

    # load the .csv file, add data, and save
    dataframe = load_csv_into_df(full_path)
    dataframe = add_data(dataframe, dicom_image)
    dataframe.to_csv(full_path)
