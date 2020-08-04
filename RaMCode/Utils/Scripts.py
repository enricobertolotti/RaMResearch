import pandas as pd
import pydicom
import os
import numpy as np


def invert_array8(array):
    return np.subtract(255, array).astype(np.uint8)


def create_CSV_file(foldername, experimental=True, refresh=False):

    def getDicomFiles(folder_name):
        returnlist = []
        for file in os.listdir(folder_name):
            if file.endswith(".dcm"):
                if experimental:
                    filesize = os.stat(folder_name + '/' + file).st_size
                    if filesize > 10000000:
                        returnlist.append(os.path.splitext(file)[0])
                else:
                    returnlist.append(os.path.splitext(file)[0])
        return returnlist

    def get_rot_depth(dicom_foldername, dicom_filename):
        dicomheadername = dicom_foldername + '/' + dicom_filename + '.dcm'
        dicomfile = pydicom.dcmread(dicomheadername)

        # Rotation
        rot = list(map(lambda x: float(x), dicomfile.PatientName.family_name.split(",")))

        # Depth
        depth_str = dicomfile.PatientName.given_name.replace(",", ".")
        depth_float = float(depth_str) if len(depth_str) > 0 else 10.1

        return rot, depth_float

    def store_dataframe(dataframe: pd.DataFrame, csv_folder, csv_filename):
        path = csv_folder + '/' + csv_filename + '.csv'
        dataframe.to_csv(path, index=False, header=True)

    if refresh:
        # Create first dict
        dicom_image = {'name': [], 'alpha_rot': [], 'beta_rot': [], 'depth': []}

        for filename in getDicomFiles(foldername):
            rot, depth = get_rot_depth(foldername, filename)
            dicom_image['name'].append(filename)
            dicom_image['alpha_rot'].append(rot[0])
            dicom_image['beta_rot'].append(rot[1])
            dicom_image['depth'].append(depth)

        # Create DataFrame
        df = pd.DataFrame(dicom_image, columns=['name', 'alpha_rot', 'beta_rot', 'depth'])
        df.sort_values(by='name', ascending=True, inplace=True)
        store_dataframe(df, csv_folder=foldername, csv_filename="Experimental_Angles")


def get_experimental_DICOM_byangle(foldername, angle1, angle2):
    angle1, angle2 = float(angle1), float(angle2)
    for file in os.listdir(foldername):
        if file.endswith(".csv"):
            filepath = foldername + '/' + file
            df = pd.read_csv(filepath)
            loc = df.loc[(df['alpha_rot'] == angle1) & (df['beta_rot'] == angle2)]
            if not loc.empty:
                return loc['name'].iloc[0]
            else:
                return None
