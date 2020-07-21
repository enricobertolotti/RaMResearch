import pandas as pd


# Defaults
excelfile_folder: str = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/Manual_Segmentation/"
excelfile_filename: str = "Manual_Segmentation_Results.csv"


class ManualSegmentation:
    # Dicom & Ring Information
    dicom_ID: int = -1

    # Excel File
    excelfile_path: str = ""
    excelfile: pd.DataFrame = None

    # Imported values
    rotation: float = 0.0

    def __init__(self, dicom_ID: int):
        self.clear_variables()  # Clear all remaining variables from last object
        self.dicom_ID = dicom_ID
        self.loadCSV()  # Load the excel file that tracks the manual segmentiation

    def clear_variables(self):
        self.dicom_ID = -1
        self.excelfile_path = excelfile_folder + excelfile_filename

    def loadCSV(self):
        df = pd.read_csv(self.excelfile_path, header=0, delimiter=";")
        dicomid_row_index = df.index[df['Dicom_ID'] == self.dicom_ID].tolist()[0]
        self.rotation = df.at[dicomid_row_index, 'Manual_Angle']

    def get_rotation(self):
        return self.rotation

