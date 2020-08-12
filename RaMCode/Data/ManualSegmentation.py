import pandas as pd


# Main class for storing data
class ManualSegmentation:

    # CSV File
    file_path: str = ""
    csvfile: pd.DataFrame = None

    # Dictionary for quick lookup
    man_seg_dict: dict = {}

    # Imported values
    rotation: float = 0.0

    def __init__(self, file_path_csv):
        self.clear_variables()  # Clear all remaining variables from last object
        self.file_path = file_path_csv
        self.loadCSV()  # Load the excel file that tracks the manual segmentiation

    def clear_variables(self):
        pass

    def loadCSV(self):
        self.csvfile = pd.read_csv(self.file_path, header=0, delimiter=";")
        csv_dict = self.csvfile.to_dict()
        try:
            csv_dict["Dicom_ID"].values()
        except KeyError:
            self.csvfile = pd.read_csv(self.file_path, header=0, delimiter="")
            csv_dict = self.csvfile.to_dict()

        self.man_seg_dict = csv_dict

    def add_dicomid(self, dicom_id):
        if not (dicom_id in self.csvfile):
            self.csvfile.append(pd.Series([dicom_id]), ignore_index=True)
            self.csvfile.sort_values(by="Dicom_ID", inplace=True)
        self.update()

    # Returns a list with all dicom id values
    def get_dicom_list(self):
        return list(self.man_seg_dict["Dicom_ID"].values())

    def get_dicom_id_index(self, dicom_id):

        # Loop through and find key
        for key, value in self.man_seg_dict['Dicom_ID'].items():
            if str(dicom_id) == str(value):
                return key

        # Dicom ID wasnt found, create a new line
        raise Exception("Could not find Dicom ID in manual segmentiation file")

    def get_file_path(self):
        return self.file_path

    def get_rotation(self, dicom_id, man_seg=True):
        dicom_id_index = self.get_dicom_id_index(dicom_id)
        rot_coloumn_prefix = "Manual_Angle_" if man_seg else "Automatic_Angle_"
        rot_transverse = self.man_seg_dict[rot_coloumn_prefix + "Transverse"][dicom_id_index]
        rot_sagital = self.man_seg_dict[rot_coloumn_prefix + "Sagital"][dicom_id_index]
        return rot_transverse, rot_sagital

    def get_ring_dim(self, dicom_id):
        dicom_id_index = self.get_dicom_id_index(dicom_id)
        ring_dim_small = self.man_seg_dict["Ring_Dim_Radius_Small"][dicom_id_index]
        ring_dim_large = self.man_seg_dict["Ring_Dim_Radius_Large"][dicom_id_index]
        return ring_dim_small, ring_dim_large

    def get_ring_position(self, dicom_id, man_seg=True):
        dicom_id_index = self.get_dicom_id_index(dicom_id)
        pos_coloumn_prefix = "Manual_Position_" if man_seg else "Automatic_Position_"
        return [self.man_seg_dict[pos_coloumn_prefix + axis][dicom_id_index]for axis in ["Z", "X", "Y"]]

    def set_position(self, dicom_id, position, man_seg=False):
        dicom_id_index = self.get_dicom_id_index(dicom_id)
        pos_coloumn_prefix = "Manual_Position_" if man_seg else "Automatic_Position_"
        axis = ["Z", "X", "Y"]
        for i in range(len(axis)):
            self.csvfile.at[dicom_id_index, pos_coloumn_prefix + axis[i]] = position[i]

        # Update after changing the dataframe
        self.update()

    def set_rotation(self, dicom_id, rotation, man_seg=False):
        dicom_id_index = self.get_dicom_id_index(dicom_id)
        pos_coloumn_prefix = "Manual_Rotation_" if man_seg else "Automatic_Rotation_"
        axis = ["Transverse", "Sagital"]
        for i in range(len(axis)):
            self.csvfile.at[dicom_id_index, pos_coloumn_prefix + axis[i]] = rotation[i]

        # Update after changing the dataframe
        self.update()

    def update(self):
        self.csvfile.to_csv(self.file_path)
        self.loadCSV()


def run(file_path):
    # Load data
    man_seg_obj = ManualSegmentation(file_path_csv=file_path)

    print(man_seg_obj.get_dicom_list())

    # Get Rotation
    rot = man_seg_obj.get_rotation(dicom_id=130113)
    print("Rotation: " + str(rot))

    # Get Ring size
    r_dim = man_seg_obj.get_ring_dim(dicom_id=130113)
    print("Ring Dimensions: " + str(r_dim))

    # Get Position
    pos = man_seg_obj.get_ring_position(dicom_id=130113)
    print("Position: " + str(pos))

    # Set position
    # man_seg_obj.set_position(dicom_id=130113, position=[-1, -1, -1])
    
    # Append testvalue
    man_seg_obj.add_dicomid(dicom_id=200000)


# Tests for the data import
if __name__ == "__main__":
    file_pth = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/Manual_Segmentation" \
                "/Manual_Segmentation_Results.csv"
    run(file_path=file_pth)
