import os
from pathlib import Path

default_data_folder = "/RaMData"
default_code_fodler = "/RaMCode"
default_output_folder = "/RaMOutput"
default_man_seg_filepath = "/RaMData/Manual_Segmentation/Manual_Segmentation_Results.csv"


class PathObject:

    windows = False
    relative_path: str = ""
    absolute_path: str = ""

    def __init__(self, relative_path):
        self.windows = os.name == "nt"
        self.relative_path = prepare_path(relative_path)
        self.create_full_path()

    def get_relative_path(self):
        return self.relative_path

    def create_full_path(self):
        posix_path = get_root_path()
        self.absolute_path = posix_path + self.relative_path

    # Path type
    def get_path(self, path_type="full"):
        path = self.absolute_path if "full" in path_type else self.relative_path
        return get_windows_path(path) if self.windows else path

    def exists(self):
        return Path(self.absolute_path).exists()

    def get_folder_list(self):
        return [x[0] for x in os.walk(self.absolute_path) if os.path.isdir(x[0]) and x[0][-1] != "_"]

    def get_file_list(self):
        return [f for f in os.listdir(self.absolute_path) if os.path.isfile(f)]


# Returns true if the operating system is windows
def get_is_windows():
    return os.name == "nt"


# Creates a standard posix path from windows if required
def prepare_path(path_str: str = ""):
    path_str = path_str.replace("\\", "/")
    path_str = path_str if path_str[0] == "/" else "/" + path_str
    return path_str


def get_windows_path(path_str):
    windows_pth = path_str.replace("/", "\\")
    windows_pth = windows_pth[1:] if windows_pth[0] == "\\" else windows_pth
    return windows_pth


def get_root_path(tree_depth=1):
    path = Path.cwd()
    for i in range(tree_depth):
        path = path.parent
    return str(path)


def get_default_file_folder(folder_type="data"):
    if "data" in folder_type.lower():
        full_path = prepare_path(get_root_path()) + default_data_folder
    elif "segmentation" in folder_type.lower():
        full_path = prepare_path(get_root_path()) + default_man_seg_filepath
    else:
        raise Exception("Folder or file doesnt have a default")

    return get_windows_path(full_path) if get_is_windows() else full_path


if __name__ == "__main__":
    print(get_windows_path(get_root_path(3)))
