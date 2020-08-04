import os
from pathlib import Path

default_data_folder = "/RaMData"
default_code_fodler = "/RaMCode"
default_output_folder = "/RaMOutput"

class PathObject:

    windows = False
    relative_path: str = ""
    absolute_path: str = ""

    def init(self, relative_path):
        self.relative_path = prepare_path(relative_path)
        self.windows = os.name == "nt"

    def get_relative_path(self):
        return self.relative_path

    def get_full_path(self):
        return self.absolute_path

    def get_windows_path(self):
        


def prepare_path(path_str: str = ""):
    path_str = path_str.replace("\/", "\\")
    path_str = path_str if path_str[0] == "\\" else path_str[1:]
    return path_str


def get_full_path(subfolder)