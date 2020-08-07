###########################################################
# A class of utilities for user input
###########################################################
from RaMCode.Utils import General as general
from pathlib import Path as path
from RaMCode.DataIO.GeneralUtils import PathObject as pathobj


######## Helper functions #################################

def merge_string_list(value_list: list, sep=", "):
    return_string = ""
    for num_strings in range(len(value_list) - 1):
        return_string += "\"" + str(value_list[num_strings]) + "\"" + sep
    return return_string + "\"" + str(value_list[-1]) + "\""


# Quick standardized format for the text to screen
def format_message(message: str):
    return message + " " if not message and message[-1] != " " else message


def get_error_message(message= "Input not recognized", options=None, sep=", "):
    error_msg = "[ERROR] " + message

    if options is not None:
        error_msg += merge_string_list(options, sep=sep)

    return error_msg


def get_string(message: str = ""):
    # Print text message if given
    print(format_message(message), end="")

    # Get the input from the command line
    user_input = input()

    if user_input:
        return user_input
    else:
        print(get_error_message(message="No input detected"))
        return get_string()


######## Input functions #################################

def get_boolean(message: str = "", options: list = ["yes", "no"]):
    # Define possible values for the return
    possible_values_true = ["yes", "true", "y"]
    possible_values_false = ["no", "false", "n"]

    user_input = get_string(message=message).lower()

    if user_input in possible_values_true:
        return True
    elif user_input in possible_values_false:
        return False
    else:
        print(get_error_message(options=options, sep=" or "))
        return get_boolean(message=message)


def get_number(message: str = ""):

    user_input = get_string(message=message)

    try:
        # Get the input from the command line
        value = int(user_input)
        return value
    except ValueError:
        pass

    try:
        # Get the input from the command line
        value = float(user_input)
        return value
    except ValueError:
        pass

    print(get_error_message())
    return get_number(message=message)


def get_directory(message: str = "", print_file_list=False):

    # Get user input
    user_input = get_string(message=message)

    # Create a path object
    path_obj = pathobj(user_input)

    if not path_obj.exists():
        print(get_error_message(message="Path doesn\'t exist"))

    else:
        test = path_obj.get_path(path_type="full")
        if print_file_list:
            for file in path_obj.get_file_list():
                print(file)
        return test


# Possible list types are int, float or str
def get_list(message: str = ""):

    # Get user input
    user_input = get_string(message=message)

    # Split into list elements
    if not user_input:
        print(get_error_message(message="Please enter values..."))
        return get_list(message=message)
    else:
        if "," in user_input:
            input_list = user_input.split(",")
        elif "&" in message:
            input_list = user_input.split("&")
        else:
            input_list = user_input.split(" ")

    # Read Number in elements
    try:
        return_list = [int(x) for x in input_list]
        return return_list
    except ValueError:
        pass

    try:
        return_list = [float(x) for x in input_list]
        return return_list
    except ValueError:
        pass

    return input_list


if __name__ == "__main__":
    general.print_divider("Testing Interaction Functions", spacers=2)

    # print(get_boolean(message="Boolean test: ", options=["yes", "no"]))
    # print(get_number(message="Number test: "))
    # print(get_directory(message="Folder test: ", print_file_list=True))

    for i in range(4):
        print(get_list(message="List test: "))
