from pathlib import Path as path
from PIL import Image, ImageChops, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from BScThesis import DicomScripts as ds, ImageFilters as imfil
import cv2

# For the export filtered images


# Possible filter types are
# binary_threshold, adaptive_threshold
def export_filtered_dicom(dicomfile, image_type="normal", array_type="base", filter_type="", values=[]):
    if "pattern" in filter_type.lower():
        foldername = "pattern"
        name_ext = "pattern"
        if dicomfile.pattern is None:
            dicomfile.createsearchpattern()
        pattern_im = dicomfile.pattern
        save_dicomimage_as_layers(dicomfile, array=pattern_im, imagetype=image_type, arraytype=array_type,
                                  folder=foldername, name_ext=name_ext, transparency=False)
    if "threshold" in filter_type.lower() and "bin" in filter_type.lower():
        for value in values:
            # File name preparations
            foldername = "filtered" + '/' + filter_type + "_" + str(value)
            name_ext = "thresh_" + str(value)
            image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
            filtered_im = imfil.threshholdBinary(image, value)
            save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                      folder=foldername, name_ext=name_ext, transparency=False)
    if "thresh" in filter_type.lower() and "adap" in filter_type.lower():
        for block_size in values:
            # File name preparations
            foldername = "filtered" + '/' + filter_type + "_" + str(block_size)
            name_ext = "adapthresh_" + str(block_size)
            image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
            # Filter Image
            filtered_im = imfil.adaptiveThreshold(image, block_size)
            # Save Image
            save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                      folder=foldername, name_ext=name_ext, transparency=False)
    if "sobel" in filter_type.lower():
        for ksize in values:
            foldername = "filtered" + '/' + filter_type + "_" + str(ksize)
            name_ext = "sobel_" + str(ksize)
            image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
            # Filter Image
            filtered_im = imfil.sobel_dicom(image, ksize=ksize)
            # Save Image
            save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                      folder=foldername, name_ext=name_ext, transparency=False)
    if "prewitt" in filter_type.lower():
        foldername = "filtered" + '/' + filter_type
        name_ext = "prewitt"
        image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
        # Filter Image
        filtered_im = imfil.prewitt_dicom(image)
        save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                  folder=foldername, name_ext=name_ext, transparency=False)
    if "scharr" in filter_type.lower():
        foldername = "filtered" + '/' + filter_type
        name_ext = "scharr"
        image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
        # Filter Image
        filtered_im = imfil.scharr_dicom(image)
        save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                  folder=foldername, name_ext=name_ext, transparency=False)
    if "robert" in filter_type.lower():
        foldername = "filtered" + '/' + filter_type
        name_ext = "roberts"
        image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
        # Filter Image
        filtered_im = imfil.roberts_dicom(image)
        save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                  folder=foldername, name_ext=name_ext, transparency=False)
    if "log" in filter_type.lower():
        for vertical_sigma, log_sigma in values:
            for gaus_1D in [True, False]:
                foldername = "filtered" + '/' + filter_type + "_" + str(vertical_sigma)+ "_" + str(log_sigma) \
                             + "_" + str(gaus_1D)
                name_ext = "LoG" + "_" + str(vertical_sigma)+ "_" + str(log_sigma)  + "_" + str(gaus_1D)
                image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
                filtered_im = imfil.gauslogfilter(image, vertical_sigma, log_sigma, gaus1D=gaus_1D)
                save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                          folder=foldername, name_ext=name_ext, transparency=False)
    if "morph" in filter_type.lower():
        for kernelsize in values:
            foldername = "filtered" + '/' + filter_type + "_" + str(kernelsize)
            name_ext = "morph" + "_" + str(kernelsize)
            image = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
            filtered_im = imfil.morph_dicom(image, kernelsize=kernelsize)
            save_dicomimage_as_layers(dicomfile, array=filtered_im, imagetype=image_type, arraytype=array_type,
                                      folder=foldername, name_ext=name_ext, transparency=False)


def export_dicom_array(dicomfile, which_array=""):
    if "correlation" in which_array.lower():
        array = dicomfile.correlated_array
        save_3D_array_as_image(dicomfile, array, "correlation", "3Dcorr")


def save_3D_array_as_image(dicomfile, array, folder="", name_ext=""):
    save_path = get_export_folder_path(dicomfile)

    if len(folder) > 0:
        save_path += folder + '/'
    else:
        save_path += folder

    # Create directory if needed
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # If its a 2D Image
    if len(array.shape) == 2:
        im = Image.fromarray(array)
        imagename = dicomfile.getname() + "_" + name_ext + ".png"
        im.save(save_path + imagename, format="png")
    else:
        for layer in range(array.shape[0]):
            im = Image.fromarray(array[layer])
            imagename = dicomfile.getname() + "_" + name_ext + "_" + str(layer) + ".png"
            im.save(save_path + imagename, format="png")


def save_dicomimage_as_layers(dicomfile, array=None,
                              imagetype="normal", arraytype="base",
                              folder="", name_ext=""
                              , transparency=False):

    # Get absolute path to folder
    save_path = get_export_folder_path(dicomfile)

    if len(folder) > 0:
        folder = folder + '/'
    else:
        folder = folder

    if len(name_ext) > 0:
        name_ext = "_" + name_ext

    save_path += folder + imagetype + "/" + arraytype + '/'
    iamgename = dicomfile.getname() + "_" + imagetype + "_" + arraytype + name_ext

    # 3D image to be exported to layers
    if array is not None:
        dicom_3D = array
    else:
        dicom_3D = dicomfile.get_image(image_type=imagetype).get_array(array_type=arraytype)

    # Create directory if needed
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    if len(dicom_3D.shape) == 2:
        im = Image.fromarray(array)
        imagename = dicomfile.getname() + "_corpattern.png"
        im.save(save_path + imagename, format="png")
    else:
        # Loop through layers and save individual png files
        for layer in range(dicom_3D.shape[0]):
            layername = iamgename + "_depth" + str(layer) + ".png"
            im = Image.fromarray(dicom_3D[layer])
            if transparency:
                alpha = im.copy()
                im.putalpha(alpha)
            im.save(save_path+layername, format="png")


def get_export_folder_path(dicomfile):
    absolutepath = "/Users/enricobertolotti/PycharmProjects/BScAssignment/Exports/"
    if isinstance(dicomfile, ds.Dicomimage):
        filefoldername = dicomfile.getname()
        return absolutepath + filefoldername + '/'
    else:
        return absolutepath


def save_ring_pos(dicomfile, which_pos="", folder="", filename=""):

    save_path = get_export_folder_path(dicomfile)

    if len(folder) > 0:
        save_path += folder + '/'
    else:
        save_path += folder

    if len(filename.split(".")) <= 1:
        filename += ".txt"

    # Create directory if needed
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    dicomfile_name = dicomfile.getname() + '\n'

    if "center" in which_pos.lower():
        ring_pos = "Ring Position = " + str(dicomfile.getringcoordinates())
    elif "closest" in which_pos.lower():
        ring_pos = "Closest Edge Position = " + str(dicomfile.getclosestedgepoint())
    elif "angle" in which_pos.lower():
        ring_pos = "Detected Ring Angle = " + str(dicomfile.determined_angle)
    elif "fit" in which_pos.lower() and "filter" in which_pos.lower():
        ring_pos = "\nAssumed Fit Filtered = " + str(dicomfile.similarty_func_coeff_filtered)
        ring_pos += '\nPhase:' + str((dicomfile.similarty_func_coeff_filtered[2] * 180 / np.pi)-90)
    elif "fit" in which_pos.lower() and "normal" in which_pos.lower():
        ring_pos = "Assumed Fit Normal = " + str(dicomfile.similarty_func_coeff_normal)
        ring_pos += '\nPhase:' + str((dicomfile.similarty_func_coeff_normal[2] * 180 / np.pi)-90)
    else:
        ring_pos = "Misused function"

    path_to_file = save_path + filename
    with open(path_to_file, "a") as file_object:
        file_object.write(dicomfile_name + ring_pos + '\n')


def save_dicom_contour_analysis(dicom):

    save_path = get_export_folder_path(dicom)
    save_path += "contour_analysis/"

    min_range = dicom.contour_thresh

    if min_range == 0:
        save_path += "all/"
    else:
        save_path += "weightmin_" + str(min_range) + "/"

    folders = ["filtered/", "normal/"]

    # Create directories if needed
    for folder in folders:
        base_path = path(save_path + folder)
        base_path.mkdir(parents=True, exist_ok=True)

    imagename = dicom.getname()

    # Save properties of the contour to text file
    textfilename = dicom.getname() + "_contour_minmax.txt"
    path_to_file = save_path + textfilename
    dicom_contour = dicom.get_max_contour()
    pair = "Pairs: " + str(dicom_contour[0]) + "\n"
    depth = "Depth: " + str(dicom_contour[1]) + "\n"
    weight = "Weight: " + str(dicom_contour[2]) + "\n"
    with open(path_to_file, "a") as file_object:
        file_object.write(pair + depth + weight)

    # Save images
    for (image_array, depth) in dicom.contour_images:
        layername = imagename + "_contour_analysis_" + str(depth) + ".png"
        im = Image.fromarray(image_array[:, :, ::-1])
        im.save(save_path + folders[0] + layername, format="png")

    for (image_array, depth) in dicom.contour_images_normal:
        layername = imagename + "_contour_analysis_norm_" + str(depth) + ".png"
        im = Image.fromarray(image_array[:, :, ::-1])
        im.save(save_path + folders[1] + layername, format="png")


def get_pattern_as_image(dicomfile):
    if dicomfile.pattern is not None:
        return Image.fromarray(dicomfile.pattern)
    else:
        return None


# Image 1 is rgb, image2 is bw
def add_two_images_screen(image1, image2):

    # BGR to RGB
    image1 = image1[:, :, ::-1]

    # Create images
    im1 = Image.fromarray(image1, mode="RGB")
    im2 = Image.fromarray(image2, mode="L")

    debug = (im1.size, im2.size)
    if im1.size[0] > im2.size[0]:
        im2 = ImageOps.pad(im2, im1.size)
    else:
        im1 = ImageOps.pad(im1, im2.size)

    debug2 = ImageChops.screen(im1, im2)

    return debug2


def export_dicom_rotation(dicomfile):

    folder = "rotation_analysis"
    file_ext = dicomfile.getname() + "_rot_analysis"

    save_dicomimage_as_layers(dicomfile, dicomfile.rotation_export[0], imagetype="filtered", arraytype="base",
                              folder=folder + "/all_layers", name_ext="_rot_analysis_filtered"
                              , transparency=False)

    save_dicomimage_as_layers(dicomfile, dicomfile.rotation_export[1], imagetype="normal", arraytype="base",
                              folder=folder + "/all_layers", name_ext="_rot_analysis_standard"
                              , transparency=False)

    save_dicomimage_as_layers(dicomfile, dicomfile.rot_ellipse, imagetype="normal", arraytype="base",
                              folder=folder + "/pattern", name_ext="_rot_analysis_searchpattern"
                              , transparency=False)

    save_ring_pos(dicomfile, "ring_angle", folder=folder, filename="rot_analysis")

    save_data_as_plot(dicomfile, dicomfile.correlation_angle_array, data_type="angle_correlation")

    crsct_w_line_norm = draw_rotation_line(dicomfile, image_type="normal")
    save_rgb_as_image(dicomfile, crsct_w_line_norm, folder=folder, file_ext="rot_analysis_w_line_normal")

    crsct_w_line_fil = draw_rotation_line(dicomfile, image_type="filtered")
    save_rgb_as_image(dicomfile, crsct_w_line_fil, folder=folder, file_ext="rot_analysis_w_line_filtered")


def file_preparations(dicomfile, folder=""):

    save_path = get_export_folder_path(dicomfile)

    if len(folder) > 0:
        save_path += folder + '/'
    elif folder[-1] != '/':
        save_path += folder + '/'
    else:
        save_path += folder

    # Create directory if needed
    base_path = path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Return full path
    return save_path


def save_data_as_plot(dicomfile, data, data_type=""):
    if "angle" in data_type.lower() and "cor" in data_type.lower():
        folder = "rotation_analysis"
        filename = dicomfile.getname()
        file_ext = "_cor_rotation_analysis.png"
        title = "Alpha Rotation Cross Correlation Result"
        x_axis = "Angle [in degrees]"
        y_axis = "Correlation Coefficient"
        x_data = np.linspace(data[0][0], data[0][1], len(data[1]))
        y_data = data[1]

    # File preparations
    full_path = file_preparations(dicomfile, folder=folder)

    # Creating plot
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)

    ax.set(xlabel=x_axis, ylabel=y_axis,
           title=title)
    ax.grid()

    fig.savefig(full_path + filename + file_ext)


def draw_rotation_line(dicomfile, image_type="normal"):

    # Angle in deg
    def calcstartandstop(imagesize, rot_angle):
        center = (int(imagesize[0]/2), int(imagesize[1]/2))
        dx = center[0]
        dy = int(np.sin(rot_angle * np.pi / 180) * dx)
        return (0, center[1]-dy), (imagesize[0], center[1]+dy)

    depth = dicomfile.determined_angle[1][1][0]
    angle = dicomfile.determined_angle[0]

    if "normal" in image_type.lower():
        image = dicomfile.contour_ring_top_view_normal[depth]
    else:
        image = dicomfile.contour_ring_top_view_filtered[depth]

    points = calcstartandstop(image.shape, angle)

    imagergb = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    cv2.line(imagergb, points[0], points[1], color=(255, 0, 0), thickness=1)

    return imagergb


def save_rgb_as_image(dicomfile, array, folder="", file_ext=""):

    full_path = file_preparations(dicomfile, folder)
    filename = dicomfile.getname() + file_ext + ".png"

    im = Image.fromarray(array)
    im.save(full_path + filename, format="png")


def save_rotation_slices(dicomfile, image_type="normal"):
    folderglobal = "rotation_analysis/slice_comparison/"
    folder_imtype = image_type + '/'
    folder_list = ["rot_slice_horizontal", "rot_slice_vertical", "rot_slice_combined", "rot_slice_multiplied"]
    save_path_list = []

    if "normal" in image_type.lower():
        slice_bank = dicomfile.slice_rot_list_export_normal
    else:
        slice_bank = dicomfile.slice_rot_list_export_filter

    for folder in folder_list:
        save_path_list.append(file_preparations(dicomfile, folderglobal+folder_imtype+folder))

    for (rot, slice_v, slice_h) in slice_bank:
        h_filename = dicomfile.getname() + "slice_h_rot_" + str(rot) + ".png"
        im_h = Image.fromarray(slice_h)
        im_h = im_h.transpose(Image.ROTATE_90)
        im_h.save(save_path_list[0] + h_filename, format="png")

        v_filename = dicomfile.getname() + "slice_v_rot_" + str(rot) + ".png"
        im_v = Image.fromarray(slice_v)
        im_v = im_v.transpose(Image.ROTATE_90)
        im_v.save(save_path_list[1] + v_filename, format="png")

        c_filename = dicomfile.getname() + "slice_c_rot_" + str(rot) + ".png"
        im_c = Image.fromarray(np.concatenate((slice_h, slice_v), axis=0))
        im_c = im_c.transpose(Image.ROTATE_90)
        im_c.save(save_path_list[2] + c_filename, format="png")

        m_filename = dicomfile.getname() + "slice_m_rot_" + str(rot) + ".png"
        im_m = ImageChops.multiply(im_h, im_v)
        im_m.save(save_path_list[3] + m_filename, format="png")


def export_multiple_plots(dicomfiles, data_type="Intensity_Variation", image_type="normal", files_used=[]):
    if "intesity" in data_type.lower() and "var" in data_type.lower():
        folder = "common"
        filename = "intensity_variation_comparison_" + image_type
        file_ext = ".png"
        base_title = "Intensity Variation Across Ultrasound Images - "
        title_ext = "Unfiltered" if "normal" in image_type.lower() else "Filtered"
        title = base_title + title_ext
        x_axis = "Pixel Location"
        y_axis = "Pixel Intensity"

    # File preparations
    full_path = file_preparations(folder, folder=folder)

    # Creating plot
    fig, ax = plt.subplots()

    legend = []
    for name in files_used:
        dicomfile = [dicomfile for dicomfile in dicomfiles if name in dicomfile.getname().lower()][0]
        y_data = dicomfile.get_image(image_type=image_type).get_array(array_type="ring crosscut")
        x_data = range(0, len(y_data))
        ax.plot(x_data, y_data)
        legend.append(name)

    ax.set(xlabel=x_axis, ylabel=y_axis, title=title)
    ax.legend(legend)
    ax.grid()

    fig.savefig(full_path + filename + file_ext)