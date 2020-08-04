import matplotlib.pyplot as plt


def plot_ring_crosscut(dicomfile, image_type="normal", array_type="base"):
    ringpos = dicomfile.getringcoordinates()
    array = dicomfile.get_image(image_type=image_type).get_array(array_type=array_type)
    cross_array = array[ringpos[0], ringpos[2], :]

    plt.plot(cross_array)
    plt.title("Pixel Intensity")
    plt.grid(True)
    plt.xlabel('Horizontal Pixel Index')
    plt.ylabel('Relative Slice Overlap / Intensity')
    plt.show()
