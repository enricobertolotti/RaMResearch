import numpy as np
from skimage.feature import match_template
from scipy import ndimage as ndi
from pathlib import Path
import copy
from scipy.ndimage.interpolation import rotate


class RingPoint:
    abs_pos = np.empty(3)
    sph_dir = np.empty(3)
    cart_eig_vec = np.empty(3)
    image_gradient = []

    def __init__(self, abs_pos):
        self.abs_pos = abs_pos

    def calc_eig_vec(self):
        pass


class RingImage:

    # Image Data
    image = None

    # Ring positional & rotational properties
    r_large = 0
    r_small = 0
    ring_angle = 0

    isfilled: bool = None

    def __init__(self, image, r_small, r_large, ring_rotatation, isfilled):
        self.image = image
        self.r_small = r_small
        self.r_large = r_large
        self.ring_angle = ring_rotatation
        self.isfilled = isfilled

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def get_r_small(self):
        return self.r_small

    def get_r_large(self):
        return self.r_large

    def set_dimensions(self, r_large=-1, r_small=-1):
        self.r_large = self.r_large if r_large == -1 else r_large
        self.r_small = self.r_small if r_small == -1 else r_small

    def get_rotation(self):
        return self.ring_angle

    def set_rotation(self, rotation):
        self.ring_angle = rotation

    def set_is_filled(self, filled):
        self.isfilled = filled

    def get_isfilled(self):
        return self.isfilled

    def get_cropped_image(self, crop_dim):

        # Define Values for both functions:
        image = self.image
        cross_cut_offset = (0, 0, -self.r_large)

        if isinstance(crop_dim[0], tuple):
            pad_amount = 256
            pad_amount = ((pad_amount, pad_amount), (pad_amount, pad_amount), (pad_amount, pad_amount))
            image = np.pad(image, pad_amount, mode="constant", constant_values=0)
            c_d = crop_dim
            r_c = np.add(np.divide(image.shape, 2).astype(np.int), (0, 0, -self.r_large))      # Ring Center
            cropped_image = image[r_c[0]-c_d[0][0]:r_c[0]+c_d[0][1], r_c[1]-c_d[1][0]:r_c[1]+c_d[1][1],
                                  r_c[2]-c_d[2][0]:r_c[2]+c_d[2][1]]
        else:
            crop_dim_adjusted = np.empty(3)

            # Loop through and set the crop_dim to the image_size if not set (-1)
            for i in range(len(crop_dim)):
                crop_dim_adjusted[i] = image.shape[i] if crop_dim[i] == -1 else crop_dim[i]

            # Pad the image to avoid image cutouts
            pad_x = int(crop_dim_adjusted[2] / 2)
            pad_amount = ((0, 0), (0, 0), (pad_x, pad_x))
            image = np.pad(image, pad_amount, mode="constant", constant_values=0)

            # Get the centerpoint of the ring in the image
            image_dim = image.shape
            image_cp = np.divide(image_dim, 2).astype(np.int)

            z_cp, x_cp, y_cp = np.add(image_cp, cross_cut_offset)

            # Calculate the crop amount
            dz, dx, dy = np.divide(crop_dim_adjusted, 2).astype(np.int)
            cropped_image = image[z_cp - dz:z_cp + dz, x_cp - dx:x_cp + dx, y_cp - dy:y_cp + dy]

        # Return the cropped image
        return cropped_image

    def save(self):
        export_ring_array(self.image, self.r_small, self.r_large, self.ring_angle)


class RingPointCloud:

    # Define Ring Properties
    crosscut_point = np.empty(3)
    radius_large = 0
    radius_small = 0
    image_size = 0

    # Define Reference Circle
    # ref_circle = np.empty((360, 3))
    circles = []

    # Define Point Array
    point_cloud_outline = []
    point_cloud_filled = []

    # Images stored as RingImage
    image_outline: RingImage = None
    image_filled: RingImage = None

    def __init__(self, lg_radius, sm_radius):
        self.radius_large = np.max([lg_radius, sm_radius])
        self.radius_small = np.min([lg_radius, sm_radius])
        self.ref_circle = [[self.radius_large, theta, 0] for theta in range(360)]
        self.create_filled()
        self.crosscut_point = [0, 0, -self.radius_large]

    def create_hull(self):
        self.point_cloud_outline = create_outline(self.radius_large, self.radius_small)

    def create_filled(self):
        # self.point_cloud_filled += copy.deepcopy(self.point_cloud_outline)
        for r_small in range(self.radius_small):
            if r_small % int(self.radius_small / 2) == 0:
                self.point_cloud_filled += create_outline(self.radius_large, r_small)

    def get_image(self, outline=True, morph_operations=True, crop_dim=(-1, -1, -1), angle=0):

        # See if images can be loaded
        angle %= 180        # Images 180 degrees apart are equal to each other
        loaded_images = import_ring_image(self.radius_small, self.radius_large, angle, filled=not outline)
        self.image_outline = loaded_images if outline else None
        self.image_filled = loaded_images if not outline else None

        # Set image object for all consecutive operations
        image_obj = self.image_outline if outline else self.image_filled

        # If no existing image was found, create a new one with all rotations
        if image_obj is None:
            image_obj = create_ring_image(self, outline=outline, morph_operations=morph_operations)
            generate_all_rotations(r_small=image_obj.r_small, r_large=image_obj.r_large, anglerange=angle)

        # Crop image if necessary and return
        if crop_dim != (-1, -1, -1):
            image_array = image_obj.get_cropped_image(crop_dim=crop_dim)
            return RingImage(image_array, image_obj.r_small, image_obj.r_large,
                             image_obj.ring_angle, isfilled=False)
        else:
            return image_obj


def create_ring_image(ringcloud: RingPointCloud, outline=True, morph_operations=True,
                      view_crosscut=True):

    r_max = ringcloud.radius_large + ringcloud.radius_small
    imsize = int(np.floor(r_max * 2.5))
    bg_image = np.zeros((imsize, imsize, imsize))

    cp = np.divide(bg_image.shape, 2)

    cloud = ringcloud.point_cloud_outline if outline else ringcloud.point_cloud_filled

    for pt_circle in cloud:
        for pt in pt_circle:
            pos = np.add(pt.abs_pos, cp).astype(np.int16)
            bg_image[pos[0]][pos[1]][pos[2]] = 255

    return_image = bg_image.astype(np.uint8)

    # If the image should be morphologically closed
    if morph_operations:
        struct = ndi.generate_binary_structure(3, 3)
        iterations = 3 if outline else np.floor(ringcloud.radius_small / 2).astype(np.uint8)
        if outline:
            return_image = ndi.binary_dilation(return_image, iterations=1)

        return_image = ndi.binary_closing(return_image, structure=struct, mask=None, iterations=iterations)*255

    if view_crosscut:
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    pos = np.add(ringcloud.crosscut_point, cp).astype(np.int16)
                    return_image[pos[0] + i][pos[1] + j][pos[2] + k] = int(outline)*255

    # Create the RingImage object
    ring_im_obj = RingImage(return_image.astype(np.uint8), ringcloud.radius_small, ringcloud.radius_large,
                            ring_rotatation=0, isfilled=not outline)
    ring_im_obj.save()

    return ring_im_obj


def create_outline(large_r, small_r):
    point_cloud = []
    r_b = large_r
    r_s = small_r
    for dz in range(-r_s, r_s + 1):
        dx = np.sqrt(r_s**2 - dz**2)
        for r in [(r_b - dx), (r_b + dx)]:
            point_ring = []
            for theta in range(360):
                p_x = np.cos(theta * np.pi/180) * r
                p_y = np.sin(theta * np.pi/180) * r
                p_z = dz
                point_ring.append(RingPoint([p_x, p_z, p_y]))
            point_cloud.append(copy.deepcopy(point_ring))
    return copy.deepcopy(point_cloud)


# Returns the result of template matching the two arrays
def crs_correlate(array1, array2):
    correlation_res = match_template(array1, array2)
    return correlation_res


# Returns a list of transposed arrays
def view_all_axis(array):
    return [array, array.copy().transpose((2, 1, 0)), array.copy().transpose((1, 2, 0))]


def create_rotated_image(base_ring_image: RingImage, rotation):

    # Get the dimensions of the ring in the base_ring_image
    r_small = base_ring_image.get_r_small()
    r_large = base_ring_image.get_r_large()
    # Compute the amount of rotation required
    angle = (rotation - base_ring_image.get_rotation()) % 180
    filled = base_ring_image.get_isfilled()

    # Check if the image already exists
    imported_ring = import_ring_image(r_small, r_large, angle)

    if not filled and imported_ring is not None:
        # The image already exists, return it
        return imported_ring
    elif filled and imported_ring is not None:
        # The image already exists, return it
        return imported_ring

    # Otherwise image doesnt exist, therefore create it
    # Create a copy of the base_ring_image
    image = copy.deepcopy(base_ring_image.get_image())
    rotated_im = rotate(image, angle, axes=(1, 0), order=0, reshape=False)

    # Create a new RingImage object and save it
    ring_im_obj = RingImage(rotated_im, r_small, r_large, angle, base_ring_image.get_isfilled())
    ring_im_obj.save()

    # Return the rotated image
    return ring_im_obj


# Returns a RingImage object if image exists, else returns None
def import_ring_image(r_small, r_large, ring_angle, filled=True):
    default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/Numpy_Ring_Definitions"
    save_path = default_folder + '/' + str(r_small) + '/' + str(r_large) + '/' + str(ring_angle)
    file_name = str(r_small) + '_' + str(r_large) + '_' + str(ring_angle) + '_'

    file_path = save_path + '/' + file_name
    file_path += "filled.npz" if filled else "outline.npz"

    file_path_pth = Path(file_path)     # Create a path object with the filepath

    if file_path_pth.exists():
        print("Loading Image With: R-Small: " + str(r_small))
        print("                    R-Large: " + str(r_large))
        print("                    R-Angle: " + str(ring_angle))

        image = np.load(file_path)['data'].astype(np.int8) * 255
        print("Image loaded")
        return RingImage(image, r_small, r_large, ring_angle, isfilled=filled)
    else:
        print("Ring image with does not yet exist with the properties:")
        print("R-Small:\t\t" + str(r_small) + "\nR-Large:\t\t" + str(r_large) + "\nAngle:\t\t" + str(r_large))
        return None


# Ring dimensions is stored as
def export_ring_array(array, r_small, r_large, ring_angle, array_type="filled"):

    # Translate Ring array to boolean array
    array = array.astype('bool')

    default_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/Numpy_Ring_Definitions"
    save_path = default_folder + '/' + str(r_small) + '/' + str(r_large) + '/' + str(ring_angle)

    # Create path if necessary
    base_path = Path(save_path)
    base_path.mkdir(parents=True, exist_ok=True)

    file_prefix = str(r_small) + '_' + str(r_large) + '_' + str(ring_angle) + '_'

    if array_type == "filled":
        np.savez_compressed(save_path + '/' + file_prefix + "filled", data=array)
    elif array_type == "outline":
        np.savez_compressed(save_path + '/' + file_prefix + "outline", data=array)
    else:
        raise Exception("Array type not recognized")


def generate_all_rotations(r_small, r_large, anglerange=(1, 90)):
    point_cloud = RingPointCloud(r_large, r_small)
    ring_im_obj = point_cloud.get_image(outline=False, morph_operations=True, angle=0)

    if isinstance(anglerange, int):
        rot_ring_im_obj = create_rotated_image(ring_im_obj, anglerange)
        rot_ring_im_obj.save()
        print("Created Ring Image with Rotation:\t" + str(anglerange))
    elif isinstance(anglerange, (tuple, list, np.ndarray)):
        for i in range(anglerange[0], anglerange[1]):
            rot_ring_im_obj = create_rotated_image(ring_im_obj, i)
            rot_ring_im_obj.save()
            print("Created Ring Image with Rotation:\t" + str(i))
    else:
        raise ValueError("Invalid input type for anglerange")
