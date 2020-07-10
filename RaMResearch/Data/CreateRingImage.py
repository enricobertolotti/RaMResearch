import numpy as np
from RaMResearch.Utils import Interfaces as intrfce


class Point:
    abs_pos = np.empty(3)
    sph_dir = np.empty(3)
    cart_eig_vec = np.empty(3)
    image_gradient = []

    def __init__(self, abs_pos, sph_dir):
        self.abs_pos = abs_pos
        self.sph_dir = sph_dir

    def calc_eig_vec(self):
        self.cart_eig_vec = coord_sph_to_cart((1, self.sph_dir[1], self.sph_dir[2]))


class RingPointCloud:

    # Define Ring Properties
    crosscut_point = []
    radius_large = 0
    radius_small = 0

    # Define Reference Circle
    ref_circle = np.empty((360, 3))

    # Define Point Array
    point_cloud = None

    def __init__(self, lg_radius, sm_radius):
        self.radius_large = lg_radius
        self.radius_small = sm_radius
        self.ref_circle = [[self.radius_large, theta, 0] for theta in range(360)]
        self.create_hull()

    def create_hull(self):
        pt_array = []
        for theta in range(360):
            # For each angle on the circle
            cart_crss = coord_sph_to_cart(self.ref_circle[theta])

            pt_circle = []
            for phi in range(360):
                if theta > 180:
                    phi = - phi
                sph_vec = [self.radius_small, theta, phi]
                cart_vec = coord_sph_to_cart(sph_vec)
                abs_pos = np.add(cart_crss, cart_vec)
                pt = Point(abs_pos, sph_vec)
                pt_circle.append(pt)

            pt_array.append(pt_circle)

        self.point_cloud = pt_array


def coord_sph_to_cart(sph_coord):
    r = sph_coord[0]
    theta = sph_coord[1] * np.pi / 180  # to radian
    phi = sph_coord[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]


def create_ring_image(ringcloud: RingPointCloud, bg_image=None):

    if bg_image is None:
        r_large = ringcloud.radius_large
        bg_image = np.zeros((r_large*4, r_large*4, r_large*4))

    cp = np.divide(bg_image.shape, 2)

    for theta in range(len(ringcloud.point_cloud)):
        for pt in ringcloud.point_cloud[theta]:
            pos = np.add(pt.abs_pos, cp).astype(np.uint8)
            bg_image[pos[0]][pos[1]][pos[2]] = 255

    return bg_image.astype(np.uint8)


# Function Tests
example_rpc = RingPointCloud(50, 10)
test = create_ring_image(example_rpc)
print(np.max(test), np.min(test))
debug = 1

viewarray = []

intrfce.imageview3d([test.copy().transpose((1, 0, 2)), test.copy(), test.copy().transpose((2, 1, 0))])
