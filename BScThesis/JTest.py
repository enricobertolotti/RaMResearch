import matplotlib.pyplot as plt
import numpy as np
import cv2


def func(distance, effect_radius, max_shift_pixels):
    offset = int(effect_radius / 2)
    distance += offset
    distance = (distance * 2 * np.pi)
    return ((1 + np.cos((distance / effect_radius))) / 2) * max_shift_pixels


def shift_xy(x_dist, y_dist, radius, max_shift):
    x_shift = func(x_dist, radius, max_shift)
    y_shift = func(y_dist, radius, max_shift)
    return x_shift, y_shift


def get_xy_dist_mouse(x, y):
    return x_mouse - x, y_mouse - y


def get_newframe(num_dots):
    for i in range(30):
        for j in range(50):
            get_xy_dist_mouse();


def showcircles(n):
    image_size = 500, 500
    image = np.zeros(image_size)
    dist = 500
    circ_pos = np.linspace(0, dist, n)

    for pos in circ_pos:
        # pos += func(pos, dist, 40)
        cv2.circle(image, (int(pos), int(image_size[0] / 2)), 3, 255, thickness=-1)

    cv2.imshow(winname="Test", mat=image)
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()


def main():
    distance = 200
    x = np.linspace(0, distance, 100)
    y = [func(x_val, distance, 30) for x_val in x]

    plt.plot(x, y)
    plt.grid()
    plt.show()
    debug = 1


main()
showcircles(20)
