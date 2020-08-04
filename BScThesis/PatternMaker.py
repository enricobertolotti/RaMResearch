import cv2
import numpy as np
from BScThesis import ImageFilters as imfil


def ringPatternImage(diameter, pixeldensity, pixthickness):

    def lentopix(length, pixeldensityval):
        pixellength = np.round(length * pixeldensityval)
        return int(pixellength)

    def getlinepoints(length, centerpoint):
        x1 = int(centerpoint[0] - (length / 2))
        x2 = int(centerpoint[0] + (length / 2))
        y = int(centerpoint[1])
        return (y, x1), (y, x2)

    diameterinpixels = lentopix(diameter, pixeldensity)
    imagedim = lentopix(2*diameter, pixeldensity)
    pattern = np.zeros((imagedim, imagedim))

    # Draw left line
    lineleftcenterpoint = (0.5*imagedim, 0.5*(imagedim-diameterinpixels))
    lineleftpoints = getlinepoints(0.5*diameterinpixels, lineleftcenterpoint)
    cv2.line(pattern, lineleftpoints[0], lineleftpoints[1], 255, pixthickness)

    # Draw right line
    linerightcenterpoint = (0.5*imagedim, 0.5*(imagedim+diameterinpixels))
    linerightpoints = getlinepoints(diameterinpixels, linerightcenterpoint)
    cv2.line(pattern, linerightpoints[0], linerightpoints[1], 255, pixthickness)

    return pattern.astype(np.uint8)


def circlePatternImage(diameter, pixeldensity, pixthickness):

    # Speed in silicon is approx 0.7 that of the average in surrounding area
    speed_factor = 0.7

    def lentopix(length, pixeldensityval):
        pixellength = np.round(length / pixeldensityval)
        return int(pixellength)

    diam_adjusted = int(diameter / speed_factor)

    imagedim = lentopix(2 * diam_adjusted, pixeldensity)
    pattern = np.zeros((imagedim, imagedim))

    center = (int(0.5*imagedim), int(0.5*imagedim))
    radius = lentopix(0.5 * diam_adjusted, pixeldensity)

    print(radius)

    cv2.circle(pattern, center, radius, 255, pixthickness)

    pattern = imfil.gausdist2d(pattern, 0, 0.3)

    return pattern.astype(np.uint8)


def create_rotating_ellipse(pix_size_w, pix_size_h, start_angle=-45, stop_angle=45, step=1, outline=False, rgb=False,
                            externalimage=None):

    angles = np.arange(start=start_angle, stop=stop_angle, step=step)
    size = int(np.max([pix_size_w, pix_size_h]))
    center = (int(size / 2), int(size / 2))
    ellipse_axis = (int(pix_size_w / 2), int(pix_size_h / 2))

    if outline:
        thickness = 2
    else:
        thickness = -1

    if rgb:
        color = (0, 0, 255)
        returnarray = np.zeros((size, size, 3))
    else:
        color = 255
        returnarray = np.zeros((size, size))

    if externalimage is not None:
        center = (int(externalimage.shape[0]/2), int(externalimage.shape[1]/2))
        color = (255, 0, 0)
        thickness = 2

    if start_angle == stop_angle:
        cv2.ellipse(returnarray, center=center, axes=ellipse_axis, angle=start_angle + 90,
                    startAngle=0, endAngle=360, color=color, thickness=thickness)
        return returnarray
    else:
        returnarray = np.zeros((len(angles), size, size))
        for angle_index in range(len(angles)):
            cv2.ellipse(returnarray[angle_index], center=center, axes=ellipse_axis, angle=angles[angle_index] + 90,
                        startAngle=0, endAngle=360, color=color, thickness=thickness)

        # ds.imageview3d(returnarray.astype(np.uint8))

        return returnarray.astype(np.uint8)



