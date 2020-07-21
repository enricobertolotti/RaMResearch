

# Class for storing simple coordinate point
class Coordinate:

    # coordinate position
    xpos: float = None
    ypos: float = None
    zpos: float = None

    def __init__(self, zyx_coord=None):
        self.reset_params()
        if zyx_coord is not None:
            self.set_coordinates(zyx_coord)

    def reset_params(self):
        self.xpos, self.ypos, self.zpos = None, None, None

    def get_coordinates(self):
        return [self.zpos, self.xpos, self.ypos]

    def get_coordinates_int(self):
        return [int(self.zpos), int(self.xpos), int(self.ypos)]

    def set_coordinates(self, zyx_coord):
        self.zpos, self.ypos, self.xpos = zyx_coord

    def get_xpos(self):
        return self.xpos

    def get_ypos(self):
        return self.ypos

    def get_zpos(self):
        return self.zpos

    def set_xpos(self, xpos):
        self.xpos = xpos

    def set_ypos(self, ypos):
        self.ypos = ypos

    def set_zpos(self, zpos):
        self.zpos = zpos
