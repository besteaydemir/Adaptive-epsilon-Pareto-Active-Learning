"""
Represents a hypercube context region.
"""


class Hypercube:
    def __init__(self, length, center):
        self.center = center
        self.length = length

    def is_pt_in_hypercube(self, point):
        # Translate the hypercube and the point
        for coordinate in (point - self.center):
            if abs(coordinate) > self.length / 2:
                return False
        return True

    def get_dimension(self):
        return len(self.center)