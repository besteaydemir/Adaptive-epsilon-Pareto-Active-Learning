"""
Represents a hypercube context region.
"""
import numpy as np

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

    def __str__(self):
        return "Center: " + str(self.center) + ", Length: " + str(self.length)

    def __eq__(self, other):
        return np.all(other.center == self.center) and other.length == self.length
