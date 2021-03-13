import numpy as np
"""
Represents a hypercube context region.
"""


class Hyperrectangle:
    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower
        self.diameter = np.sum((upper-lower)**2)

    def is_pt_in_hypercube(self, point):
        # Translate the hypercube and the point
        for coordinate in (point - self.center):
            if abs(coordinate) > self.length / 2:
                return False
        return True

    def get_dimension(self):
        return len(self.center)

    def intersection(self, rect):
        lower_new = []
        upper_new = []
        for l1, l2 in zip(self.lower, rect.get_lower()):
            lower_new.append(max(l1, l2))

        for u1, u2 in zip(self.upper, rect.get_upper()):
            upper_new.append(min(u1, u2))

        return Hyperrectangle(lower_new, upper_new)

    def get_lower(self):
        return self.lower

    def get_upper(self):
        return self.upper
