import numpy as np
"""
Represents a hypercube context region.
"""


class Hyperrectangle:
    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower
        self.diameter = np.sum((np.array(upper)-np.array(lower))**2)


    def get_dimension(self):
        return len(self.center)

    def intersect(self, rect):
        lower_new = []
        upper_new = []
        for l1, l2 in zip(self.lower, rect.get_lower()):
            #print(l1, l2)
            lower_new.append(max(l1, l2))

        for u1, u2 in zip(self.upper, rect.get_upper()):
            #print(u1, u2)
            upper_new.append(min(u1, u2))

        return Hyperrectangle(lower_new, upper_new)

    def get_lower(self):
        return np.array(self.lower)

    def get_upper(self):
        return np.array(self.upper)

    def __str__(self):
        return "Upper: " + str(self.upper) + ", Lower: " + str(self.lower)
