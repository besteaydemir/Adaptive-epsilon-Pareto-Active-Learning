import numpy as np
from Hypercube import Hypercube
from typing import List
from math import sqrt
from statistics import mean

from Hyperrectangle import Hyperrectangle


class Node:
    hypercube_list: List[Hypercube]

    def __init__(self, parent_node, h, hypercube_list):
        self.parent_node = parent_node
        self.h = h
        self.hypercube_list = hypercube_list
        self.dimension = self.hypercube_list[0].get_dimension()

        # self.mu = None  # GP Inference on the node
        # self.sigma = None

        self.R_t = None  # Cumulative confidence hyper-rectangle of the node

    def reproduce(self):
        """
        This fun creates N new nodes and assigns regions (i.e. hypercubes) to them.
        :return: A list of the N new nodes.
        """
        if len(self.hypercube_list) == 1:
            new_hypercubes = []
            new_hypercube_length = self.hypercube_list[0].length / 2
            old_center = self.hypercube_list[0].center
            num_new_hypercubes = 2 ** self.dimension
            for i in range(num_new_hypercubes):
                center_translation = np.fromiter(
                    map(lambda x: new_hypercube_length / 2 if x == '1' else -new_hypercube_length / 2,
                        list(bin(i)[2:].zfill(self.dimension))),
                    dtype=np.float)
                new_hypercubes.append(Hypercube(new_hypercube_length, old_center + center_translation))

            return [Node(self, self.h + 1, new_hypercubes[:int(num_new_hypercubes / 2)]),
                    Node(self, self.h + 1, new_hypercubes[int(num_new_hypercubes / 2):])]
        else:
            return [Node(self, self.h + 1, self.hypercube_list[:int(len(self.hypercube_list) / 2)]),
                    Node(self, self.h + 1, self.hypercube_list[int(len(self.hypercube_list) / 2):])]

    def contains_context(self, context):
        for hypercube in self.hypercube_list:
            if hypercube.is_pt_in_hypercube(context):
                return True
        return False

    def get_center(self):
        return mean([hypercube.center for hypercube in self.hypercube_list])


    def printhyper(self):
        print(self.hypercube_list)

    def update_cumulative_conf_rect(self, mu, sigma, mu_parent, sigma_parent, beta, V_h, V_h_1):  # B
        # High probability lower and upper bound, B
        term1 = mu - sqrt(beta) * sigma
        term2 = mu_parent - sqrt(beta) * sigma_parent - V_h_1
        B_lower = np.maximum(term1, term2)

        term1 = mu + sqrt(beta) * sigma
        term2 = mu_parent + sqrt(beta) * sigma_parent + V_h_1
        B_upper = np.minimum(term1, term2)

        # Upper index of node in all objectives, L and U
        L = B_lower - V_h
        U = B_upper + V_h

        # Confidence hyper-rectangle, Q
        Q = Hyperrectangle(L, U)

        # Cumulative confidence hyper-rectangle, R
        if self.R_t is None:
            self.R_t = Q
        else:
            self.R_t = self.R_t.intersect(Q)

    # def gp_inference(self, gp):
    #     self.mu, self.sigma = gp.predict_y(self.return_center())
    #     return self.mu, self.sigma

        #with also gp list
