import numpy as np
from Hypercube import Hypercube
from typing import List
from math import sqrt
from statistics import mean

from Hyperrectangle import Hyperrectangle


class Node:
    hypercube_list: List[Hypercube]

    def __init__(self, parent_node, h, hypercube_list, R_t):
        self.parent_node = parent_node
        if parent_node is None:
            self.parent_node = self
        self.h = h
        self.hypercube_list = hypercube_list
        self.dimension = self.hypercube_list[0].get_dimension()
        self.R_t = R_t  # Cumulative confidence hyper-rectangle of the node

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

            return [Node(self, self.h + 1, new_hypercubes[:int(num_new_hypercubes / 2)], self.R_t),
                    Node(self, self.h + 1, new_hypercubes[int(num_new_hypercubes / 2):], self.R_t)]
        else:
            return [Node(self, self.h + 1, self.hypercube_list[:int(len(self.hypercube_list) / 2)], self.R_t),
                    Node(self, self.h + 1, self.hypercube_list[int(len(self.hypercube_list) / 2):], self.R_t)]

    def get_center(self):
        return np.mean(np.array([hypercube.center for hypercube in self.hypercube_list]), axis=0).reshape(
            (1, self.dimension))

    def printhyper(self):
        print(self.hypercube_list)

    def __eq__(self, other):
        return self.hypercube_list == other.hypercube_list

    def __str__(self):
        name = "Node: \nDepth: " + str(self.h) + " \nHyperrectangle: " + self.R_t.__str__() + '\nHypercubes: '
        for cube in self.hypercube_list:
            name += cube.__str__() + '\n'
        return name

    # def __hash__(self):
    #     return hash(())

    def update_cumulative_conf_rect(self, mu, sigma, mu_parent, sigma_parent, beta, V_h, V_h_1):  # B
        # High probability lower and upper bound, B
        #print("conf")
        #print(mu)
        #print(sigma)
        term1 = mu - sqrt(beta) * sigma
        term2 = mu_parent - sqrt(beta) * sigma_parent - V_h_1
        #print(term1,term2)
        B_lower = np.maximum(term1, term2)
        #print(B_lower)

        term1 = mu + sqrt(beta) * sigma
        term2 = mu_parent + sqrt(beta) * sigma_parent + V_h_1
        B_upper = np.minimum(term1, term2)

        # Upper index of node in all objectives, L and U
        L = B_lower - V_h
        U = B_upper + V_h

        # Confidence hyper-rectangle, Q
        Q = Hyperrectangle(list(L.reshape(-1)), list(U.reshape(-1)))

        # Cumulative confidence hyper-rectangle, R
        self.R_t = self.R_t.intersect(Q)

    # def gp_inference(self, gp):
    #     self.mu, self.sigma = gp.predict_y(self.return_center())
    #     return self.mu, self.sigma

    # with also gp list
