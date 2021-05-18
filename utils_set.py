import copy
import numpy as np


def pess(node_set):
    """
    The set of Pessimistic Pareto set of nodes of a set of nodes.

    :param node_set: List of Node objects.
    :return: List of Node objects.
    """
    pess_set = []

    for i in range(len(node_set)):
        set_include = True

        for j in range(len(node_set)):
            if j == i:
                continue
            if np.all(node_set[i].R_t.get_lower() <= node_set[j].R_t.get_lower()):
                set_include = False
                break

        if set_include:
            pess_set.append(node_set[i])

    return pess_set


def set_diff(s1, s2):
    """
    Set difference of two sets.

    :param s1: List of Node objects.
    :param s2: List of Node objects.
    :return: List of Node objects.
    """

    tmp = copy.deepcopy(s1)

    for node in s2:
        if node in tmp:
            tmp.remove(node)

    return tmp