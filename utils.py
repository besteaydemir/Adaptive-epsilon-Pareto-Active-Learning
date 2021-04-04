import numpy as np


def dominated_by(v1, v2, epsilon=None):  # Weakly
    # print("dom by")
    # print(v1)
    # print(v2)
    v11 = np.asarray(v1).reshape(-1,1)
    v22 = np.asarray(v2).reshape(-1,1)
    # print(v11)
    # print(v22)
    # print(epsilon)


    if epsilon is not None:
        #print(v11 <= v22 + epsilon)
        return np.all(v11 <= v22 + epsilon)
    return np.all(v1 <= v2)


def printl(list):
    print("~~~~List begin~~~~~~~~~~~~~~~~~~~~~~~~~~" , len(list))
    for node in list:
        print(node)
    print("~~~~List end~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")