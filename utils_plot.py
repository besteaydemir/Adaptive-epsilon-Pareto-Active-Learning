import numpy as np
import matplotlib.pyplot as plt

def plot_func_list(list, range1, range2, title1, title2):
    x1 = np.linspace(range1[0], range1[1], 50)
    x2 = np.linspace(range2[0], range2[1], 50)
    px1, px2 = np.meshgrid(x1, x2)

    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    func_val1 = list[0](np.vstack((px1.flatten(), px2.flatten())).T)
    ax1.scatter3D(px1, px2,  func_val1, c=func_val1, cmap='viridis')
    ax1.set_xlabel('$X_1$')
    ax1.set_ylabel('$X_2$')
    ax1.set_title(title1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    func_val2 = list[1](np.vstack((px1.flatten(), px2.flatten())).T)
    ax2.scatter3D(px1, px2, func_val2, c=func_val2, cmap='viridis')
    ax2.set_xlabel('$X_1$')
    ax2.set_ylabel('$X_2$')
    ax2.set_title(title2)
    plt.show()

    return func_val1, func_val2

def plot_pareto_front(func_val1, func_val2, mask):
    fig = plt.figure(figsize=(8, 5))
    plt.rcParams["figure.dpi"] = 200
    ax = plt.axes()
    ax.scatter(func_val1[mask ==False], func_val2[mask == False], s=8, color='gray')
    ax.scatter(func_val1[mask], func_val2[mask], s=8, color='red')
    ax.set_xlabel('$Objective 1$')
    ax.set_ylabel('$Objective 2$')
    plt.grid(True, linewidth=0.5, color='gray', linestyle='-')
    plt.show()

