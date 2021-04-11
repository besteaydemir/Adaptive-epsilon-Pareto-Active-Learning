import numpy as np
import matplotlib.pyplot as plt

def plot_func_list(list, range1, range2, title1, title2, h = None):
    if h is None:
        x1 = np.linspace(range1[0], range1[1], 11)
        x2 = np.linspace(range2[0], range2[1], 11)
    else:
        x1 = np.linspace(range1[0], range1[1], 2**h)
        x2 = np.linspace(range2[0], range2[1], 2**h)
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

def plot_pareto_front(func_val1, func_val2, mask, y1 = None, y2 = None, title=None):
    fig = plt.figure(figsize=(8, 5))
    #plt.rcParams["figure.dpi"] = 400
    ax = plt.axes()
    ax.scatter(func_val1[mask ==False], func_val2[mask == False], s=6, color='gray')
    ax.scatter(func_val1[mask], func_val2[mask], s=8, color='g')

    func_val1sorted = np.sort(func_val1[mask])
    func_val2sorted = np.sort(func_val2[mask])[::-1]

    for i in range (func_val1[mask].shape[0] - 1):
        x_values = [func_val1sorted [i], func_val1sorted[i]]
        y_values = [func_val2sorted[i], func_val2sorted[i+1]]
        plt.plot(x_values, y_values, color='darkseagreen')

        x_values = [func_val1sorted[i], func_val1sorted[i+1]]
        y_values = [func_val2sorted[i+1], func_val2sorted[i + 1]]
        plt.plot(x_values, y_values, color='darkseagreen')



    if (y1 is not None):
        func_val1sorted = np.sort(y1)
        func_val2sorted = np.sort(y2)[::-1]
        ax.scatter(y1, y2, s=8, color='red')
        for i in range(y1.shape[0] - 1):
            x_values = [func_val1sorted[i], func_val1sorted[i]]
            y_values = [func_val2sorted[i], func_val2sorted[i + 1]]
            plt.plot(x_values, y_values, color='lightcoral')

            x_values = [func_val1sorted[i], func_val1sorted[i + 1]]
            y_values = [func_val2sorted[i + 1], func_val2sorted[i + 1]]
            plt.plot(x_values, y_values, color='lightcoral')

    ax.set_xlabel('$Objective 1$')
    ax.set_ylabel('$Objective 2$')
    plt.title(title)
    plt.grid(True, linewidth=0.5, color='gray', linestyle='-')

    if title is not None:
        plt.title(title)
    plt.show()

