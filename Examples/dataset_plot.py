from paretoset import paretoset
import pandas as pd


# Generate the function lists
from utils_plot import plot_pareto_front

df = pd.read_csv("data.txt", sep=';', header = None)
x_vals = df[[0,1,2]].to_numpy()
y = df[[3,4]].to_numpy()


# Visualize the functions (two functions)
title1 = "Objective 1$"
#title1 = "Six-Hump Camel Back (Neg)"
title2 = "Objective 2"
#func_val1, func_val2 = plot_func_list(func_list, (0, 1), (0, 1), title1, title2)

# Plot pareto front (two functions)
hotels = pd.DataFrame({"price": y[:,0], "distance_to_beach": y[:,1]})
mask = paretoset(hotels, sense=["max", "max"])
plot_pareto_front(y[:,0], y[:,1], mask)