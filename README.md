# Adaptive epsilon-PAL
This code implements [Adaptive epsilon-PAL](https://arxiv.org/abs/2006.14061) algorithm. The code performs multi-objective Bayesian optimization and identifies the Pareto set and the Pareto front for the objective functions. For the examples (in Examples and Multiprocess Examples folders), [NoC and SNW data sets](http://www.spiral.net/software/pal.html) are used. In order to run them again, the data sets should be copied to Experiments/Data fodler. 

# Data sets
A data set can be uploaded to Data folder and the algorithm can be run to find the Pareto set. The algorithm does not use all points in the data set, it uses the closest point selected in the algorithm. This is done to simulate evaluation of an arbitrary point.

# To use
Codes in Examples or Multiprocess Examples files can be run to obtain the results. After running the algorithm function, Adaptive epsilon-PAL object returns the Pareto set and the associated cells. The code can be modified to include or omit certain plots.
