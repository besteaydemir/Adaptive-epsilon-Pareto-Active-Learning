from AdaptiveEpsilonPAL import AdaptiveEpsilonPAL
from OptimizationProblem import OptimizationProblem
from GaussianProcessModel import GaussianProcessModel
from Hypercube import Hypercube

func_list = []
problem_model = OptimizationProblem(func_list)

data = [] #Can be generated with opt problem instance for syn. data
datay = []
mean_list = []
kernel_list = []
gp = GaussianProcessModel(data, datay, multi = False, m = 2, kernel_list=kernel_list, mean_list=mean_list)

hyp = Hypercube(2, (3, 3))
pareto_set, pareto_set_cells = AdaptiveEpsilonPAL(problem_model, epsilon = 1, delta=1, beta=1, gp = gp, initial_hypercube = hyp)