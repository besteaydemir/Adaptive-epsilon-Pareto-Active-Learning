from Node import Node
import numpy as np
from utils import dominated_by


def pess(a):
    pess_set = []
    for i in range(len(a)):
        set_include = True
        for j in range(i + 1, len(a)):
            if np.all(a[i].R_t.get_lower() <= a[j].R_t.get_lower()):
                set_include = False

        if set_include:
            pess_set.append(a[i])

    return pess_set


class AdaptiveEpsilonPAL:
    def __init__(self, problem_model, epsilon, delta, gp, initial_hypercube):
        self.problem_model = problem_model
        self.epsilon = epsilon  # Accuracy level given as input to the algorithm
        self.delta = delta  # ???
        self.gp = gp

        # Initialize
        self.t = 1  # Total number of iterations
        self.tau = 0  # Number of evaluation rounds

        self.p_t = []  # Predicted epsilon accurate Pareto set of nodes at round t
        self.s_t = [Node(None, 0, [initial_hypercube])]  # Undecided set of nodes at round t

        self.V = []
        self.beta = []

    def algorithm(self):
        while self.s_t: # While s_t is not empty
            a_t = self.p_t + self.s_t  # Active nodes, union of sets s_t and p_t at the beginning of round t
            p_pess = pess(a_t)  # Pessimistic Pareto set of A_t

            "Modeling"
            self.beta.append(self.find_beta(self.t))
            for node in a_t:
                # Obtain mu_tau and sigma_tau of the node
                mu_tau, sigma_tau = self.gp.inference(node.get_center())

                if len(self.V) <= node.h:
                    self.V.append(self.find_V(node.h))

                mu_tau_parent, sigma_tau_parent = self.gp.inference(node.parent_node.get_center())
                node.update_cumulative_conf_rect(mu_tau, sigma_tau, mu_tau_parent, sigma_tau_parent, self.beta[self.t], self.V[node.h], self.V[node.h - 1])

            "Discarding"
            for node in list(np.setdiff1d(self.s_t, p_pess)):
                for pess_node in p_pess:
                    if dominated_by(node.R_t.upper, pess_node.R_t.lower, self.epsilon):
                        self.s_t.remove(node)
                        break

            w_t = self.p_t + self.s_t  # The union of sets St and Pt at the end of the discarding phase of round t

            "Epsilon Covering"
            for node in self.s_t:
                belongs = True
                for w_node in w_t:
                    if dominated_by(node.R_t.lower, w_node.R_t.upper, -self.epsilon): #Doesn't belong to O_epsilon and therefore not removed
                        belongs = False
                        break
                if belongs:
                    self.s_t.remove(node)
                    self.p_t.append(node)

            "Refining / Evaluating"
            if self.s_t:  # If s_t is not empty
                unc_node_ind = np.argmax(np.array([node.R_t.diameter for node in w_t]))
                unc_node = w_t[unc_node_ind]
                condition = True #help

                if condition and unc_node in self.s_t:
                    self.s_t.remove(unc_node)
                    self.s_t.append(unc_node.reproduce())
                elif condition and unc_node in self.p_t:
                    self.p_t.remove(unc_node)
                    self.p_t.append(unc_node.reproduce())
                else:
                    y = self.problem_model.observe(unc_node.get_center())
                    # Update GP parameters
                    self.gp = self.gp.update(unc_node.get_center(), y)
                    self.tau += 1

            self.t += 1

        pareto_cells = [node.hypercube_list for node in self.p_t]
        return self.p_t, pareto_cells

    def find_beta(self, tau):
        """
        The confidence term beta_tau as explained in ref. [3].
        Args:
            (int) tau: Number of evaluations performed so far.

        Returns:
            (float) beta: The confidence term.
        """
        m = self.problem_model.m  # Number obj. functions.
        card = self.problem_model.card # Cardinality of the design space.
        delta = self.delta

        return (2/9)*np.log(m * card * np.pi**2 * tau**2 / (6*delta))

    def find_V(self, h):
        v_1 = np.sqrt(2)
        rho = 1/2
        alpha = 1

        m = self.problem_model.m
        delta = self.delta
        N = self.problem_model.N
        D_1 = self.problem_model.D_1

        # Constants associated with metric dimension D1
        C_1 = np.sqrt(2 * self.problem_model.v / self.problem_model.L)
        C_k = C_1
        C_2 = 2 * np.log(2 * C_1 ** 2 * np.pi ** 2 / 6)
        C_3 = 0.91501 + 2.6945 * np.sqrt(2 * D_1 * alpha * np.log(2))  # eta_1 and eta_2 in the paper.

        term1 = np.sqrt(C_2 + 2 * np.log(2 * h**2 * np.pi**2 * m / (6*delta)) + h * np.log(N) +
                        np.maximum(0, -4 * (D_1 / alpha) * np.log (C_k * (v_1 * rho ** h) ** alpha))) + C_3
        term2 = 4 * C_k * (v_1 * rho **h)**alpha
        return term2 * term1