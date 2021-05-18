import numpy as np
import matplotlib.pyplot as plt
import time

from Node import Node
from Hyperrectangle import Hyperrectangle

from utils import dominated_by
from utils_set import set_diff, pess


class AdaptiveEpsilonPAL:

    def __init__(self, problem_model, epsilon, delta, gp, initial_hypercube):

        self.problem_model = problem_model
        self.epsilon = epsilon
        self.delta = delta
        self.gp = gp

        self.t = 1  # Total number of iterations
        self.tau = 0  # Number of evaluation rounds

        self.p_t = []  # Predicted epsilon accurate Pareto set of nodes at round t
        self.s_t = [Node(None, 0, [initial_hypercube], Hyperrectangle([-np.inf]*problem_model.m, [np.inf]*problem_model.m))]  # Undecided set of nodes at round t

        self.V = [self.find_V(0)]
        self.beta = [0]
        self.hmax = 0
        self.t_tau = []

        self.time_elapsed = 0


    def algorithm(self, verbose=False, progress_plots=False, titles=None, vis=False):

        t1 = time.time()

        sigmas = np.ones((1500,))
        sigmabeta = np.ones((1500,))
        Vt = np.ones((1500,))
        conf_diameter = np.ones((1500,))

        while self.s_t:  # While s_t is not empty

            if verbose is True:
                print("t: %3d, tau: %3d, S_t length: %3d, S_t length: %3d" % (self.t, self.tau, len(self.s_t), len(self.p_t)))

            # Active nodes, union of sets s_t and p_t at the beginning of round t
            a_t = self.p_t + self.s_t

            # Pessimistic Pareto set of A_t
            p_pess = pess(a_t)


            "Modeling"

            for node in a_t:
                # Obtain mu_tau and sigma_tau of the node
                mu_tau, sigma_tau = self.gp.inference(node.get_center())

                if len(self.V) <= node.h:
                    self.V.append(self.find_V(node.h))
                    self.hmax += 1

                if node.h == 0:
                    V_h_1 = self.V[node.h]

                else:
                    V_h_1 = self.V[node.h - 1]


                mu_tau_parent, sigma_tau_parent = self.gp.inference(node.parent_node.get_center())


                node.update_cumulative_conf_rect(mu_tau, sigma_tau, mu_tau_parent, sigma_tau_parent,
                                                 self.find_beta(self.t),
                                                 self.V[node.h], V_h_1)


            "Discarding"

            templist = set_diff(self.s_t, p_pess)

            for node in templist:
                for pess_node in p_pess:
                    if dominated_by(node.R_t.upper, pess_node.R_t.lower, self.epsilon):
                        self.s_t.remove(node)
                        break


            # The union of sets St and Pt at the end of the discarding phase of round t
            w_t = self.p_t + self.s_t


            "Epsilon Covering"

            for node in self.s_t:
                belongs = True
                for w_node in w_t:
                    if dominated_by(node.R_t.lower, w_node.R_t.upper,
                                    -self.epsilon):  # Doesn't belong to O_epsilon and therefore not removed
                        belongs = False
                        break
                if belongs:
                    self.s_t.remove(node)
                    self.p_t.append(node)


            "Refining / Evaluating"

            if self.s_t:  # If s_t is not empty

                # Find the most uncertain node
                unc_node_ind = np.argmax(np.array([node.R_t.diameter for node in w_t]))
                unc_node = w_t[unc_node_ind]
                conf_diameter[self.t] = unc_node.R_t.diameter

                mu_unc, sigma_unc = self.gp.inference(unc_node.get_center())

                sigmas[self.t] = np.linalg.norm(sigma_unc)
                sigmabeta[self.t] = np.sqrt(self.find_beta(self.t)) * np.linalg.norm(sigma_unc)
                Vt[self.t] = self.V[unc_node.h] * np.sqrt(self.problem_model.m)

                condition = np.sqrt(self.find_beta(self.t)) * np.linalg.norm(sigma_unc) <= self.V[unc_node.h] * np.sqrt(self.problem_model.m)  # Norm V_h vector


                if condition and unc_node in self.s_t:
                    self.s_t.remove(unc_node)
                    repro = unc_node.reproduce()
                    self.s_t = self.s_t + repro

                elif condition and unc_node in self.p_t:
                    self.p_t.remove(unc_node)
                    self.p_t = self.p_t + unc_node.reproduce()

                else:
                    y = self.problem_model.observe(unc_node.get_center(), std=self.problem_model.obs_noise)

                    self.gp.update(unc_node.get_center(), y)
                    self.t_tau.append(self.t)
                    self.tau += 1


                    if vis is True:
                        self.problem_model.plot_gp_1d(rang=(0,1), gp=self.gp, x=unc_node.get_center(), y=y)


            self.t += 1


        if progress_plots is True:
            plt.figure()
            ax = plt.axes()
            ax.plot(range(1,self.t-1), sigmas[1:self.t-1], label = r'$||\sigma_{\tau}(x_{h_t,i_t})||_2$')
            ax.scatter(self.t_tau, sigmas[self.t_tau], color = 'red', s=6, label=r"$\tau$")
            ax.set_xlabel('$t$')
            ax.set_ylabel(r'$||\sigma_{\tau}(x_{h_t,i_t})||_2$')
            ax.legend()
            plt.title(r"Posterior Variance after $\tau$ Evaluations")
            plt.savefig(titles+ ".png", bbox_inches='tight')
            plt.show()

            plt.figure()
            ax = plt.axes()
            ax.plot(range(1, self.t - 1), sigmabeta[1:self.t - 1], label=r'$ \beta_{\tau}^{1/2} ||\sigma_{\tau}(x_{h_t,i_t})||_2$')
            ax.plot(range(1, self.t - 1), Vt[1:self.t - 1], label=r'$||V_ht||_2$')
            ax.scatter(self.t_tau, sigmabeta[self.t_tau], color='red', s=6, label=r"$\tau$")
            ax.scatter(self.t_tau, Vt[self.t_tau], color='red', s=6, label=r"$\tau$")
            ax.set_xlabel('$t$')
            ax.set_yscale('log')
            ax.legend()
            plt.title(r"Refine/Evaluate Condition after $\tau$ Evaluations")
            plt.savefig(titles + "other" +  ".png", bbox_inches='tight')
            plt.show()

            plt.figure()
            ax = plt.axes()
            ax.plot(range(1,self.t-1), conf_diameter[1:self.t-1], label='$\omega_t(x_{h_t,i_t})$')
            ax.set_yscale('log')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$\omega_t(x_{h_t,i_t})$')
            plt.axhline(self.epsilon, color='green', label='$\epsilon$')
            ax.legend()
            plt.title("Diameter of the Cumulative Confidence \n Hyper-rectangle of the Most Uncertain Node")
            plt.savefig(titles + "dia" + ".png", bbox_inches='tight')
            plt.show()


        t2 = time.time()
        self.time_elapsed = t2-t1

        pareto_cells = [node.hypercube_list for node in self.p_t]


        return self.p_t, pareto_cells


    def find_beta(self, t):
        """
        The confidence term (with t).

        :param t: t.
        :return: beta_t.
        """
        m = self.problem_model.m  # Number obj. functions.
        card = self.problem_model.cardinality  # Cardinality of the design space.
        delta = self.delta

        return (2 / 9) * np.log(m * card * np.pi** 2 * t ** 2 / (6 * delta))


    def find_beta_tau(self, tau):
        """
        The confidence term (with tau).

        :param tau:
        :return: beta_tau.
        """
        return 2 * np.log(2 * self.problem_model.m * np.pi**2 * 2**9 * (tau+1)**2 / (3 * self.delta))


    def find_V(self, h):
        """
        Calculate V_h in high probability bounds on the variation of f.

        :param h: Depth of a node.
        :return: V_h.
        """
        v_1 = np.sqrt(2)
        rho = 0.5
        alpha = 1

        m = self.problem_model.m
        delta = self.delta
        N = self.problem_model.N
        D_1 = self.problem_model.D_1

        # Constants associated with metric dimension D1
        C_1 = np.sqrt(2 * self.gp.v) / self.gp.L

        C_k = C_1
        C_2 = 2 * np.log(2 * C_1 ** 2 * np.pi ** 2 / 6)
        C_3 = 0.91501 + 2.6945 * np.sqrt(2 * D_1 * alpha * np.log(2))  # eta_1 and eta_2 in the paper.

        if (h == 0):
            log_term = 0
        else:
            log_term = np.log(2 * h ** 2 * np.pi ** 2 * m / (6 * delta))


        term1 = np.sqrt(C_2 + 2 * log_term + h * np.log(N) +
                        np.maximum(0, -4 * (D_1 / alpha) * np.log(C_k * (v_1 * rho ** h) ** alpha))) + C_3
        term2 = 4 * C_k * (v_1 * rho ** h) ** alpha

        return term2 * term1




