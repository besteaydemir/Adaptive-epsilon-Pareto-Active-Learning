from Node import Node
from Hyperrectangle import Hyperrectangle
import numpy as np
import matplotlib.pyplot as plt
from utils import dominated_by, printl
import copy
import time



def pess(a):
    # print("pesssss")
    # printl(a)
    pess_set = []
    for i in range(len(a)):
        # print("i", i)
        set_include = True
        for j in range(len(a)):
            if j == i:
                continue
            # print(len(a))
            # print("j", j)
            # print(a[i].R_t.get_lower())
            # print(a[j].R_t.get_lower())
            # print(np.all(a[i].R_t.get_lower() <= a[j].R_t.get_lower()))
            if np.all(a[i].R_t.get_lower() <= a[j].R_t.get_lower()):
                set_include = False
                # print("here")
                break

            #print(set_include)

        # print("deciding for the node--------------------")
        # print(set_include)

        if set_include:
            pess_set.append(a[i])
    #     printl(pess_set)
    # printl(a)
    # print("returning")
    return pess_set


def set_diff(s1, s2):
    # printl(s1)
    # printl(s2)
    tmp = copy.deepcopy(s1)
    for node in s2:
        if node in tmp:
            tmp.remove(node)
    # printl(tmp)
    return tmp



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
        self.s_t = [Node(None, 0, [initial_hypercube], Hyperrectangle([-np.inf]*problem_model.m, [np.inf]*problem_model.m))]  # Undecided set of nodes at round t

        self.V = [self.find_V(0)]
        self.beta = [0]
        self.hmax = 0
        self.hmax_len = 0
        self.t_tau = []

        self.time_elapsed = 0


    def algorithm(self):
        np.random.seed(7)
        t1 = time.time()
        tau_change = True # Initial
        sigmas = np.ones((1500,))
        conf_diameter = np.ones((1500,))
        while self.s_t and self.t < 800:  # While s_t is not empty
            print("-------------------------------------------------------------------------------")
            print("tau" , self.tau)
            print("t" , self.t)
            print("hmax", self.hmax)

            print('s_t length')
            print(len(self.s_t))
            print("p_t length")
            print(len(self.p_t))
            # if self.p_t:
            #     printl(self.p_t)
            a_t = self.p_t + self.s_t  # Active nodes, union of sets s_t and p_t at the beginning of round t
            # print("a_t")
            # printl(a_t)
            p_pess = pess(a_t)  # Pessimistic Pareto set of A_t
            print("p_pess(a_t)")
            print(len(p_pess))

            "Modeling"
            print("Modeling")
            #self.beta.append(self.find_beta(self.t))
            print("VH max", len(self.V) -1)

            # if len(self.V) < self.hmax + 1:
            #     self.V.append(self.find_V(self.hmax))
            #print('s_t before modeling')
            #printl(self.s_t)


            if True:
                for node in a_t:
                    # Obtain mu_tau and sigma_tau of the node
                    mu_tau, sigma_tau = self.gp.inference(node.get_center())

                    if len(self.V) <= node.h:
                        self.V.append(self.find_V(node.h))
                        self.hmax += 1
                        self.hmax_len = node.hypercube_list[0].length

                    if node.h == 0:
                        V_h_1 = self.V[node.h]
                    else:
                        V_h_1 = self.V[node.h - 1]


                    mu_tau_parent, sigma_tau_parent = self.gp.inference(node.parent_node.get_center())
                    node.update_cumulative_conf_rect(mu_tau, sigma_tau, mu_tau_parent, sigma_tau_parent,
                                                     #self.beta[self.t],
                                                     self.find_beta(self.t),
                                                     self.V[node.h], V_h_1)

            #print('s_t')
            #printl(self.s_t)
            # print("a_t")
            # printl(a_t)

            print('s_t length before discard')
            print(len(self.s_t))

            "Discarding"
            print("Discarding")
            templist = set_diff(self.s_t, p_pess)
            # printl(self.s_t)
            # printl(p_pess)
            # print("set_diff(self.s_t, p_pess)------------------")
            # printl(templist)
            count = 0
            for node in templist:
                for pess_node in p_pess:
                    count += 1
                    if count <= 5:
                        print(node.R_t.upper, pess_node.R_t.lower)
                    if dominated_by(node.R_t.upper, pess_node.R_t.lower, self.epsilon):
                        print("dominated by")
                        self.s_t.remove(node)
                        break

            print('s_t length after discard')
            print(len(self.s_t))


            w_t = self.p_t + self.s_t  # The union of sets St and Pt at the end of the discarding phase of round t
            print('w_t')
            print(len(w_t))
            # printl(w_t)

            "Epsilon Covering"
            print("epsilon Covering")
            counter = 0
            for node in self.s_t:
                belongs = True
                for w_node in w_t:
                    if dominated_by(node.R_t.lower, w_node.R_t.upper,
                                    -self.epsilon):  # Doesn't belong to O_epsilon and therefore not removed
                        belongs = False
                        if counter < 5:
                            print(node.R_t.lower, w_node.R_t.upper, self.epsilon)
                        counter += 1
                        break
                if belongs:
                    print("belongs")
                    self.s_t.remove(node)
                    self.p_t.append(node)
            #print("count", count)

            print('s_t after e covering')
            print(len(self.s_t))


            "Refining / Evaluating"
            print("refining evaluating")
            if self.s_t:  # If s_t is not empty
                # print("look here")
                # print(np.array([node.R_t.diameter for node in w_t]))
                unc_node_ind = np.argmax(np.array([node.R_t.diameter for node in w_t]))
                unc_node = w_t[unc_node_ind]
                conf_diameter[self.t] = unc_node.R_t.diameter
                print("unc_node")
                print(unc_node)
                mu_unc, sigma_unc = self.gp.inference(unc_node.get_center())
                print(sigma_unc)

                sigmas[self.t] = np.linalg.norm(sigma_unc)

                condition = np.sqrt(self.find_beta(self.t)) * np.linalg.norm(sigma_unc) <= self.V[unc_node.h] * np.sqrt(self.problem_model.m)  # Norm V_h vector
                print("condition")
                print("beta", np.sqrt(self.find_beta(self.t)))
                print("sigma", np.linalg.norm(sigma_unc))
                print("beta*sigma", np.sqrt(self.find_beta(self.t)) * np.linalg.norm(sigma_unc))
                print("V", self.V[unc_node.h] * np.sqrt(self.problem_model.m))

                if condition and unc_node in self.s_t:
                    self.s_t.remove(unc_node)
                    # print(unc_node)
                    # print("reproduce")
                    repro = unc_node.reproduce()
                    # printl(repro)
                    self.s_t = self.s_t + repro
                    tau_change = False
                elif condition and unc_node in self.p_t:
                    self.p_t.remove(unc_node)
                    self.p_t = self.p_t + unc_node.reproduce()
                    tau_change = False
                else:
                    y = self.problem_model.observe(unc_node.get_center(), std= 0.05)
                    # Update GP parameters
                    self.gp.update(unc_node.get_center(), y)
                    self.t_tau.append(self.t)
                    self.tau += 1
                    tau_change = True

            self.t += 1
        plt.figure()
        ax = plt.axes()
        ax.plot(range(1,self.t-1), sigmas[1:self.t-1], label = r'$||\sigma_{\tau}(x_{h_t,i_t})||_2$')
        print("herere")
        print(self.t_tau)
        ax.scatter(self.t_tau, sigmas[self.t_tau], color = 'red', label=r"$\tau$")
        ax.set_xlabel('$t$')
        ax.set_ylabel(r'$||\sigma_{\tau}(x_{h_t,i_t})||_2$')
        ax.legend()
        plt.title(r"Posterior Variance after $\tau$ Evaluations")

        #print(conf_diameter[:self.t])

        plt.figure()
        ax = plt.axes()
        ax.plot(range(1,self.t-1), conf_diameter[1:self.t-1], label='$\omega_t(x_{h_t,i_t})$')
        ax.set_yscale('log')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$\omega_t(x_{h_t,i_t})$')
        plt.axhline(self.epsilon, color='red', label='$\epsilon$')
        ax.legend()
        plt.title("Diameter of the Cumulative Confidence \n Hyper-rectangle of the Most Uncertain Node")


        t2 = time.time()
        self.time_elapsed = t2-t1
        print("time")
        print(t2 - t1)
        pareto_cells = [node.hypercube_list for node in self.p_t]

        #node.hypercube_list[i]
        #for i in range(len(node.hypercube_list))
        #printl(self.p_t)

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
        card = self.problem_model.cardinality  # Cardinality of the design space.
        delta = self.delta

        return (2 / 9) * np.log(m * card * np.pi** 2 * tau ** 2 / (6 * delta))

    def find_beta_tau(self, tau):
        return 2*np.log(2 * self.problem_model.m * np.pi**2 * 2**10 * (tau+1)**2 / (3*self.delta))

    def find_V(self, h):
        v_1 = np.sqrt(2)
        rho = 0.5
        alpha = 1

        m = self.problem_model.m
        delta = self.delta
        N = self.problem_model.N
        D_1 = self.problem_model.D_1

        # Constants associated with metric dimension D1
        C_1 = np.sqrt(2 * self.gp.v) / self.gp.L
        print("c1", C_1)
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
        #print("vh for h", h, term1 * term2)
        return term2 * term1




