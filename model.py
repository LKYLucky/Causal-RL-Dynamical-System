import numpy as np
import scipy.optimize as so


class RateConstantModel():
    def __init__(self, num_species= 2, num_reactions=3, alpha=0.01, lamb=0.99, method='SLSQP', tol=1e-16):
        self.alpha = alpha
        self.lamb = lamb
        self.N = num_species
        self.R = num_reactions
        self.init_xi = np.zeros(num_reactions)
        self.method = method
        self.tol = tol

        #self.rates

    def compute_theta(self, Z):
        y1 = []
        for z in Z[:, 0]:
            y1.append(np.transpose([z, 0]))

        y1 = np.array(y1)

        y2 = []
        for z in Z[:, 1]:
            y2.append(np.transpose([0, -z]))
        y2 = np.array(y2)

        y3 = np.array(np.transpose([- Z[:, 0] * Z[:, 1], Z[:, 0] * Z[:, 1]]))
        theta = np.transpose([y1, y2, y3], (1, 0, 2))
        return theta


    def elastic_net_func(self, propensities,  Z_arr, theta_arr, alpha, lamb):

        num_species = self.N
        num_reactions = self.R

        result = 0
        total_time_steps = 0

        for i in range(len(theta_arr)):
            theta = theta_arr[i]
            time_steps = len(theta)
            Z = Z_arr[i]
            dZ = np.gradient(Z)  # Look into this later, returns an array of size 2x100x2, for now I just choose dZ[0]
            for t in range(time_steps):
                    for s in range(num_species):
                        x = dZ[0][t][s] #dZ
                        for r in range(num_reactions):
                            x -= propensities[r] * theta[t][r][s]
                        result += x**2

            total_time_steps += time_steps

        result *= 1.0 / (2.0 * total_time_steps)

        regulator = 0

        l1_regulator = 0
        for r in range(num_reactions):
            l1_regulator += abs(propensities[r])
        l1_regulator *= alpha * lamb
        regulator += l1_regulator

        if self.alpha != 0 and lamb < 1.0:
            l2_regulator = 0
            for r in range(num_reactions):
                l2_regulator += propensities[r]**2
            l2_regulator *= alpha * (1-lamb)
            regulator += l2_regulator

        return result + regulator

    def solve_minimize(self, Z_arr, theta_arr):
        def objective(x):
            obj = self.elastic_net_func(x, Z_arr, theta_arr, self.alpha, self.lamb)
            return obj

        result = so.minimize(
            objective,
            x0=self.init_xi,
            bounds=None,
            tol=self.tol,
            method=self.method,
            jac=None,
            options=None)
        return result

