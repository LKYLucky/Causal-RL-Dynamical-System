import numpy as np
import scipy.optimize as so


class RateConstantModel():
    def __init__(self, num_species= 2, num_reactions=3, alpha=0.0, lamb=1.0, method='SLSQP', tol=1e-16, approx_jac = False):
        self.alpha = alpha
        self.lamb = lamb
        self.N = num_species
        self.R = num_reactions
        self.rates = [0.1, 0.05, 0.05]
        self.init_xi = np.zeros_like(self.rates)
        self.method = method
        self.tol = tol
        self.approx_jac = approx_jac


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


    def elastic_net_func(self, propensities,  Z_arr, theta_arr, dt, alpha, lamb):

        num_species = self.N
        num_reactions = self.R

        result = 0
        total_time_steps = 0

        for i in range(len(theta_arr)):
            theta = theta_arr[i]
            time_steps = len(theta)
            Z = Z_arr[i] * int(1/dt)
            dZ = np.gradient(Z)  # Look into this later, returns an array of size 2x100x2, for now I just choose dZ[0] ###dt = 0.01 hard code 100
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

        if alpha != 0 and lamb < 1.0:
            l2_regulator = 0
            for r in range(num_reactions):
                l2_regulator += propensities[r]**2
            l2_regulator *= alpha * (1-lamb)
            regulator += l2_regulator

        return result + regulator

    def elastic_net_jac(self, propensities,  Z_arr, theta_arr, alpha, lamb):

        num_species = self.N
        num_reactions = self.R

        result = []
        total_time_steps = 0

        for i in range(len(theta_arr)):
            theta = theta_arr[i]
            time_steps = len(theta)
            Z = Z_arr[i]
            dZ = np.gradient(Z)  # Look into this later, returns an array of size 2x100x2, for now I just choose dZ[0]
            total_time_steps += time_steps
            for j in range(num_reactions):
                for t in range(time_steps):
                    for s in range(num_species):
                        x = dZ[0][t][s] #dZ
                        theta_t_j_s = theta[t][j][s]
                        for r in range(num_reactions):
                            x -= propensities[r] * theta[t][r][s]
                        result[j] += theta_t_j_s * x
            
        result /= -total_time_steps


        for j in range(num_reactions):
            #l1 regulator
            result[i] += alpha * lamb
            # l2 regulator
            result[i] += 2 * alpha * (1-lamb) * propensities[i]



    def solve_minimize(self, Z_arr, theta_arr, dt):
        def objective(x):
            obj = self.elastic_net_func(x, Z_arr, theta_arr, dt, self.alpha, self.lamb)
            return obj

        jac = False if self.approx_jac else \
            lambda x: self.elastic_net_jac(x, Z_arr, theta_arr, self.alpha, self.lamb)

        result = so.minimize(
            objective,
            x0=self.init_xi,
            bounds=None,
            tol=self.tol,
            method=self.method,
            jac=None,
            options=None)
        return result

