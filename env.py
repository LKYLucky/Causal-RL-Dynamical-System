import numpy as np
from scipy.integrate import odeint
import gym


# we could generalize/modify this class to take a more generic ODE function as input, or generalise self.f(...)

class ODEBaseEnv(gym.Env):

    def __init__(self, num_species=2, time_interval_action=1, dt=1e-3, init_state=np.array([1.0, 1.5])):
        # may need to add more here

        low = np.zeros((num_species), dtype=np.float32)
        high = np.array([np.finfo(np.float32).max] * num_species, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_species + 1)
        # self.action_space = gym.spaces.Box(low, high, dtype=np.float32) ##replace with gym.spaces.discrete
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.N = num_species
        self.tau = time_interval_action  # we'd need to modify this if we consider irregular time intervals between observations and/or actions
        self.dt = dt

        self.init_state = init_state
        #
        theta_lin = np.array([0.1, -0.05])  # np.ones((N,))
        theta_quad = np.zeros((num_species, num_species))
        theta_quad[0, 1] += -0.05
        theta_quad[1, 0] += 0.05
        self.theta = [theta_lin, theta_quad]  # there may be a better way to collectively denote the parameters


    def compute_orth(self, Z, t):
        # linear terms
        Zdot = self.f(Z, t, remove_u = True)
        orth = np.empty_like(Zdot)
        orth[0] = -Zdot[1]
        orth[1] = Zdot[0]

        return orth

    def step(self, action):
        'integrates the ODE system with a constant additive force vector (action)'
        'returns a reward based on the ODE variables at the *end* of this time interval'

        tt = np.linspace(0, self.tau, int(1 / self.dt))
        state_prev = self.state

        self.u = 0.1 * action
        z_init = self.state
        z = odeint(self.f, z_init, tt)
        self.state = z[-1]
        state_curr = self.state

        # print("state_prev",state_prev,"state_curr",state_curr)
        ratio_optimal = 1
        # reward = 1 - (ratio_optimal - self.state[0]/self.state[1])**2
        reward = -((state_prev - state_curr) ** 2).sum()
        # reward = -(state_prev[0] - state_curr[0])**2-(state_prev[1] - state_curr[1])**2
        done = False  # indicates whether the episode is terminated; optional
        info = {}  # can be used in custom Gym environments; optional

        # we assume z is observed with zero noise
        obs = self.state
        ### I can only return 4 arguments for the baseline a2c
        #return obs, reward, done, info
        return obs, reward, done, info, z

    def reset(self):
        'resets the ODE system to initial conditions'
        self.state = self.init_state

        return self.state

class LotkaVolterraEnv(ODEBaseEnv):

    def f(self, Z, t, remove_u = False):
        # linear terms
        Zdot = np.multiply(Z, self.theta[0])
        # quadratic terms
        Zdot += np.multiply(Z, np.einsum('ij,j->i', self.theta[1], Z))
        # control terms
        if not remove_u:
            Zdot += self.u
        return Zdot

class BrusselatorEnv(ODEBaseEnv):

    def f(self, Z, t):
        A = 1
        B = 1.7
        k1 = 1
        k2 = 1
        k3 = 1
        k4 = 1
        X, Y = Z
        Zdot = [k1*A + k2*X**2*Y - k3*B*X - k4*X, - k2*X**2*Y+ k3*B*X]
        Zdot += self.u
        return Zdot