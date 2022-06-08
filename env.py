import numpy as np
from scipy.integrate import odeint
import gym
    

# we could generalize/modify this class to take a more generic ODE function as input, or generalise self.f(...)

class LotkaVolterraEnv(gym.Env):

    def __init__(self, num_species=2, time_interval_action=1, dt=1e-3, init_state = np.array([1.0, 1.5])):

        # may need to add more here

        low = np.zeros((num_species), dtype=np.float32)
        high = np.array([np.finfo(np.float32).max]*num_species, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_species+1)
        #self.action_space = gym.spaces.Box(low, high, dtype=np.float32) ##replace with gym.spaces.discrete
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.N = num_species
        self.tau = time_interval_action # we'd need to modify this if we consider irregular time intervals between observations and/or actions
        self.dt = dt

        self.init_state = init_state
        theta_lin = np.array([0.1, -0.05]) # np.ones((N,))
        theta_quad = np.zeros((num_species,num_species))
        theta_quad[0,1] += -0.05
        theta_quad[1,0] += 0.05
        self.theta = [theta_lin, theta_quad] # there may be a better way to collectively denote the parameters
        
    def f(self, Z, t):
   
        # linear terms
        Zdot = np.multiply(Z, self.theta[0])
        # quadratic terms
        Zdot += np.multiply(Z, np.einsum('ij,j->i', self.theta[1], Z))
        # control terms
        Zdot += self.u

        return Zdot

    def step(self, action):
        'integrates the ODE system with a constant additive force vector (action)'

        tt = np.linspace(0, self.tau, int(1/self.dt))

        self.u = action
        z_init = self.state
        z = odeint(self.f, z_init, tt)
        self.state = z[-1]

        #reward = -self.state[0]
        #add prey
        if self.u[0] ==0 and  self.u[1] == 1:
            self.state[0] += 0.1
        #add predator
        elif self.u[0] ==1 and self.u[1] == 0:
            self.state[1] += 0.1

        #do nothing

        ratio_optimal = 1
        reward = (ratio_optimal - self.state[0]/self.state[1])**2
        
        done = False # indicates whether the episode is terminated; optional
        info = {} # can be used in custom Gym environments; optional

        # we assume z is observed with zero noise
        #obs = z +
        obs = self.state
        return self.state, reward, done, info, z#return obs, reward, done, info

    def reset(self):
        'resets the ODE system to initial conditions'
        #raise NotImplementedError
        self.state = self.init_state

        return self.state

