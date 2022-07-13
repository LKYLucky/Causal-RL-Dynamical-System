import numpy as np
from matplotlib import pyplot as plt
from env import LotkaVolterraEnv, BrusselatorEnv, GeneralizedEnv
from env_model import LotkaVolterraEnvModel, BrusselatorEnvModel, GeneralizedEnvModel

import torch
import scipy.optimize as so
from model import RateConstantModel

# import torch.nn.functional as F
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)

        # print('action probs:')
        # print(probs)

        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        return action


class Actor(torch.nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)
        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        return action


class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)

def get_z_init():

    prey = np.random.uniform(0.5, 1.5)
    pred = np.random.uniform(1, 3)
    Z_init = np.array([prey, pred])

    #Z_init = np.array([1,1])
    return Z_init


def uphill_policy(observation, critic):
    state = torch.tensor(observation, dtype=torch.float, requires_grad=True)
    critic(state).backward()
    u = state.grad.detach().numpy() * 0.1
    return u

def convert_to_vec(action):
    if action == 0:
        u = np.array([0, 0])
    elif action == 1:
        u = np.array([0, 1])
    elif action == 2:
        u = np.array([1, 0])
    return u


def discounted_rewards(rewards,R, gamma):
    returns = np.zeros_like(rewards)
    ### remove last 25 time steps
    for t in reversed(range(len(rewards))):
        R = R * gamma + rewards[t]
        returns[t] = R
    return returns


def update_policy_reinforce(states, actions, returns, model, optimizer):
    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.int)
    returns = torch.tensor(returns, dtype=torch.float)

    probs = model(states)  # https://pytorch.org/docs/stable/distributions.html
    m = torch.distributions.Categorical(probs=probs)
    log_prob = m.log_prob(actions)
    loss = (-log_prob * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer, critic_optimizer):
    states = torch.tensor(states, dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.float)
    returns = torch.tensor(returns, dtype=torch.float)

    probs = actor(states)  # https://pytorch.org/docs/stable/distributions.html
    policy = torch.distributions.Categorical(probs=probs)

    values = critic(states)

    advantages = returns - torch.reshape(values, (-1,))
    actor_loss = - (policy.log_prob(actions) * advantages.detach()).mean()
    print("actor_loss",actor_loss)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_loss = advantages.pow(2).mean()

    # loss for critic (MSE)

    critic_optimizer.zero_grad()
    critic_loss.backward()

    critic_optimizer.step()


def find_rate_constants(Z_arr, theta_arr,rc_model, action):

    #theta_arr = [theta] #only keep current value
    #Z_arr = [Z]

    '''
    if len(Z_arr) > 10:
        theta_arr = theta_arr[-10:]
        Z_arr = Z_arr[-10:]
    '''
    result = rc_model.solve_minimize(Z_arr, theta_arr, dt,action)
    rc_model.rates = result.x

    return result

def run_one_episode(env_option, max_step, algorithm, model, actor, critic, uphill, Z_arr, theta_arr, rc_model, calc_rate):

    states = []
    actions = []
    rewards = []

    Z_init = get_z_init()
    env_option.init_state = Z_init
    Z_history = np.expand_dims(Z_init, 0)

    observation = env_option.reset()
    rc_list = []
    for i in range(max_step):  # while not done:

        if algorithm == "reinforce":
            action = model.select_action(observation)
        elif algorithm == "a2c":
            action = actor.select_action(observation)
            if uphill:
                u = uphill_policy(observation, critic)

        if not uphill:
            u = convert_to_vec(action)

        obs, reward, _, _, Z = env_option.step(u) ##add gaussian noise
        for j in range(len(Z)):
            mu, sigma = 0, 0.001  # mean and standard deviation
            s = np.random.normal(mu, sigma)
            Z[j] = Z[j]+s


        Z_history = np.concatenate((Z_history, Z), 0)
        states.append(observation)
        observation = obs
        actions.append(action)
        rewards.append(reward)


        if calc_rate: #updating every 10 time steps
            theta = rc_model.compute_theta(Z, env.species_constants)
            theta_arr.append(theta)
            Z_arr.append(Z)



            #rc_list.append(estimated_rates)

    #print(rc_list)
    if calc_rate:
        '''
        result = find_rate_constants(Z, Z_arr, theta_arr, rc_model, env_option.u)
        estimated_rates = result.x.tolist()
        print(estimated_rates)
        '''
        result = find_rate_constants(Z_arr, theta_arr, rc_model, env_option.u)
        estimated_rates = result.x.tolist()
        print("estimated_rates", estimated_rates)
        return rewards, states, observation, actions, Z_history, Z, estimated_rates
    else:
        return rewards, states, observation, actions, Z_history, Z, None

def run(env, env_model, ODE_env, algorithm, uphill):
    model = Model()
    actor = Actor()
    critic = Critic()

    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    print("obs space", env.observation_space)
    print("action_space", env.action_space.n)

    # train
    max_episode = 400
    n_episode = 0
    max_step = 200
    N = 10
    scores = []
    prob_list = []
    states_list = []
    returns_list = []
    '''
    theta_arr = []
    Z_arr = []
    '''
    if ODE_env == "LV":
        rc_model = RateConstantModel(rates = [0, 0, 0], ODE_env = "LV")
    elif ODE_env == "Brusselator":
        rc_model = RateConstantModel(num_reactions=4, rates = [0, 0, 0, 0], ODE_env = "Brusselator")

    elif ODE_env == "Generalized":
        rc_model = RateConstantModel(num_reactions=6, rates = [0, 0, 0, 0, 0, 0],ODE_env = "Generalized") #LV
    while n_episode < max_episode:
        theta_arr = []
        Z_arr = []
        print('starting training episode %d' % n_episode)

        if ODE_env == "LV":
            env.rate_constants = [0.1, 0.05, 0.05]  # LV
        elif ODE_env == "Brusselator":
            env.rate_constants = [1, 1, 1, 1]  # Brusselator

        elif ODE_env == "Generalized":
            env.rate_constants = [0.1, 0.05, 0.05, 0, 0, 0]  #LV

        if n_episode % N == 0:
            env_option = env
            calc_rate = True
        else:
            env_option = env_model
            calc_rate = False

        rewards, states, observation, actions, Z_history, Z, estimated_rates = run_one_episode(env_option, max_step,
                                                                                               algorithm, model,
                                                                                               actor, critic,
                                                                                               uphill, Z_arr, theta_arr, rc_model, calc_rate)

        if n_episode % N == 0:
            env_model.rate_constants = estimated_rates

        #print("rate constant", env_option.rate_constants)

        x = torch.tensor([1, 2], dtype=torch.float)
        y = model.forward(x)
        y = y.tolist()
        prob_list.append(y)

        scores.append(sum(rewards))
        # prob_list.append([zero_prob, one_prob, two_prob])
        n_episode += 1

        if algorithm == "reinforce":
            R = 0
        elif algorithm == "a2c":
            R = critic(torch.tensor(observation, dtype=torch.float)).detach().numpy()[0]

        returns = discounted_rewards(rewards, R, gamma=0.4)
        states = states[:-25]
        actions = actions[:-25]
        returns = returns[:-25]
        if algorithm == "reinforce":
            update_policy_reinforce(states, actions, returns, model, optimizer)
        elif algorithm == "a2c":
            update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer, critic_optimizer)

        for s in range(len(states)):
            states_list.append(states[s].tolist())
            returns_list.append(returns[s])

    #eval -- let's make this a separate function, analogous to 'run' but without any training or policy updating
    #done = False
    theta_arr = []
    Z_arr = []


    rewards, states, observation, actions, Z_history, Z, estimated_rates= run_one_episode(env, max_step, algorithm, model, actor, critic, uphill, Z_arr, theta_arr, rc_model, False)
    #result = find_rate_constants(Z, Z_arr, theta_arr, rc_model, env.u)
    #print("result", result)

    x = np.arange(0, 2,0.02)
    y = np.arange(0, 4,0.04)
    X, Y = np.meshgrid(x, y)

    Z = []
    for i in x:
        Z_vec = []
        for j in y:
             obs = torch.tensor([i, j], dtype=torch.float)
             z = critic.forward(obs).detach().numpy()[0]
             Z_vec.append(z)
        Z.append([Z_vec])


    Z = np.array(Z).reshape(100, 100)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel("Prey")
    ax.set_ylabel("Predators")
    plt.savefig('./Z_arr_contour_plot.png')

    V = []
    for i in x:
        V_vec = []
        for j in y:
             obs = torch.tensor([i, j], dtype=torch.float)
             val = critic(obs)
             V_vec.append(val.detach().numpy())
        V.append([V_vec])

    V = np.array(V).reshape(100, 100)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, V)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_xlabel("Prey")
    ax.set_ylabel("Predators")
    plt.savefig('./Value_arr_contour_plot.png')


    X = []
    Y = []
    for s in states_list:
        X.append(s[0])
        Y.append(s[1])

    X = np.array(X)
    Y = np.array(Y)
    returns_list = np.array(returns_list)

    fig, ax = plt.subplots()

    z = ax.tricontour(X, Y, returns_list,20)
    fig.colorbar(z)
    ax.tricontour(X, Y, returns_list,20)

    ax.plot(X, Y)
    ax.set_xlabel("Prey")
    ax.set_ylabel("Predators")
    plt.savefig('./returns.png')

    fig, ax = plt.subplots()
    ax.plot(Z_history[:, 0], (Z_history[:, 1]))
    ax.set_xlabel("Prey")
    ax.set_ylabel("Predators")
    plt.savefig('./Z_history_contour_plot.png')


    plt.figure(1)
    plt.cla()
    scores = scores[::10]
    print("scores model based high noise", scores)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('./rewards.png')
    #plt.show()

    tt = np.linspace(0, max_step, Z_history.shape[0])
    plt.cla()
    plt.plot(tt, Z_history[:, 0], 'b', label='prey')
    plt.plot(tt, Z_history[:, 1], 'r', label='predators')
    plt.plot(tt, Z_history[:, 0]/Z_history[:, 1], 'k', label='ratio')
    plt.legend(shadow=True, loc='upper right')
    plt.xlabel('t')
    plt.ylabel('n')
    plt.savefig('./state_trajectory.png')
    #plt.show()


    tt = np.linspace(0, max_episode, max_episode)
    plt.cla()
    zeros_prob = [item[0] for item in prob_list]
    ones_prob = [item[1] for item in prob_list]
    twos_prob = [item[2] for item in prob_list]
    plt.plot(tt, zeros_prob, 'kx', label='Item 0')
    plt.plot(tt, ones_prob, 'rx', label='Item 1')
    plt.plot(tt, twos_prob, 'o', label='Item 2')
    plt.legend(shadow=True, loc='lower right')
    plt.xlabel('Episode #')
    plt.ylabel('Probabilities')
    plt.savefig('./policy_probs.png')

    plt.show()

    exit()
    env.close()

N = 2 # number of species
tau = 1
dt = 0.02#1e-2

#ODE_env = "LV"
#ODE_env = "Brusselator"
ODE_env = "Generalized"
if ODE_env == "LV":
    env = LotkaVolterraEnv(N, tau, dt)
    env_model =  LotkaVolterraEnvModel(N, tau, dt)
elif ODE_env == "Brusselator":
    env = BrusselatorEnv(N, tau, dt)
    env_model = BrusselatorEnvModel(N, tau, dt)
elif ODE_env == "Generalized":
    env = GeneralizedEnv(N, tau, dt)
    env_model = GeneralizedEnvModel(N, tau, dt)

#run(env, algorithm = "reinforce")
run(env, env_model, ODE_env, algorithm = "a2c", uphill = True)
#run(env, algorithm = "optimal policy")